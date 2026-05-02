import json
import os
import torch
import zmq
from interpretability.transcoders.universal_transcoder import Transcoder


class LatentServer:
    def __init__(
        self,
        agent,
        sae_path="le-probe/interpretability/transcoders/sae_weights.pt",
        clt_path="le-probe/interpretability/transcoders/clt_weights.pt",
        labels_path="le-probe/interpretability/transcoders/feature_labels.json",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.agent = agent
        self.model = agent.model
        self.device = device
        self.labels_path = labels_path

        # 1. Load SAE (Perception Lens - Identity L0 -> L0)
        sae_data = torch.load(sae_path, map_location=device)
        self.sae = Transcoder(
            d_model=sae_data.get("input_dim", 192),
            d_dict=sae_data.get("dict_size", 12288),
        ).to(device)
        self.sae.load_state_dict(sae_data["state_dict"])
        self.sae_stats = sae_data["norm_stats"]
        self.sae.eval()

        # 2. Load CLT (Intent Proxy - Transition L0 -> Predictor)
        clt_data = torch.load(clt_path, map_location=device)
        self.clt = Transcoder(
            d_model=clt_data["config"]["d_model"],
            d_dict=clt_data["config"]["d_sae"],
            l1_coeff=clt_data["config"]["l1_coeff"],
        ).to(device)
        self.clt.load_state_dict(clt_data["state_dict"])
        self.clt_stats = clt_data["norm_stats"]
        self.clt.eval()

        # 3. Load Labels
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

    def save_labels(self):
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f, indent=4)

    @torch.no_grad()
    def get_activations(self, image_np, action_np):
        """
        Calculates activations using a Dual-Probe architecture:
        1. Perception (SAE) -> [0:12288]
        2. Intention (CLT)  -> [12288:24576]
        """
        # 1. Image -> z_enc
        batch = self.agent.transform({"pixels": image_np})
        pixels = (
            batch["pixels"].to(self.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, 3, 224, 224)
        info = self.model.encode({"pixels": pixels})
        z_enc = info["emb"]  # (1, 1, 192)
        z_enc_flat = z_enc.squeeze(0).squeeze(0).unsqueeze(0)

        # 2. Action -> z_pred (Imagination)
        action_t = (
            torch.tensor(action_np, device=self.device)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        act_emb = self.model.action_encoder(action_t)
        raw_pred = self.model.predictor(z_enc, act_emb)
        z_pred = self.model.projector(raw_pred.squeeze(0)).unsqueeze(0)
        z_pred_flat = z_pred.squeeze(0).squeeze(0).unsqueeze(0)

        # 3. Path A: Perception (SAE)
        mean_s = self.sae_stats["mean"].to(self.device)
        std_s = self.sae_stats["std"].to(self.device)
        z_enc_norm_s = (z_enc_flat - mean_s) / (std_s + 1e-6)
        sae_out = self.sae(z_enc_norm_s)
        acts_perceptual = sae_out["activations"].squeeze(0)

        # Path B: Intentional Imagination (CLT)
        z_enc_clt_norm = (z_enc_flat - self.clt_stats["mean_L"].to(self.device)) / (
            self.clt_stats["std_L"].to(self.device) + 1e-6
        )
        clt_out = self.clt(z_enc_clt_norm)  # In inference, we only need source L
        acts_intentional = clt_out["activations"].squeeze(0)

        # Unified 24k Feature Map (Concatenation to avoid collisions)
        acts_combined = torch.cat([acts_perceptual, acts_intentional], dim=-1)

        return acts_combined.cpu().numpy()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        print("Ready to audit latents.")

        while True:
            msg = socket.recv_pyobj()
            if msg["type"] == "get_acts":
                acts = self.get_activations(msg["image"], msg["action"])
                socket.send_pyobj(acts)
            elif msg["type"] == "save_label":
                self.labels[str(msg["index"])] = msg["label"]
                self.save_labels()
                socket.send_pyobj({"status": "ok"})
