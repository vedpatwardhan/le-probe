import os
import sys
import torch
import zmq
import msgpack
import json
import numpy as np
from pathlib import Path

# --- Path Stabilization ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LEWM_DIR = os.path.join(PROBE_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

for p in [PROBE_DIR, LEWM_DIR, LE_WM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
# --------------------------

from lewm.goal_mapper import GoalMapper
from interpretability.sae.sae_model import SparseAutoencoder
from interpretability.clt.clt_model import CrossLayerTranscoder
from gr1_protocol import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PORT = 5557


class LatentExplorerServer:
    def __init__(self, model_path, sae_path, clt_path, labels_path):
        print("--- Initializing Latent Explorer Server ---")
        self.scaler = StandardScaler()
        self.device = DEVICE

        # 1. Load LeWM (Predictor)
        self.agent = GoalMapper(model_path, dataset_root=".")
        self.model = self.agent.model.to(DEVICE).eval()

        # 2. Load SAE (Definitive Format from train_sae.py)
        # Note: dict_size=1024, d_model=192 as per report
        self.sae = SparseAutoencoder(d_model=192, d_sae=1024).to(DEVICE).eval()
        sae_checkpoint = torch.load(sae_path, map_location=DEVICE)
        self.sae.load_state_dict(sae_checkpoint["state_dict"])

        # 3. Load CLT (Definitive Format from train_clt.py)
        clt_checkpoint = torch.load(clt_path, map_location=DEVICE)
        self.clt = CrossLayerTranscoder(d_model=192, d_sae=1024).to(DEVICE).eval()
        self.clt.load_state_dict(clt_checkpoint["state_dict"])
        self.clt_norm = clt_checkpoint["norm_stats"]  # <--- Bit-Perfect Normalization

        self.labels_path = Path(labels_path)
        self.load_labels()

        # State tracking
        self.current_z_enc = None

    def load_labels(self):
        if self.labels_path.exists():
            with open(self.labels_path, "r") as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

    def save_labels(self):
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f, indent=4)

    @torch.no_grad()
    def get_activations(self, image_np, action_np):
        """
        Calculates activations for the 'Imagined' state.
        Pipeline: Image -> Encoder -> z_enc -> Predictor(action) -> z_pred -> CLT -> Acts
        """
        # 1. Image -> z_enc
        batch = self.agent.transform({"pixels": image_np})
        pixels = (
            batch["pixels"].to(self.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, 3, 224, 224)
        info = self.model.encode({"pixels": pixels})
        z_enc = info["emb"]  # (1, 1, 192)

        # 2. Action -> z_pred
        # We manually run the predictor and use the ENCODER projector (self.model.projector)
        # to ensure we stay in the CLT's training manifold.
        action_t = (
            torch.tensor(action_np, device=self.device)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        act_emb = self.model.action_encoder(action_t)

        # Predict but DON'T use model.pred_proj (which might differ from model.projector)
        raw_pred = self.model.predictor(z_enc, act_emb)
        z_pred = self.model.projector(raw_pred.squeeze(0)).unsqueeze(
            0
        )  # Use Encoder Projector

        # 3. Dual-Path Audit: Visual (z_enc) + Predicted (z_pred)
        # This ensures we catch signals even if the predictor has no temporal context yet.
        z_enc_flat = z_enc.squeeze(0).squeeze(0).unsqueeze(0)
        z_pred_flat = z_pred.squeeze(0).squeeze(0).unsqueeze(0)

        # Apply normalization to both
        mean = self.clt_norm["mean_L"].to(self.device)
        std = self.clt_norm["std_L"].to(self.device)

        z_enc_norm = (z_enc_flat - mean) / std
        z_pred_norm = (z_pred_flat - mean) / std

        # Get activations for both
        acts_visual = self.clt(z_enc_norm)["activations"].squeeze(0)
        acts_intent = self.clt(z_pred_norm)["activations"].squeeze(0)

        # Take the maximum signal across both paths for the UI
        acts_combined = torch.max(acts_visual, acts_intent)

        return acts_combined.cpu().numpy()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{PORT}")
        print(f"🚀 Latent Server LISTENING on port {PORT}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)
                cmd = req.get("command")

                if cmd == "get_activations":
                    image = np.frombuffer(
                        req["image"]["data"], dtype=req["image"]["dtype"]
                    ).reshape(req["image"]["shape"])
                    action = np.array(req["action"], dtype=np.float32)

                    acts = self.get_activations(image, action)

                    # Get Top 15
                    top_indices = np.argsort(acts)[-15:][::-1]
                    top_features = []
                    for idx in top_indices:
                        val = float(acts[idx])
                        if val > 0:
                            label = self.labels.get(str(idx), None)
                            top_features.append([int(idx), val, label])

                    socket.send(
                        msgpack.packb(
                            {
                                "status": "ok",
                                "top_features": top_features,
                                "activations": acts.tolist(),
                                "labels": self.labels,
                            }
                        )
                    )

                elif cmd == "update_label":
                    fid = str(req["feature_id"])
                    label = req["label"]
                    self.labels[fid] = label
                    self.save_labels()
                    socket.send(msgpack.packb({"status": "ok"}))

                else:
                    socket.send(
                        msgpack.packb({"status": "error", "message": "Unknown command"})
                    )

            except Exception as e:
                print(f"❌ Server Error: {e}")
                import traceback

                traceback.print_exc()
                socket.send(msgpack.packb({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--sae", type=str, default="sae_weights.pt")
    parser.add_argument("--clt", type=str, default="clt_weights.pt")
    parser.add_argument(
        "--labels", type=str, default="le-probe/interpretability/feature_labels.json"
    )

    def resolve_path(p):
        if os.path.isabs(p):
            return p
        if os.path.exists(p):
            return os.path.abspath(p)
        # Try relative to project root (two levels up from this script)
        fallback = os.path.join(PROBE_DIR, "..", p)
        if os.path.exists(fallback):
            return fallback
        # Try relative to the script's le-probe root
        fallback_probe = os.path.join(PROBE_DIR, p)
        if os.path.exists(fallback_probe):
            return fallback_probe
        return os.path.abspath(p)

    args = parser.parse_args()

    m_path = resolve_path(args.model)
    s_path = resolve_path(args.sae)
    c_path = resolve_path(args.clt)
    l_path = resolve_path(args.labels)

    server = LatentExplorerServer(m_path, s_path, c_path, l_path)
    server.run()
