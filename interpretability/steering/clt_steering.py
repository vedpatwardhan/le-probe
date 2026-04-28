import torch
import torch.nn.functional as F
from typing import Dict, Optional

from interpretability.clt.clt_model import CrossLayerTranscoder


class CLTSteerer:
    """
    Cross-Layer Steering for LeWorldModel.
    Intervenes on the handoff between Encoder and Predictor using CLT features.
    """

    def __init__(self, model: torch.nn.Module, clt_path: str, device: str = "cpu"):
        self.model = model
        self.device = device

        # 1. Load CLT and Stats
        checkpoint = torch.load(clt_path, map_location=device)
        cfg = checkpoint["config"]
        self.norm = checkpoint["norm_stats"]

        self.clt = CrossLayerTranscoder(d_model=cfg["d_model"], d_sae=cfg["d_sae"]).to(
            device
        )
        self.clt.load_state_dict(checkpoint["state_dict"])
        self.clt.eval()

        self.interventions = {}  # {feature_idx: boost_factor}
        self.hook = None

    def set_boost(self, feature_idx: int, factor: float):
        """Sets a multiplication factor for a specific CLT feature."""
        self.interventions[feature_idx] = factor

    def _steering_hook(self, module, input, output):
        """
        Intervention Hook:
        Maps Encoder Output -> (CLT + Boost) -> Predictor Input
        """
        # output is the Encoder latent (B, 192)
        x = output

        # 1. Normalize
        x_norm = (x - self.norm["mean_L"].to(self.device)) / self.norm["std_L"].to(
            self.device
        )

        # 2. Project to CLT Sparse Space
        # (Cloning clt.forward logic to allow interventions on 'acts')
        x_centered = x_norm - self.clt.b_dec
        acts = F.relu(self.clt.encoder(x_centered) + self.clt.b_enc)

        # 3. Apply Boosts (e.g. Neuron 90 *= 5.0)
        for idx, factor in self.interventions.items():
            acts[..., idx] *= factor

        # 4. Reconstruct and Un-normalize to Target Space
        x_transcoded = self.clt.decoder(acts) + self.clt.b_dec
        x_steered = (x_transcoded * self.norm["std_T"].to(self.device)) + self.norm[
            "mean_T"
        ].to(self.device)

        return x_steered

    def attach(self, layer_path: str = "projector"):
        """
        Attaches to the model. Defaults to 'projector' which is the
        final stage of the LeWM Encoder.
        """
        module = self.model
        for part in layer_path.split("."):
            module = getattr(module, part)

        self.hook = module.register_forward_hook(self._steering_hook)
        print(
            f"🚀 CLT Steering attached to {layer_path}. Boosts active: {self.interventions}"
        )

    def detach(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
            print("🛑 Steering detached.")


if __name__ == "__main__":
    print("🧠 CLT Steerer Module Loaded.")
