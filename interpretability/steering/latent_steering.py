import torch
from typing import Dict, List, Optional
import torch.nn.functional as F


class LatentSteerer:
    """
    Enables Feature Steering and Ablation during a model's forward pass.
    Uses trained SAEs to intervene in the latent space.
    """

    def __init__(self, model: torch.nn.Module, sae_map: Dict[str, str]):
        """
        model: The target PyTorch model (e.g. JEPA)
        sae_map: Mapping of {layer_name: sae_path}
        """
        self.model = model
        self.saes = {}
        self.active_interventions = {}  # {layer_name: {feature_idx: value}}
        self.hooks = []

        # Load SAEs
        for name, path in sae_map.items():
            checkpoint = torch.load(path, map_location="cpu")
            # Reconstruct SAE (assumes sae_model structure)
            from interpretability.sae.sae_model import SparseAutoencoder

            cfg = checkpoint["config"]
            sae = SparseAutoencoder(d_model=cfg["d_model"], d_sae=cfg["d_sae"])
            sae.load_state_dict(checkpoint["state_dict"])
            sae.eval().to(next(model.parameters()).device)
            self.saes[name] = sae

    def set_intervention(self, layer_name: str, feature_idx: int, value: float):
        """
        Sets a target value for a specific sparse feature.
        value = 0.0 for ablation.
        value > original for amplification.
        """
        if layer_name not in self.active_interventions:
            self.active_interventions[layer_name] = {}
        self.active_interventions[layer_name][feature_idx] = value

    def clear_interventions(self, layer_name: Optional[str] = None):
        if layer_name:
            self.active_interventions.pop(layer_name, None)
        else:
            self.active_interventions = {}

    def steering_hook(self, name: str):
        def fn(module, input, output):
            if name not in self.saes or name not in self.active_interventions:
                return output

            sae = self.saes[name]
            interventions = self.active_interventions[name]

            # 1. Project to sparse space
            x = output
            if isinstance(x, tuple):
                x = x[0]

            x_centered = x - sae.b_dec
            acts = F.relu(sae.encoder(x_centered) + sae.b_enc)

            # 2. Apply Interventions
            for idx, val in interventions.items():
                acts[..., idx] = val

            # 3. Reconstruct steered latent
            x_steered = sae.decoder(acts) + sae.b_dec

            if isinstance(output, tuple):
                return (x_steered,) + output[1:]
            return x_steered

        return fn

    def attach(self, layer_paths: Dict[str, str]):
        """
        Attaches steering hooks to the model.
        layer_paths: {layer_name: dot.path.to.module}
        """
        for name, path in layer_paths.items():
            module = self.model
            for part in path.split("."):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)

            h = module.register_forward_hook(self.steering_hook(name))
            self.hooks.append(h)
            print(f"🚀 Steering attached to: {name}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
