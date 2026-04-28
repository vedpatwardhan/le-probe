import torch
from typing import Dict, List, Optional
import numpy as np


class ActivationHarvester:
    """
    Hook manager to collect intermediate activations from a PyTorch model.
    Designed for Zero-Impact modularity (no model code changes).
    """

    def __init__(self):
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []

    def hook_fn(self, name: str):
        def fn(module, input, output):
            # If output is a tuple (e.g. from some RNNs or Transformers), take the first element
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            # Detach and move to CPU to avoid VRAM bloat during harvesting
            self.activations[name].append(out.detach().cpu())

        return fn

    def register(self, model: torch.nn.Module, layers: Dict[str, str]):
        """
        Registers hooks based on a mapping of {friendly_name: module_path}.
        Example: {"predictor_mlp": "predictor.transformer.layers.5.mlp"}
        """
        for name, path in layers.items():
            self.activations[name] = []

            # Navigate to the target submodule
            module = model
            for part in path.split("."):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)

            # Register the hook
            h = module.register_forward_hook(self.hook_fn(name))
            self.hooks.append(h)
            print(f"✅ Registered hook for: {name} ({path})")

    def flush(self) -> Dict[str, torch.Tensor]:
        """Concatenates all collected activations into single tensors."""
        results = {}
        for name, data in self.activations.items():
            if data:
                results[name] = torch.cat(data, dim=0)
                self.activations[name] = []  # Clear memory
        return results

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
