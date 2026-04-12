"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames into goal latents).
3. Implements the Planning Protocol (.predict and .get_cost) for any CEM/MPC solver.
"""

import torch
from pathlib import Path
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP

# Project-specific imports
from research.goal_utils import get_goal_pixels


class GoalMapper:
    """
    Utility to map task success (last frame of dataset) to World Model latent embeddings.
    Used for Zero-Shot Goal-Conditioned MPC.
    """

    def __init__(self, model_path, dataset_root, img_size=224):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_root = Path(dataset_root)
        self.img_size = img_size

        # Initialize Model
        self.encoder = spt.backbone.utils.vit_hf(
            "tiny", patch_size=14, image_size=224, pretrained=False
        )
        self.predictor = ARPredictor(
            num_frames=3,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
        )
        self.action_encoder = GR1Embedder(input_dim=64, emb_dim=192)
        self.projector = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)
        self.predictor_proj = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)

        self.model = (
            JEPA(
                encoder=self.encoder,
                predictor=self.predictor,
                action_encoder=self.action_encoder,
                projector=self.projector,
                pred_proj=self.predictor_proj,
            )
            .to(self.device)
            .eval()
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_sd, strict=False)

    @torch.no_grad()
    def set_goal(self, episode_idx=0):
        """
        Fetches the success state (last frame) from the dataset and
        stores its latent embedding internally as the planning target.
        """
        pixels = get_goal_pixels(self.dataset_root, episode_idx, self.img_size)
        if pixels is None:
            return False

        pixels = pixels.to(self.device)
        info = self.model.encode({"pixels": pixels})
        self.goal_latent = info["emb"]  # (1, 1, D)
        return True

    def predict(self, *args, **kwargs):
        """Proxy to the internal World Model's prediction logic."""
        return self.model.predict(*args, **kwargs)

    def get_cost(self, obs_dict, actions):
        """
        Calculates the latent distance between the final state of an imagined
        trajectory and the stored goal.
        """
        assert hasattr(self, "goal_latent"), "Goal not set. Call set_goal() first."

        with torch.no_grad():
            output = self.predict(obs_dict, actions)
            # Final state latent: (B, D)
            last_latent = output["emb"][:, -1, :]
            # L2 Distance to internal goal latent
            dist = torch.norm(last_latent - self.goal_latent, dim=-1)
            return dist
