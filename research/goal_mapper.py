"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames into goal latents).
3. Implements the Planning Protocol (.predict and .get_cost) for any CEM/MPC solver.
"""

import torch
import sys
from pathlib import Path
from torchvision.transforms import v2 as transforms

# Project paths
# Note: Root and le_wm paths are added by the calling scripts (diagnose/eval)

# Project-specific imports
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP
from research.goal_utils import get_goal_pixels, get_episode_video_path


class GoalMapper:
    """
    Utility to map task success (last frame of dataset) to World Model latent embeddings.
    Used for Zero-Shot Goal-Conditioned MPC.
    """

    def __init__(self, model_path, dataset_root, img_size=224):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_root = Path(dataset_root)
        self.img_size = img_size

        # Standard JEPA Transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            ]
        )

        # Initialize Model Architecture (v17 Hardcoded Abstractions)
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

        # Load Weights
        print(f"🧠 Loading Oracle Weights: {Path(model_path).name}")
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
        video_path = get_episode_video_path(self.dataset_root, episode_idx)
        pixels = get_goal_pixels(video_path)

        if pixels is None:
            return False

        # Preprocess: (3, H, W) -> (1, 1, 3, 224, 224)
        pixels = self.transform(pixels).to(self.device)
        pixels = pixels.unsqueeze(0).unsqueeze(0)

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
            # Final state latent: (B, D) or (B, T, D)
            last_latent = output["emb"][:, -1, :]

            # Use L2 Distance to internal goal latent
            # goal_latent is (1, 1, D) -> squash to (D)
            dist = torch.norm(last_latent - self.goal_latent.view(-1), dim=-1)
            return dist
