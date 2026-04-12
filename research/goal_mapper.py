import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP


class GoalMapper:
    """
    Utility to map 3D task coordinates to World Model latent embeddings.
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
    def find_goal_latent(self, target_xyz):
        """
        1. Find a frame in the dataset where the cube is at target_xyz.
        2. Encode that frame to get the goal latent z_goal.
        """
        from research.goal_utils import find_goal_pixels

        pixels = find_goal_pixels(target_xyz, None, self.dataset_root, self.img_size)
        if pixels is None:
            return None

        pixels = pixels.to(self.device)
        # JEPA.encode expects dict with 'pixels'
        # shape (B, T, C, H, W) -> (1, 1, 3, 224, 224)
        info = self.model.encode({"pixels": pixels})
        return info["emb"]  # (1, 1, D)
