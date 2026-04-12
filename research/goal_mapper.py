"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames once).
3. Implements high-performance windowed rollouts with VRAM de-duplication.
"""

import torch
import sys
from pathlib import Path
from einops import rearrange

# Project-specific imports
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP
from le_wm.utils import get_img_preprocessor
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

        # 🎯 OFFICIAL TRAINING TRANSFORMS (Strict Parity)
        self.transform = get_img_preprocessor(
            source="pixels", target="pixels", img_size=img_size
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
        self.projector_proj = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)

        self.model = (
            JEPA(
                encoder=self.encoder,
                predictor=self.predictor,
                action_encoder=self.action_encoder,
                projector=self.projector,
                pred_proj=self.projector_proj,
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
        encodes it once into a latent vector stored in self.goal_latent.
        """
        video_path = get_episode_video_path(self.dataset_root, episode_idx)
        pixels = get_goal_pixels(video_path)

        if pixels is None:
            return False

        # 1. Transform: (3, H, W) -> (1, 1, 3, 224, 224)
        batch = self.transform({"pixels": pixels})
        processed_pixels = batch["pixels"].to(self.device)
        processed_pixels = processed_pixels.unsqueeze(0).unsqueeze(0)

        # 2. Encode to Latent once
        info = self.model.encode({"pixels": processed_pixels})
        self.goal_latent = info["emb"].detach()  # (1, 1, D)

        print(f"✅ Goal Latent Cached: {self.goal_latent.shape}")
        return True

    def predict(self, *args, **kwargs):
        """Proxy to the internal World Model's prediction logic."""
        return self.model.predict(*args, **kwargs)

    @torch.no_grad()
    def get_cost(self, obs_dict, actions):
        """
        FAST-PATH MPC COST (Temporally Aligned & VRAM Optimized)
        Calculates the distance to the goal latent by rolling out candidate actions.
        """
        assert hasattr(self, "goal_latent"), "Goal not set. Call set_goal() first."

        # 1. Memory-Aware Observation Encoding
        pixels = obs_dict["pixels"]
        num_candidates = actions.size(0) * actions.size(1)
        B, S = actions.size(0), actions.size(1)

        # 🚀 OPTIMIZATION: De-duplicate Encoding
        # If pixels are 6D (B, S, T, C, H, W) and S > 1, check if redundant
        if pixels.ndim == 6 and S > 1:
            # We assume for single-env planning that all samples share the same history
            # Encode only the FIRST sample to save 99.9% VRAM
            unique_pixels = pixels[:, 0, :, :, :, :]  # (B, T, C, H, W)
            info = self.model.encode({"pixels": unique_pixels})
            init_emb = info["emb"]  # (B, T, D)
            # Expand latents downstream to match num_candidates
            latent_needs_expansion = True
        else:
            # Standard 5D path or non-redundant
            info = self.model.encode(obs_dict)
            init_emb = info["emb"]
            latent_needs_expansion = init_emb.size(0) == B and S > 1

        # 2. Robust History Action Extraction
        hist_actions = obs_dict.get("action", None)
        if hist_actions is None:
            hist_actions = torch.zeros(B, init_emb.size(1), actions.size(-1)).to(
                self.device
            )

        # Stabilize History Action Shapes
        if hist_actions.ndim == 4:
            # Slicing the FIRST sample's history to match the unique_pixels logic
            hist_actions = hist_actions[:, 0, : init_emb.size(1), :]  # (B, 3, D)
        else:
            hist_actions = hist_actions[:, : init_emb.size(1), :]

        # Final History expansion to match all candidates
        hist_actions = (
            hist_actions.repeat_interleave(S, dim=0).to(self.device).float()
        )  # (BS, 3, D)

        # 3. Prepare Plan Actions & Concatenate
        plan_actions = (
            actions.view(num_candidates, -1, actions.size(-1)).to(self.device).float()
        )
        all_actions = torch.cat([hist_actions, plan_actions], dim=1)

        # 4. Sliding Window Rollout
        if latent_needs_expansion:
            curr_emb = init_emb.repeat_interleave(S, dim=0)
        else:
            curr_emb = init_emb

        curr_emb = curr_emb.to(self.device)
        T_horizon = actions.size(2)
        history_size = init_emb.size(1)

        for t in range(T_horizon):
            emb_window = curr_emb[:, -history_size:, :]
            act_window = all_actions[:, t : t + history_size, :]

            act_emb = self.model.action_encoder(act_window)
            pred_emb = self.model.predict(emb_window, act_emb)

            last_pred = pred_emb[:, -1:, :]
            curr_emb = torch.cat([curr_emb, last_pred], dim=1)

        # 5. Latent Distance (MSE) to cached goal
        final_latent = curr_emb[:, -1, :]
        goal_vec = self.goal_latent.view(1, -1)
        dist = torch.norm(final_latent - goal_vec, dim=-1)  # (BS,)

        return dist.view(B, S)
