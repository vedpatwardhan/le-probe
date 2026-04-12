"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames once).
3. Implements high-performance windowed rollouts for the MPC cost loop.
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
        self.projector_proj = GR1MLP(
            input_dim=192, output_dim=192, hidden_dim=2048
        )  # training uses pred_proj

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
        FAST-PATH MPC COST (Temporally Aligned)
        Calculates the distance to the goal latent by rolling out candidate actions.
        Maintains a sliding window of exactly 3 tokens for latents and actions.
        """
        assert hasattr(self, "goal_latent"), "Goal not set. Call set_goal() first."

        # 1. Robust Observation Encoding
        pixels = obs_dict["pixels"]
        if pixels.ndim == 6:
            B, S, T, C, H, W = pixels.shape
            pixels = rearrange(pixels, "b s t c h w -> (b s) t c h w")
            info = self.model.encode({"pixels": pixels})
        else:
            B = pixels.size(0)
            S = actions.size(1)  # Assumed from solver context
            info = self.model.encode(obs_dict)

        init_emb = info["emb"]  # (N, T_history, D)
        num_candidates = B * S

        # 2. Prepare Action Stream (History + Plan)
        # We need the actions that lead to/transition the history frames.
        # obs_dict['action'] usually contains (B, T_history+offset, action_dim)
        hist_actions = obs_dict.get("action", None)
        if hist_actions is None:
            # Fallback to zero'd history actions if not provided
            hist_actions = torch.zeros(B, init_emb.size(1), actions.size(-1)).to(
                self.device
            )

        # We only need the first T_history actions for alignment
        hist_actions = hist_actions[:, : init_emb.size(1), :]  # (B, 3, adim)
        hist_actions = hist_actions.repeat_interleave(S, dim=0)  # (BS, 3, adim)
        hist_actions = hist_actions.to(self.device)

        # Candidate actions from solver: (B, S, T_horizon, adim)
        plan_actions = actions.view(num_candidates, -1, actions.size(-1)).to(
            self.device
        )

        # Full action stream for the predictor: (BS, 3 + T_horizon, adim)
        all_actions = torch.cat([hist_actions, plan_actions], dim=1)

        # 3. Sliding Window Rollout
        # Expand latents to match candidates if needed
        if init_emb.size(0) == B:
            curr_emb = init_emb.repeat_interleave(S, dim=0)
        else:
            curr_emb = init_emb

        curr_emb = curr_emb.to(self.device)

        T_horizon = actions.size(2)
        history_size = init_emb.size(1)  # 3

        for t in range(T_horizon):
            # Window slicing:
            # We use the current most recent 3 latents
            # and the 3 actions leading to their transitions/next states.
            # Step 0: x is emb[0,1,2], c is act[0,1,2]. Predicts emb[3].
            # Step 1: x is emb[1,2,3], c is act[1,2,3]. Predicts emb[4].
            emb_window = curr_emb[:, -history_size:, :]
            act_window = all_actions[:, t : t + history_size, :]

            # Embed the actions
            act_emb = self.model.action_encoder(act_window)

            # Predict next latent
            pred_emb = self.model.predict(emb_window, act_emb)

            # Append last prediction: (BS, 1, D)
            last_pred = pred_emb[:, -1:, :]
            curr_emb = torch.cat([curr_emb, last_pred], dim=1)

        # 4. Latent Distance (MSE) to cached goal
        final_latent = curr_emb[:, -1, :]  # (BS, D)
        goal_vec = self.goal_latent.view(1, -1)  # (1, D)
        dist = torch.norm(final_latent - goal_vec, dim=-1)  # (BS,)

        return dist.view(B, S)
