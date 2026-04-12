"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames once).
3. Implements high-performance latent rollouts for the MPC cost loop.
"""

import torch
import sys
from pathlib import Path

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
        FAST-PATH MPC COST
        Calculates the distance to the goal latent by rolling out the candidate actions.
        Avoids redundant encoding of the goal and initial state.
        """
        assert hasattr(self, "goal_latent"), "Goal not set. Call set_goal() first."

        # 1. Prepare Initial Latent (Current Robot State)
        # obs_dict['pixels'] should be (B, T, C, H, W)
        # We only need to encode it ONCE for all candidates
        # Many solvers pass B=1 for the observation, but S candidates for actions.
        info = self.model.encode(obs_dict)
        init_emb = info["emb"]  # (B, T_history, D)

        # 2. Vectorized Rollout (Fast Path)
        # actions: (B, S, T_horizon, action_dim)
        # But le_wm predict expects (B, T, action_dim)
        # We flatten B and S to process all candidates at once
        B, S, T_horizon, a_dim = actions.shape
        curr_emb = init_emb.repeat_interleave(S, dim=0)  # (B*S, T_history, D)
        flat_actions = actions.view(B * S, T_horizon, a_dim)

        # Move inputs to device
        curr_emb = curr_emb.to(self.device)
        flat_actions = flat_actions.to(self.device)

        # Autoregressive Prediction Loop (v17 handover logic)
        history_size = init_emb.size(1)
        for t in range(T_horizon):
            # Encode only the action for this step
            # Note: predictor expects history slice
            # act_emb needs to be (BS, T, D) for the predictor view
            # but training uses a simplified single-step prediction for loss.
            # We follow the standard le_wm rollout pattern:
            act_slice = flat_actions[:, : t + 1, :]
            act_emb = self.model.action_encoder(act_slice)

            # Predict next latent
            pred_emb = self.model.predict(
                curr_emb[:, -history_size:], act_emb[:, -history_size:]
            )

            # Extract last prediction and append to simulation "memory"
            last_pred = pred_emb[:, -1:, :]
            curr_emb = torch.cat([curr_emb, last_pred], dim=1)

        # 3. Latent Distance (MSE) to cached goal
        # Comparing the VERY LAST state of the imagined trajectory
        final_latent = curr_emb[:, -1, :]  # (B*S, D)

        # Compute L2 distance to stored goal latent
        # goal_latent is (1, 1, D) -> rescaled to match BS
        goal_vec = self.goal_latent.view(1, -1)  # (1, D)

        dist = torch.norm(final_latent - goal_vec, dim=-1)  # (B*S,)

        # Reshape cost back to (B, S) for the CEM solver
        return dist.view(B, S)
