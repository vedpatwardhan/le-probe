"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames once).
3. Implements high-performance windowed rollouts with VRAM de-duplication.
"""

import torch
from pathlib import Path

# Project-specific imports
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP
from le_wm.utils import get_img_preprocessor
from research.goal_utils import get_goal_pixels, get_episode_video_path
from research.train_lewm import RewardPredictor


class GoalMapper:
    """
    Utility to map task success (last frame of dataset) to World Model latent embeddings.
    Used for Zero-Shot Goal-Conditioned MPC.
    """

    def __init__(self, model_path, dataset_root, img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.action_encoder = GR1Embedder(input_dim=32, emb_dim=192)
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

        # 🌟 RA-LeWM Reward Head Integration 🌟
        self.model.reward_head = (
            RewardPredictor(input_dim=192, hidden_dim=512).to(self.device).eval()
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
        FAST-PATH MPC COST (Ironclad 5D Protocol)
        Guarantees that the World Model only ever sees 5D (B, T, C, H, W).
        """
        # 1. Extract and Force 5D Observation
        raw_pixels = obs_dict["pixels"]
        B, S = actions.size(0), actions.size(1)

        # Defensive Dimension Stripping: (Batch, Samples, T, C, H, W) -> (Batch, T, C, H, W)
        pixels_5d = raw_pixels
        while pixels_5d.ndim > 5:
            pixels_5d = pixels_5d[:, 0]  # Squeeze the Sample/S dimensions

        # 2. Optimized Encoding
        # All samples S share the same history. We encode the 5D batch once.
        info = self.model.encode({"pixels": pixels_5d})
        init_emb = info["emb"]  # (B, T, D)

        # Expand latents for the solver: (B, T, D) -> (B*S, T, D)
        curr_emb = init_emb.repeat_interleave(S, dim=0)

        # 3. History Actions Normalization
        raw_hist_actions = obs_dict.get("action", None)
        if raw_hist_actions is None:
            raw_hist_actions = torch.zeros(B, init_emb.size(1), actions.size(-1)).to(
                self.device
            )

        hist_actions_5d = raw_hist_actions
        while hist_actions_5d.ndim > 3:  # (B, S, T, D) -> (B, T, D)
            hist_actions_5d = hist_actions_5d[:, 0]

        # Flatten to BS space
        flat_hist_actions = (
            hist_actions_5d.repeat_interleave(S, dim=0).to(self.device).float()
        )

        # 4. Prepare Plan Actions (BS, T, D)
        flat_plan_actions = (
            actions.view(B * S, -1, actions.size(-1)).to(self.device).float()
        )
        all_actions = torch.cat([flat_hist_actions, flat_plan_actions], dim=1)

        # 5. Sliding Window Rollout (Flattened BS space)
        history_size = init_emb.size(1)
        T_horizon = actions.size(2)

        # Track all predicted latents for dense costing (BS, T_horizon, D)
        pred_latents = []

        for t in range(T_horizon):
            emb_window = curr_emb[:, -history_size:, :]
            act_window = all_actions[:, t : t + history_size, :]

            act_emb = self.model.action_encoder(act_window)
            pred_emb = self.model.predict(emb_window, act_emb)

            last_pred = pred_emb[:, -1:, :]
            curr_emb = torch.cat([curr_emb, last_pred], dim=1)
            pred_latents.append(last_pred)

        # 6. Optimized Planning Cost Logic
        # Combine all predictions: (BS, T_horizon, D)
        all_preds = torch.cat(pred_latents, dim=1)

        # CHOICE: Use the Reward Head for task-specific optimization
        with torch.no_grad():
            # (BS, T_horizon, D) -> (BS, T_horizon, 1) -> (BS, T_horizon)
            rewards = self.model.reward_head(all_preds).squeeze(-1)

            # Cost = Negative Reward (since CEM minimizes cost)
            # We take the MAX reward found across the horizon (Best reachable state)
            dist = -rewards.max(dim=-1).values

            # 🌟 PHYSICAL GRACE PATCH: Smoothness Penalty 🌟
            # 1. Delta from Current State (BS, D)
            current_state = flat_hist_actions[:, -1, :]
            first_plan_action = flat_plan_actions[:, 0, :]
            jump_start = torch.mean((first_plan_action - current_state) ** 2, dim=-1)

            # 2. Delta within the Plan Horizon (BS, T-1, D)
            if T_horizon > 1:
                jitters = torch.mean(
                    (flat_plan_actions[:, 1:, :] - flat_plan_actions[:, :-1, :]) ** 2,
                    dim=-1,
                )
                jump_internal = torch.mean(jitters, dim=1)
            else:
                jump_internal = 0.0

            # Combined Smoothness Cost (Weight: 500.0)
            # Prevents 'teleporting' to reward peaks at the cost of physical stability
            smoothness_weight = 500.0
            dist = dist + (jump_start + jump_internal) * smoothness_weight

        # 7. Unflatten back to (B, S) for the Solver
        # Scaled by 100 to match the diagnostic sweep's threshold logic
        return dist.view(B, S) * 100.0
