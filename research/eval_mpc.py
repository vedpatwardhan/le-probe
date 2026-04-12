import os
import sys
from pathlib import Path

# Add paths for custom modules
CORTEX_GR1 = Path("/Users/vedpatwardhan/Desktop/cortex-os/cortex-gr1")
sys.path.append(str(CORTEX_GR1 / "le_wm"))
sys.path.append(str(CORTEX_GR1 / "research"))

# Setup headless MuJoCo
os.environ["MUJOCO_GL"] = "egl"

# Core imports
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import stable_pretraining as spt

# Project-specific imports (Absolute relative to sys.path)
from goal_mapper import GoalMapper
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP


def get_img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def inject_cost_function(model, goal_latent):
    """
    Injects a get_cost method into the Oracle model for CEM solver.
    This fulfills the JEPA-WM requirement for planning.
    """

    def get_cost(obs_dict, actions):
        # obs_dict['pixels']: (B, T, C, H, W)
        # actions: (B, H, action_dim)

        # 1. rollout the predictor
        # For simplicity in this wrapper, we assume batch size 1 for now
        with torch.no_grad():
            output = model.predict(obs_dict, actions)
            # output['emb'] contains the predicted latent trajectory
            # shape: (B, Horizon, D)

            last_latent = output["emb"][:, -1, :]  # Final state latent

            # L2 Distance to goal latent
            dist = torch.norm(last_latent - goal_latent, dim=-1)

            # Action smoothness penalty
            # acc = actions[:, 1:] - actions[:, :-1]
            # smooth_penalty = torch.norm(acc, dim=-1).mean(dim=1) * 0.1

            return dist  # + smooth_penalty

    model.get_cost = get_cost
    return model


@hydra.main(version_base=None, config_path="./config/eval", config_name="gr1_manip")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧬 Initializing GR-1 MPC Evaluator on {device}")

    # 1. Goal Mapping
    mapper = GoalMapper(cfg.oracle_path, cfg.dataset_root)
    goal_latent = mapper.find_goal_latent(cfg.target_xyz)
    if goal_latent is None:
        print("❌ Could not find goal latent in dataset. Exiting.")
        return
    goal_latent = goal_latent.to(device)

    # 2. Manual Model Load (Oracle v17)

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False
    )
    predictor = ARPredictor(
        num_frames=3,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
    )
    action_encoder = GR1Embedder(input_dim=64, emb_dim=192)
    projector = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)
    predictor_proj = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    ).to(device)
    checkpoint = torch.load(cfg.oracle_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # 3. Inject Cost Function
    model = inject_cost_function(model, goal_latent)

    # 4. Setup World & Solver
    # world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": get_img_transform(cfg)}

    # In a real run, we would instantiate the solver and policy here
    # Since we are in "Logic Implementation" phase, we will stop here for today
    # and provide the user with the diagostic tool next.
    print("✅ Model, Cost Function, and Goal Latent established.")
    print(f"🎯 Goal Latent Norm: {torch.norm(goal_latent).item():.4f}")


if __name__ == "__main__":
    main()
