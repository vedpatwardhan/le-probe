import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import mujoco
import numpy as np
import wandb

# Add the submodule to the path so we don't have to touch its internal files
sys.path.append(os.path.join(os.path.dirname(__file__), "le_wm"))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

# 1. GR-1 Configuration
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
HIDDEN_DIM = 192  # ViT-Tiny
ACTION_DIM = 64  # Rosetta-64 Protocol
HISTORY_LEN = 3  # 3 frames of history
REPO_ID_MODEL = "quentinll/lewm-cube"
REPO_ID_DATASET = "vedpatwardhan/gr1_pickup_large"

# 2. MuJoCo Kinematics Setup (For verified height grounding)
XML_PATH = os.path.join(os.path.dirname(__file__), "sim_assets/scene_gr1_pickup.xml")
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data = mujoco.MjData(mj_model)

JOINT_MAP = {
    0: "left_shoulder_pitch_joint",
    1: "left_shoulder_roll_joint",
    2: "left_shoulder_yaw_joint",
    3: "left_elbow_pitch_joint",
    4: "left_wrist_yaw_joint",
    5: "left_wrist_roll_joint",
    6: "left_wrist_pitch_joint",
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_pitch_joint",
    26: "right_wrist_yaw_joint",
    27: "right_wrist_roll_joint",
    28: "right_wrist_pitch_joint",
    41: "waist_yaw_joint",
    42: "waist_pitch_joint",
    43: "waist_roll_joint",
}

QPOS_ADDR = {
    k: mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, v)]
    for k, v in JOINT_MAP.items()
}
HAND_BODY_ID = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_pitch_link"
)


# 3. GR-1 Height Regressor (Abstracted outside le_wm as requested)
class HeightRegressor(nn.Module):
    """Predicts the Z-height of the Right Hand from latent embeddings."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


def get_hand_z(state_batch):
    """
    Computes the ground-truth Z-height of the right hand using MuJoCo Forward Kinematics.

    This function:
    1. Maps the Rosetta-64 state vector indices to the corresponding MuJoCo joints.
    2. Sets the robot's pose in the physics engine.
    3. Solves for the global position (xpos) of the 'right_hand_pitch_link'.

    Args:
        state_batch (torch.Tensor): A batch of robot states in Rosetta-64 format (B, 64).

    Returns:
        torch.Tensor: A tensor of hand Z-coordinates in meters (B, 1).
    """
    B = state_batch.shape[0]
    heights = []
    states_np = state_batch.detach().cpu().numpy()
    for i in range(B):
        # first 7 indices of qpos: X, Y, Z (base position) and w, x, y, z (orientation)
        mj_data.qpos[:] = 0.0
        mj_data.qpos[2] = 0.92
        for rosetta_idx, q_addr in QPOS_ADDR.items():
            mj_data.qpos[q_addr] = states_np[i, rosetta_idx]
        mujoco.mj_kinematics(mj_model, mj_data)
        heights.append(mj_data.xpos[HAND_BODY_ID][2])
    return torch.tensor(heights, device=state_batch.device).unsqueeze(-1)


def train():
    print(f"🚀 Initializing GR-1 World Model Refinement (Phase 1) on {DEVICE}...")
    print(f"Fetching dataset from Hugging Face Hub: {REPO_ID_DATASET}")

    # Initialize WandB experiment
    wandb.init(
        project="gr1-lewm-phase1",
        name="refinement-run-001",
        config={
            "hidden_dim": HIDDEN_DIM,
            "action_dim": ACTION_DIM,
            "lr": 5e-5,
            "epochs": 5,
            "history_len": HISTORY_LEN,
        },
    )

    # A. Load Dataset (Automatically downloads from Hub if not cached)
    dataset = LeRobotDataset(repo_id=REPO_ID_DATASET)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # B. Build Architecture
    config = ViTConfig(
        hidden_size=HIDDEN_DIM,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=HIDDEN_DIM * 4,
        image_size=224,
        patch_size=14,
    )
    encoder = ViTModel(config, add_pooling_layer=False)

    predictor = ARPredictor(
        num_frames=HISTORY_LEN,
        input_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=HIDDEN_DIM,
        depth=6,
        heads=16,
        mlp_dim=2048,
    )
    action_encoder = Embedder(input_dim=ACTION_DIM, emb_dim=HIDDEN_DIM)

    projector = MLP(input_dim=HIDDEN_DIM, hidden_dim=2048, output_dim=HIDDEN_DIM)
    pred_proj = MLP(input_dim=HIDDEN_DIM, hidden_dim=2048, output_dim=HIDDEN_DIM)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    ).to(DEVICE)
    height_head = HeightRegressor(HIDDEN_DIM).to(DEVICE)
    sigreg = SIGReg().to(DEVICE)

    # C. Load Pre-trained Weights
    print("Downloading pre-trained lewm-cube weights...")
    weights_path = hf_hub_download(repo_id=REPO_ID_MODEL, filename="weights.pt")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    vision_keys = {
        k: v
        for k, v in state_dict.items()
        if "encoder" in k and "action_encoder" not in k
    }
    model.load_state_dict(vision_keys, strict=False)
    print("✅ Vision Backbone Initialized.")

    # D. Training Loop
    params = list(model.parameters()) + list(height_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-5)

    for epoch in range(5):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            pixels = batch["observation.images.world_center"].to(DEVICE)
            actions = batch["action"].to(DEVICE)
            state = batch["observation.state"].to(DEVICE)

            target = (
                batch["next.observation.images.world_center"].to(DEVICE)
                if "next.observation.images.world_center" in batch
                else pixels
            )
            with torch.no_grad():
                z_target = model.encoder(target).last_hidden_state[:, 0, :]

            z_curr = model.encoder(pixels).last_hidden_state[:, 0, :].unsqueeze(1)
            z_history = z_curr.repeat(1, HISTORY_LEN, 1)
            c = model.action_encoder(actions.unsqueeze(1).repeat(1, HISTORY_LEN, 1))

            z_pred_seq = model.predictor(z_history, c)
            z_pred = z_pred_seq[:, -1:]

            # Loss components
            loss_jepa = F.mse_loss(z_pred.squeeze(1), z_target)
            loss_sigreg = sigreg(z_pred)

            # Grounding with MuJoCo FK
            gt_height = get_hand_z(state)
            pred_height = height_head(z_pred.squeeze(1))
            loss_height = F.mse_loss(pred_height, gt_height)

            total_loss = loss_jepa + (0.1 * loss_sigreg) + (2.0 * loss_height)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Live Metrics & Height Trace Logging
            metrics = {
                "loss/total": total_loss.item(),
                "loss/jepa": loss_jepa.item(),
                "loss/sigreg": loss_sigreg.item(),
                "loss/height": loss_height.item(),
                "trace/pred_z": pred_height[0].item(),
                "trace/gt_z": gt_height[0].item(),
            }
            wandb.log(metrics)
            pbar.set_postfix(
                jepa=f"{loss_jepa.item():.4f}",
                sig=f"{loss_sigreg.item():.4f}",
                h=f"{loss_height.item():.4f}",
            )

    print("✅ Training Complete. Saving GR-1 World Model...")
    torch.save(
        {"jepa": model.state_dict(), "height_head": height_head.state_dict()},
        "gr1_lewm_refinement.pt",
    )
    wandb.finish()


if __name__ == "__main__":
    train()
