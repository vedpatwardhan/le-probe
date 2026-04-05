import torch
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

from le_wm.jepa import JEPA
from le_wm.module import ARPredictor, Embedder, MLP

# 1. Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HIDDEN_DIM = 192  # ViT-Tiny
ACTION_DIM = 64  # Rosetta-64 Protocol
HISTORY_LEN = 3  # 3 frames of history
REPO_ID = "quentinll/lewm-cube"
DATASET_PATH = "datasets/vedpatwardhan/gr1_pickup_large"


def train():
    print(f"🚀 Initializing GR-1 World Model Training on {DEVICE}...")

    # A. Load Dataset (Single View: world_center)
    dataset = LeRobotDataset(
        repo_id="gr1_pickup_large",
        root=DATASET_PATH,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # B. Build Architecture (Transfer Learning)
    config = ViTConfig(
        hidden_size=HIDDEN_DIM,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=HIDDEN_DIM * 4,
        image_size=224,
        patch_size=14,
    )
    encoder = ViTModel(config, add_pooling_layer=False)

    # Initialize a clean predictor and 64-D action encoder
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

    projector = MLP(input_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM, hidden_dim=2048)
    pred_proj = MLP(input_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM, hidden_dim=2048)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    ).to(DEVICE)

    # C. Load Pre-trained Vision Weights (lewm-cube)
    print("Downloading pre-trained Vision weights...")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="weights.pt")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Load vision encoder only (keep actions/predictor fresh for GR1)
    vision_keys = {
        k: v
        for k, v in state_dict.items()
        if "encoder" in k and "action_encoder" not in k
    }
    model.load_state_dict(vision_keys, strict=False)
    print("✅ Vision Backbone Initialized from pre-trained weights.")

    # D. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Prepare Observations (extract world_center)
            # Shapes: images (B, C, H, W) -> we need (B, T, C, H, W)
            # For simplicity in this demo, we assume the dataset loader yields
            # sequences. If not, you'd buffer them here.
            pixels = batch["observation.images.world_center"].to(
                DEVICE
            )  # (B, 3, 224, 224)
            actions = batch["action"].to(DEVICE)  # (B, 64)

            # 1. Encode Target (Next State)
            target = (
                batch["next.observation.images.world_center"].to(DEVICE)
                if "next.observation.images.world_center" in batch
                else pixels
            )
            with torch.no_grad():
                z_target = model.encoder(target).last_hidden_state[:, 0, :]

            # 2. Predict State
            # Mocking history for a single-step fine-tuning pass
            z_curr = (
                model.encoder(pixels).last_hidden_state[:, 0, :].unsqueeze(1)
            )  # (B, 1, D)
            # Pad history to HISTORY_LEN
            z_history = z_curr.repeat(1, HISTORY_LEN, 1)

            # Encode Action
            c = model.action_encoder(actions.unsqueeze(1).repeat(1, HISTORY_LEN, 1))

            # Forward Pass
            z_pred = model.predictor(z_history, c)[:, -1:]  # (B, 1, D)

            # 3. Calculate JEPA Loss
            loss = F.mse_loss(z_pred.squeeze(1), z_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())


if __name__ == "__main__":
    train()
