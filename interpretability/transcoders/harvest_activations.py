import os
import sys

# --- Path Stabilization ---
# 1. Project Root (le-probe/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 2. LEWM Logic (lewm/)
LEWM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lewm"))
if LEWM_DIR not in sys.path:
    sys.path.insert(0, LEWM_DIR)

# 3. Submodule Logic (le_wm/)
SUBMODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lewm", "le_wm")
)
if SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, SUBMODULE_DIR)
# --------------------------

import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from nnsight import LanguageModel

# Official LeWM Imports
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP
from lewm.train_lewm import RewardPredictor
from lewm.lewm_data_plugin import LEWMDataPlugin
from lewm.le_wm.utils import get_img_preprocessor


class JEPATraceWrapper(torch.nn.Module):
    def __init__(self, jepa_model):
        super().__init__()
        self.jepa = jepa_model

    def forward(self, pixels, action=None):
        info = {"pixels": pixels}
        if action is not None:
            info["action"] = action
        return self.jepa.encode(info)


def harvest_activations(
    model_path,
    repo_id,
    component,
    layer_idx,
    output_path,
    num_episodes=50,
    batch_size=32,
):
    """
    Harvests activations from a specific layer using nnsight.
    Supports 'encoder' and 'predictor' components.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"🚀 Harvesting Activations | {component} Layer {layer_idx} | Device: {device}"
    )

    # 1. Initialize Model Architecture (Strict Parity with GoalMapper)
    jepa_model = JEPA(
        encoder=spt.backbone.utils.vit_hf(
            "tiny", patch_size=14, image_size=224, pretrained=False
        ),
        predictor=ARPredictor(
            num_frames=3,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
        ),
        action_encoder=GR1Embedder(input_dim=32, emb_dim=192),
        projector=GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048),
        pred_proj=GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048),
    ).to(device)
    jepa_model.reward_head = RewardPredictor(input_dim=192, hidden_dim=512).to(device)

    # 2. Load Official Weights
    print(f"🧠 Loading Oracle Weights: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    jepa_model.load_state_dict(new_sd, strict=False)
    jepa_model.eval()

    # 3. Setup Trace Wrapper (To avoid touching submodule)
    model = JEPATraceWrapper(jepa_model)
    tracer = LanguageModel(model, device_map=device)

    # 4. Setup Data Plugin (Direct Bypass)
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    dataset = LEWMDataPlugin(
        repo_id=repo_id,
        keys_to_load=["pixels", "action"],
        num_steps=3,
        transform=transform,
    )

    # Filter for first N episodes
    ep_indices = dataset.episode_indices
    max_frame_idx = torch.where(ep_indices < num_episodes)[0][-1].item()

    dataloader = DataLoader(
        torch.utils.data.Subset(dataset, range(max_frame_idx)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # 5. Sanity Check (Ghost Trace)
    print("👻 Running Sanity Check (Ghost Trace)...")
    dummy_pixels = torch.randn(1, 3, 3, 224, 224).to(device)
    dummy_actions = torch.zeros(1, 3, 32).to(device)

    with tracer.trace(dummy_pixels, action=dummy_actions) as invocation:
        # Try to capture the very first encoder block as a probe
        probe = model.jepa.encoder.blocks[0].output.save()

    if probe.value is not None and probe.value.shape[-1] == 192:
        print(f"✅ Sanity Check Passed: Captured Ghost Activation {probe.value.shape}")
    else:
        print("🚨 Sanity Check Failed: Could not capture internal activations.")
        return

    activations = []

    # 6. Harvesting Loop
    print(f"📊 Harvesting from {num_episodes} episodes (~{max_frame_idx} frames)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Tracing"):
            pixels = batch["pixels"].to(device)  # (B, T, C, H, W)
            actions = batch["action"].to(device)  # (B, T, D)

            # Trace the model
            with tracer.trace(pixels, action=actions) as invocation:
                if component == "encoder":
                    # Hook the specific ViT block
                    # Note: ViT structure in spt.backbone is model.jepa.encoder.blocks[i]
                    target = model.jepa.encoder.blocks[layer_idx].output.save()
                elif component == "predictor":
                    # Hook the ARPredictor block
                    target = model.jepa.predictor.layers[layer_idx].output.save()
                else:
                    raise ValueError(f"Unknown component: {component}")

            activations.append(target.value.cpu())

    # 6. Save Results
    final_acts = torch.cat(activations, dim=0)
    torch.save(final_acts, output_path)
    print(f"💾 Saved {final_acts.shape} activations to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to gr1_reward_tuned_v2.ckpt"
    )
    parser.add_argument("--dataset", type=str, default="vedpatwardhan/gr1_pickup_grasp")
    parser.add_argument(
        "--component", type=str, choices=["encoder", "predictor"], default="encoder"
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--output", type=str, default="activations.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    harvest_activations(
        args.model,
        args.dataset,
        args.component,
        args.layer,
        args.output,
        args.episodes,
        args.batch_size,
    )
