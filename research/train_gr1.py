import builtins
import logging
import sys
import os
import time
from datetime import datetime


# --- GLOBAL CONSOLE INSTRUMENTATION ---
# Consolidated stateful stream wrapper to ensure exactly one timestamp per line
class TimestampedStream:
    def __init__(self, original_stream):
        self.stream = original_stream
        self.at_start_of_line = True

    def write(self, text):
        if not text:
            return

        # If we are starting a new line, add the timestamp
        if (
            self.at_start_of_line
            and text.strip()
            and not text.startswith("[")
            and not text.startswith("\r")
        ):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.stream.write(f"[{ts}] ")

        self.stream.write(text)
        self.at_start_of_line = text.endswith("\n")

    def flush(self):
        self.stream.flush()


# Apply to both stdout and stderr for total coverage
sys.stdout = TimestampedStream(sys.stdout)
sys.stderr = TimestampedStream(sys.stderr)

# Disable basicConfig's own timestamp to avoid duplication with our global wrapper
logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=logging.INFO,
    force=True,
)
# --------------------------------------

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from huggingface_hub import hf_hub_download
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
import pkg_resources

# Ensure we can import from the le_wm submodule
LEWM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../le_wm"))
sys.path.append(LEWM_ROOT)

# Import official LeWM components
from jepa import JEPA
from module import ARPredictor, SIGReg
from gr1_modules import GR1Embedder, GR1MLP
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from lewm_data_plugin import LEWMDataPlugin
from metrics import MetricsCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    t0 = time.time()
    if hasattr(self, "_step_end_time"):
        wait_time = t0 - self._step_end_time
        print(f"📦 Batch received! (Wait Time: {wait_time:.4f}s)")
    else:
        print(f"📦 Batch received (First Step)")
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    # --- ENHANCED DIAGNOSTIC PROBE ---
    if self.trainer.global_step == 0:
        px = batch["pixels"]
        print(f"\n🩺 [STEP 0] DATA HEALTH CHECK:")
        print(f"  - Pixel Shape:    {px.shape}")
        print(f"  - Pixel Range:    [{px.min():.2f}, {px.max():.2f}]")
        print(f"  - Pixel Mean/Var: {px.mean():.4f} / {px.var():.8f}")

        # BATCH UNIQUENESS CHECK
        if px.shape[0] > 1:
            px_diff = (px[0] - px[1]).abs().var()
            act_diff = (batch["action"][0] - batch["action"][1]).abs().var()
            print(f"  - Batch Variance (Sample 0 vs 1):")
            print(f"    - Pixel Diff Var:  {px_diff:.8f}")
            print(f"    - Action Diff Var: {act_diff:.8f}")
            if px_diff < 1e-8:
                print("🚨 CRITICAL: BATCH IS CLONED! Sample 0 and 1 are identical.")

    # PREPARATION
    # Pixels are already normalized by the dataloader transforms [0, 1] -> ImageNet norm
    pixels = batch["pixels"]
    actions = torch.nan_to_num(batch["action"], 0.0)

    # Forward pass through model
    info = {"pixels": pixels, "action": actions}
    output = self.model.encode(info)
    emb = output["emb"]  # (B, T, D)
    self.last_z = emb.detach()

    if self.trainer.global_step == 0:
        emb_diff = (emb[0] - emb[1]).abs().var() if emb.shape[0] > 1 else 0.0
        print(f"  - Latent Variance:  {emb.var():.8f}")
        print(f"  - Latent Diff Var:  {emb_diff:.8f}")
        if emb_diff < 1e-8:
            print("🚨 ALERT: Latent manifold has zero batch variance.")
        print("---------------------------------\n")

    # SIGReg weight balancing
    batch_size = actions.shape[0]
    lambd = cfg.loss.sigreg.weight / batch_size

    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss (Force SIGReg to float32 for SVD stability)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.float().transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"].to(
        output["pred_loss"].dtype
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self._step_end_time = time.time()
    return output


@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def run(cfg):
    print("🏗️  Initializing GR-1 Implementation of LeWorldModel...")

    #########################
    ##       dataset       ##
    #########################

    # 1. Initialize the Data Plugin (LeRobot -> LeWorldModel Shim)
    # We prioritize the 'processed' dataset for significant training speedups.
    # Note: Even if a checkpoint reloads an old repo_id, we override it here for performance.
    repo_id = "vedpatwardhan/gr1_pickup_processed"
    print(f"📦 Initializing Data Plugin for: {repo_id}")

    # Standard keys to load if not specified in config
    default_keys = ["observation.state", "observation.images.world_center"]
    keys_to_load = cfg.data.get("keys_to_load") or default_keys

    dataset = LEWMDataPlugin(
        repo_id=repo_id,
        keys_to_load=keys_to_load,
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        use_virtual_actions=cfg.data.get("use_virtual_actions", True),
    )

    # 2. Rescale & Normalize Pixels
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    # 3. Standardize States/Actions (Z-Score)
    with open_dict(cfg):
        for col in keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            # Update WM dims for the predictor
            col_dim = dataset.get_dim(col)
            setattr(cfg.wm, f"{col}_dim", col_dim)
            print(f"📊 Auto-detected {col} dimension: {col_dim}")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        num_workers=2,
        shuffle=False,
        drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = GR1Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = GR1MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
    )

    predictor_proj = GR1MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": lambda optimizer, module: LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps=max(
                    1,
                    int(
                        0.01
                        * getattr(module.trainer, "estimated_stepping_batches", 100)
                    ),
                ),
                max_steps=getattr(module.trainer, "estimated_stepping_batches", 1000),
                warmup_start_lr=1e-5,
            ),
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or "gr1_official"
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        # Override wandb name for GR-1
        cfg.wandb.config.name = f"gr1-lewm-{cfg.data.dataset.repo_id.split('/')[-1]}"
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    # AUTO-BALANCE VALIDATION: If we are in a debug run (limited train batches),
    # automatically limit validation batches to prevent long hangs.
    with open_dict(cfg):
        if cfg.trainer.get("limit_train_batches"):
            # Debug Mode: Set val batches to 1/2 of train batches
            balanced_val = max(2, int(cfg.trainer.limit_train_batches * 0.5))
        else:
            # Full Run Mode: Use a safety cap of 50 batches unless overridden in config
            # This prevents validation from taking 10x longer than training.
            balanced_val = cfg.trainer.get("limit_val_batches", 50)

        cfg.trainer.limit_val_batches = balanced_val
        print(f"⚖️  Validation capped at {balanced_val} batches per epoch.")

    metrics_callback = MetricsCallback(log_every_n_steps=1)

    # 💾 CHECKPOINT PERSISTENCE LOGIC
    # 1. Sanitize the checkpoint path
    ckpt_path_str = cfg.get("ckpt_path")
    if ckpt_path_str:
        ckpt_path_str = ckpt_path_str.strip("\"'")
        checkpoint_dir = str(Path(ckpt_path_str).parent)
        print(f"📊 PERSISTENCE: Saving subsequent checkpoints to: {checkpoint_dir}")
    else:
        checkpoint_dir = str(run_dir / "checkpoints")

    # 2. Configure versioned checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gr1-epoch={epoch:02d}-step={step:06d}",
        save_top_k=-1,  # Research mode: Keep all historical saves
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )

    # 3. 🔍 RESTORE RESEARCH FEATURES
    # We manually load the 10+ research callbacks that spt.Manager usually adds.
    # This ensures WandB resumption and CPU offloading work correctly.
    spt_callbacks = []
    print("🔖 Research Features: Loading library callbacks...")
    for entry_point in pkg_resources.iter_entry_points("stablepretraining_callbacks"):
        try:
            cb_cls = entry_point.load()
            spt_callbacks.append(cb_cls())
            print(f"  ✓ Attached: {entry_point.name}")
        except Exception as e:
            print(f"  ❌ Failed to load {entry_point.name}: {e}")

    # Combine Library Callbacks + Our Custom Callbacks
    all_callbacks = spt_callbacks + [
        object_dump_callback,
        metrics_callback,
        checkpoint_callback,
    ]

    # 4. Instantiate Trainer directly (Safest way to avoid OmegaConf errors)
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=all_callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path_str,
    )

    # 🔗 Warm-start from Pretrained Weights (HF: quentinll/lewm-cube)
    # This seeds the Vision Encoder and Predictor with manipulation "common sense"
    # while allowing the action_encoder to re-initialize for the 32-DoF GR-1.
    # SKIPPED if resuming from a checkpoint to avoid overwriting robot-specific weights.
    if cfg.get("use_pretrained_cube") and not cfg.get("ckpt_path"):
        print("📥 Downloading pretrained cube manipulation weights from HF...")
        weights_path = hf_hub_download(
            repo_id="quentinll/lewm-cube", filename="weights.pt"
        )
        state_dict = torch.load(weights_path, map_location="cpu")

        print("🧠 Loading weights into World Model (Warm-start)...")
        model_dict = world_model.model.state_dict()

        # STRICT VERIFICATION: We must load the Vision Encoder Patch Embeddings.
        # If this fails, the model is effectively blind.
        patch_key = "encoder.embeddings.patch_embeddings.projection.weight"
        if (
            patch_key in state_dict
            and state_dict[patch_key].shape != model_dict[patch_key].shape
        ):
            raise RuntimeError(
                f"🚨 FATAL: Vision Encoder Patch Size Mismatch! "
                f"Hub: {state_dict[patch_key].shape} Local: {model_dict[patch_key].shape}. "
                f"Ensure patch_size is correctly aligned in config."
            )

        filtered_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # Load into world_model.model
        msg = world_model.model.load_state_dict(filtered_dict, strict=False)
        print(
            f"✅ Weights loaded. Transferred: {len(filtered_dict)} layers. "
            f"Skipped: {len(state_dict) - len(filtered_dict)} (due to configuration mismatch)."
        )

        if len(msg.missing_keys) > 200:
            print(
                f"⚠️ WARNING: High number of missing keys ({len(msg.missing_keys)}). Check compatibility!"
            )

    print("🚀 Launching GR-1 Official Training Loop...")
    manager()
    return


if __name__ == "__main__":
    run()
