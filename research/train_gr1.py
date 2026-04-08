import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

# Ensure we can import from the le_wm submodule
LEWM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../le_wm"))
sys.path.append(LEWM_ROOT)

# Import official LeWM components
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from lewm_data_plugin import LEWMDataPlugin
from metrics import MetricsCallback


class ColabLoggingCallback(pl.Callback):
    """Minimalist logger for Colab to prevent console flicker."""

    def on_train_epoch_start(self, trainer, pl_module):
        print(
            f"🚀 [Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] Training in progress..."
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            print(f"🔍 [Epoch {trainer.current_epoch + 1}] Running validation...")


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    # NORMALIZATION: The SIGReg class in le_wm multiplies by B.
    # We divide by B here to make the config weight batch-size invariant.
    batch_size = batch["action"].shape[0]
    lambd = cfg.loss.sigreg.weight / batch_size

    # Replace NaN values with 0
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def run(cfg):
    print("🏗️  Initializing GR-1 Implementation of LeWorldModel...")

    #########################
    ##       dataset       ##
    #########################

    # Inject GR1DataPlugin if repo_id is present in the config
    if "repo_id" in cfg.data.dataset:
        print(f"📦 Using LEWMDataPlugin for repository: {cfg.data.dataset.repo_id}")
        dataset = LEWMDataPlugin(
            repo_id=cfg.data.dataset.repo_id,
            keys_to_load=cfg.data.dataset.keys_to_load,
            num_steps=cfg.data.dataset.num_steps,
        )
    else:
        # Fallback to official dataset loader
        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    # 1. Rescale & Normalize Pixels
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    # 2. Standardize States/Actions (Z-Score)
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
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

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
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
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
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
            # Set val batches to 1/2 of train batches, but at least 2
            balanced_val = max(2, int(cfg.trainer.limit_train_batches * 0.5))
            cfg.trainer.limit_val_batches = balanced_val
            print(f"⚖️  Auto-balanced validation to {balanced_val} batches.")

    metrics_callback = MetricsCallback(log_every_n_steps=1)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, metrics_callback, ColabLoggingCallback()],
        num_sanity_val_steps=1,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        # ckpt_path=None  # Start from scratch or load weights below
    )

    # Note: If you want to load pre-trained vision weights,
    # you would do encoder.load_state_dict(...) here.

    print("🚀 Launching GR-1 Official Training Loop...")
    manager()
    return


if __name__ == "__main__":
    run()
