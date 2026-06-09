import os
import sys
import time
import torch
import numpy as np
import hydra
import lightning as pl
import stable_pretraining as spt
from functools import partial
from pathlib import Path
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Add repo root to import cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imports from baseline/v1
from lewm.v2.model import MultiViewJEPAv2
from lewm.v2.data import SkeletonDataPluginV2
from lewm.train_lewm import lejepa_forward, RewardPredictor, SIGReg
from lewm.gr1_modules import GR1Embedder, GR1MLP
from lewm.multi_view_encoder import get_multi_view_encoder
from lewm.skeleton.encoder import patch_vit_for_skeleton
from metrics import MetricsCallback
from utils import get_img_preprocessor, ModelObjectCallBack
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR


class SkeletonImportanceCallback(pl.Callback):
    """Logs the relative importance of the 4th channel (Skeleton) during training."""

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            backbone = pl_module.model.encoder.backbone
            weight = backbone.embeddings.patch_embeddings.projection.weight
            rgb_weight_norm = weight[:, :3, :, :].abs().mean()
            skel_weight_norm = weight[:, 3:, :, :].abs().mean()
            importance_ratio = skel_weight_norm / (rgb_weight_norm + 1e-8)
            pl_module.log_dict(
                {
                    "skeleton/weight_norm": skel_weight_norm,
                    "skeleton/rgb_norm_ratio": importance_ratio,
                },
                sync_dist=True,
            )
        except Exception:
            pass


def waim_phase1_forward(self, batch, stage, cfg, log_metrics=True):
    """
    WAIM Phase 1 forward pass:
    1. Computes J-EPA state embeddings and AR predictor rollouts.
    2. Aligns DINO-waypoint targets (z_bar) with detached visual latents at checkpoints.
    3. Optimizes Reward Predictor.
    4. Optimizes Goal-Conditioned Value Head via Temporal Difference (TD) learning.
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    # A. Standard Visual Embedding and Prediction Rollout
    pixels = batch["pixels"]
    actions = torch.nan_to_num(batch["action"], 0.0)
    info = {"pixels": pixels, "action": actions}
    output = self.model.encode(info)
    emb = output["emb"]  # [B, T, D]

    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # [B, n_preds, D]

    # Dynamics prediction loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # B. Reward Head Supervision
    if "progress" in batch:
        R = batch["progress"].to(pred_emb.dtype)
        # Reward head predictor on future rollouts
        future_pred_emb = pred_emb[:, (ctx_len - n_preds) :]
        pred_reward = self.model.reward_head(future_pred_emb).squeeze(-1)
        target_reward = R[:, ctx_len:].to(pred_reward.dtype)
        output["reward_loss"] = F.mse_loss(pred_reward, target_reward)
    else:
        output["reward_loss"] = torch.zeros_like(output["pred_loss"])

    # C. DINO Subgoal Waypoint Projection and Visual Alignment
    if cfg.get("use_dino", False) and "dino_anchor" in batch:
        phi_dino = batch["dino_anchor"]  # [B, T, V, 384]
        B, T = phi_dino.shape[:2]

        # Fuse DINO views to z_bar: [B, T, D]
        z_bar = self.model.project_dino(phi_dino, aggregate_views=True)

        # Visual Alignment Loss: Align projected DINO subgoals with actual embeddings at checkpoint frames
        is_checkpoint = batch.get("is_checkpoint")  # [B, T]
        if is_checkpoint is not None:
            is_checkpoint = is_checkpoint.to(emb.device)
            if is_checkpoint.any():
                z_bar_masked = z_bar[is_checkpoint]
                emb_masked = emb[is_checkpoint]
                # DETACH actual JEPA embeddings to ensure gradient flows only to the Dino Projector
                output["align_loss"] = F.mse_loss(z_bar_masked, emb_masked.detach())
            else:
                output["align_loss"] = torch.zeros_like(output["pred_loss"])
        else:
            output["align_loss"] = torch.zeros_like(output["pred_loss"])

        # D. TD Learning for Goal-Conditioned Value Head V(z_t | z_bar)
        # We can perform TD over sequence steps [0, T-2] to predict V(z_t | z_bar) -> r_t + gamma * V(z_{t+1} | z_bar)
        gamma = cfg.loss.get("gamma", 0.95)

        # We extract matching states for TD transition
        z_t = emb[:, :-1]  # [B, T-1, D]
        z_next = emb[:, 1:]  # [B, T-1, D]
        z_bar_truncated = z_bar[:, :-1]  # [B, T-1, D]

        # Predict values
        val_t = self.model.value_head(z_t, z_bar_truncated).squeeze(-1)  # [B, T-1]
        with torch.no_grad():
            val_next = self.model.value_head(z_next, z_bar[:, 1:]).squeeze(
                -1
            )  # [B, T-1]

        # Immediate reward (progress delta or sparse reward)
        if "progress" in batch:
            rewards = batch["progress"][:, :-1].to(emb.dtype)
        else:
            rewards = torch.zeros_like(val_t)

        # TD Target: r_t + gamma * V(z_{t+1})
        td_target = rewards + gamma * val_next
        output["value_loss"] = F.mse_loss(val_t, td_target)
    else:
        output["align_loss"] = torch.zeros_like(output["pred_loss"])
        output["value_loss"] = torch.zeros_like(output["pred_loss"])

    output["sigreg_loss"] = self.sigreg(emb.float().transpose(0, 1))

    # E. Combine losses
    reward_weight = cfg.loss.get("reward", {}).get("weight", 0.1)
    sigreg_weight = cfg.loss.sigreg.weight
    align_weight = cfg.loss.get("align", {}).get("weight", 0.5)
    value_weight = cfg.loss.get("value", {}).get("weight", 0.5)

    output["loss"] = (
        output["pred_loss"]
        + sigreg_weight * output["sigreg_loss"].to(output["pred_loss"].dtype)
        + reward_weight * output["reward_loss"]
        + align_weight * output["align_loss"]
        + value_weight * output["value_loss"]
    )

    if log_metrics:
        losses_dict = {
            f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k
        }
        self.log_dict(losses_dict, on_step=True, sync_dist=True)

    self._step_end_time = time.time()
    return output


@hydra.main(version_base=None, config_path="../config", config_name="lewm")
def run(cfg):
    print("🦾 Starting Le-Probe v2 (WAIM) Phase 1 Training Loop...")

    pl.seed_everything(cfg.get("seed", 3072), workers=True)

    repo_id = cfg.data.dataset.get("repo_id", "gr1_pickup_grasp")
    keys_to_load = [
        "observation.state",
        "action",
        "world_center",
        "world_left",
        "world_right",
        "world_top",
        "world_wrist",
    ]

    dataset = SkeletonDataPluginV2(
        repo_id=repo_id,
        keys_to_load=keys_to_load,
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        use_virtual_actions=cfg.data.get("use_virtual_actions", True),
        use_multi_view=True,
        img_size=cfg.img_size,
        use_subset=cfg.get("use_subset", False),
    )

    # Transform setup
    transforms = []
    with open_dict(cfg):
        for col in keys_to_load:
            if any(k in col for k in ["pixels", "images", "world_"]):
                transforms.append(
                    get_img_preprocessor(source=col, target=col, img_size=cfg.img_size)
                )
            else:
                col_data = dataset.get_col_data(col)
                data_tensor = torch.from_numpy(np.array(col_data))
                data_tensor = data_tensor[~torch.isnan(data_tensor).any(dim=1)]
                mean = data_tensor.mean(0, keepdim=True).clone()
                std = data_tensor.std(0, keepdim=True).clone()

                def norm_fn(x, m=mean, s=std):
                    return ((x - m) / (s + 1e-8)).float()

                transforms.append(
                    spt.data.transforms.WrapTorchTransform(
                        norm_fn, source=col, target=col
                    )
                )
                col_dim = dataset.get_dim(col)
                clean_name = col.split(".")[-1]
                setattr(cfg.wm, f"{clean_name}_dim", col_dim)

    dataset.orig_transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = dataset.tiled_transform_wrapper
    dataset.clear_cache()

    # Model Initialization
    encoder = get_multi_view_encoder(cfg)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    fusion_type = cfg.get("fusion_type", "mean")
    num_views = cfg.get("num_views", 5)

    world_model = MultiViewJEPAv2(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        ),
        action_encoder=GR1Embedder(input_dim=effective_act_dim, emb_dim=embed_dim),
        projector=GR1MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048),
        pred_proj=GR1MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048),
        embed_dim=embed_dim,
        use_dino=bool(cfg.get("use_dino", False)),
        fusion_type=fusion_type,
        num_views=num_views,
    )
    world_model.reward_head = RewardPredictor(input_dim=embed_dim, hidden_dim=512)

    # Cold start / warm start weight logic
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        # Load cube baseline
        weights_path = hf_hub_download(
            repo_id="quentinll/lewm-cube", filename="weights.pt"
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model_dict = world_model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "") if k.startswith("model.") else k
            if (
                cfg.get("use_multi_view", True)
                and new_key.startswith("encoder.")
                and not new_key.startswith("encoder.backbone.")
            ):
                new_key = new_key.replace("encoder.", "encoder.backbone.", 1)
            new_state_dict[new_key] = v

        filtered_dict = {
            k: v
            for k, v in new_state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        world_model.load_state_dict(filtered_dict, strict=False)
        patch_vit_for_skeleton(encoder.backbone)

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

    world_model_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(waim_phase1_forward, cfg=cfg),
        optim=optimizers,
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)

    # Loader Split
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.loader.num_workers > 0,
        generator=rnd_gen,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.loader.num_workers > 0,
    )

    run_id = cfg.get("subdir") or "gr1_skeleton_v2_official"
    run_dir = Path("./outputs", run_id).absolute()
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=run_dir,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
        callbacks=[
            SkeletonImportanceCallback(),
            ModelObjectCallBack(
                dirpath=run_dir,
                filename="skeleton_lewm_v2",
                epoch_interval=cfg.get("save_interval", 1),
            ),
            MetricsCallback(log_every_n_steps=1),
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                every_n_epochs=cfg.get("save_interval", 1),
            ),
        ],
    )

    trainer.fit(
        model=world_model_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    run()
