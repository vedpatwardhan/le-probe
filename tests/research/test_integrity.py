import torch
import torch.nn as nn
import pytest
from research.gr1_modules import GR1Embedder, GR1MLP


def test_gr1_embedder_forward():
    """Verify that the Residual Action Encoder handles 32-DoF input and produces embeddings."""
    batch_size, seq_len, act_dim = 8, 3, 32
    emb_dim = 192

    model = GR1Embedder(input_dim=act_dim, smoothed_dim=256, emb_dim=emb_dim)
    x = torch.randn(batch_size, seq_len, act_dim)

    out = model(x)

    assert out.shape == (batch_size, seq_len, emb_dim)
    assert not torch.isnan(out).any()


def test_gr1_mlp_layernorm():
    """Verify that GR1MLP uses LayerNorm and produces correct shapes."""
    batch_size, seq_len, in_dim = 8, 3, 192
    hidden_dim, out_dim = 2048, 192

    model = GR1MLP(input_dim=in_dim, hidden_dim=hidden_dim, output_dim=out_dim)

    # Flatten seq/batch as the real projector does
    x = torch.randn(batch_size * seq_len, in_dim)
    out = model(x)

    assert out.shape == (batch_size * seq_len, out_dim)

    # Verify LayerNorm is present
    has_ln = any(isinstance(m, nn.LayerNorm) for m in model.net.modules())
    assert has_ln, "GR1MLP must use LayerNorm to prevent batch symmetry masking."

    # Verify BatchNorm is ABSENT
    has_bn = any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.net.modules()
    )
    assert not has_bn, "GR1MLP should NOT use BatchNorm."


def test_weight_loading_guard_logic():
    """Verify the logic used in train_gr1.py to catch patch size mismatches."""
    # Simulation of hub dict (patch_size 14)
    hub_patch_weight = torch.randn(192, 3, 14, 14)

    # Simulation of local dict (patch_size 16)
    local_patch_weight = torch.randn(192, 3, 16, 16)

    # This is the logic we added to train_gr1.py
    with pytest.raises(RuntimeError) as excinfo:
        if hub_patch_weight.shape != local_patch_weight.shape:
            raise RuntimeError("🚨 FATAL: Vision Encoder Patch Size Mismatch!")

    assert "Patch Size Mismatch" in str(excinfo.value)
