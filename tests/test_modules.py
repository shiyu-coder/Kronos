"""Unit tests for model/module.py components."""

import pytest
import torch
import torch.nn as nn

from model.module import (
    BinarySphericalQuantizer,
    BSQuantizer,
    FeedForward,
    FixedEmbedding,
    HierarchicalEmbedding,
    MultiHeadAttentionWithRoPE,
    RMSNorm,
    RotaryPositionalEmbedding,
    TemporalEmbedding,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_forward_preserves_shape(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 8, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_is_normalized(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 8, 32) * 100  # large values
        out = norm(x)
        # RMS of output along last dim should be close to 1 (weight is ones)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_learnable_weight(self):
        norm = RMSNorm(dim=16)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (16,)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_forward_preserves_shape(self):
        ff = FeedForward(d_model=32, ff_dim=64)
        x = torch.randn(2, 8, 32)
        out = ff(x)
        assert out.shape == x.shape

    def test_different_input_gives_different_output(self):
        torch.manual_seed(0)
        ff = FeedForward(d_model=32, ff_dim=64)
        x1 = torch.randn(1, 4, 32)
        x2 = torch.randn(1, 4, 32)
        out1 = ff(x1)
        out2 = ff(x2)
        assert not torch.allclose(out1, out2)

    def test_zero_dropout(self):
        ff = FeedForward(d_model=16, ff_dim=32, ffn_dropout_p=0.0)
        x = torch.randn(1, 4, 16)
        # deterministic with zero dropout
        out1 = ff(x)
        out2 = ff(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# RotaryPositionalEmbedding
# ---------------------------------------------------------------------------

class TestRotaryPositionalEmbedding:
    def test_output_shape_matches(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.randn(2, 4, 8, 16)  # batch, heads, seq, dim
        k = torch.randn(2, 4, 8, 16)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_cached_vs_fresh_computation(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        # First call: populates cache
        q_out1, k_out1 = rope(q, k)
        # Second call: uses cache
        q_out2, k_out2 = rope(q, k)
        assert torch.allclose(q_out1, q_out2)
        assert torch.allclose(k_out1, k_out2)

    def test_different_seq_len_updates_cache(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q4 = torch.randn(1, 2, 4, 16)
        k4 = torch.randn(1, 2, 4, 16)
        rope(q4, k4)
        assert rope.seq_len_cached == 4

        q8 = torch.randn(1, 2, 8, 16)
        k8 = torch.randn(1, 2, 8, 16)
        rope(q8, k8)
        assert rope.seq_len_cached == 8


# ---------------------------------------------------------------------------
# MultiHeadAttentionWithRoPE
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionWithRoPE:
    def test_output_shape(self):
        attn = MultiHeadAttentionWithRoPE(d_model=32, n_heads=2)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self):
        """Future tokens should not affect past outputs."""
        torch.manual_seed(42)
        attn = MultiHeadAttentionWithRoPE(d_model=32, n_heads=2)
        attn.eval()

        x = torch.randn(1, 8, 32)
        out_full = attn(x)

        # Modify future tokens (positions 4-7) and check past (positions 0-3)
        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn(1, 4, 32) * 10
        out_modified = attn(x_modified)

        # Past positions should be identical under causal masking
        assert torch.allclose(out_full[:, :4, :], out_modified[:, :4, :], atol=1e-5)

    def test_with_padding_mask(self):
        attn = MultiHeadAttentionWithRoPE(d_model=32, n_heads=2)
        x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 6:] = True  # mask last 2 positions for first batch
        out = attn(x, key_padding_mask=mask)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# BinarySphericalQuantizer
# ---------------------------------------------------------------------------

class TestBinarySphericalQuantizer:
    @pytest.fixture
    def bsq(self):
        return BinarySphericalQuantizer(
            embed_dim=9, beta=0.1, gamma0=0.1, gamma=0.1, zeta=0.1,
            group_size=9
        )

    def test_encode_decode_roundtrip(self, bsq):
        z = torch.randn(2, 4, 9)
        zq, _, _ = bsq(z, collect_metrics=False)
        # Quantized output should be scaled binary
        q_scale = 1.0 / (9 ** 0.5)
        unscaled = zq / q_scale
        assert torch.all((unscaled.abs() - 1.0).abs() < 1e-5)

    def test_output_is_binary(self, bsq):
        z = torch.randn(2, 4, 9)
        zq, _, _ = bsq(z, collect_metrics=False)
        q_scale = 1.0 / (9 ** 0.5)
        unscaled = zq / q_scale
        # Values should be -1 or +1
        is_plus_one = (unscaled - 1.0).abs() < 1e-5
        is_minus_one = (unscaled + 1.0).abs() < 1e-5
        assert torch.all(is_plus_one | is_minus_one)

    def test_indices_valid_range(self, bsq):
        z = torch.randn(2, 4, 9)
        zq, _, metrics = bsq(z, collect_metrics=True)
        indices = metrics["indices"]
        assert indices.min() >= 0
        assert indices.max() < 2 ** 9

    def test_codes_to_indexes_roundtrip(self, bsq):
        z = torch.randn(2, 4, 9)
        zq = bsq.quantize(z)
        indices = bsq.codes_to_indexes(zq)
        codes = bsq.indexes_to_codes(indices)
        assert torch.allclose(codes.float(), zq.float())


# ---------------------------------------------------------------------------
# BSQuantizer (wrapper)
# ---------------------------------------------------------------------------

class TestBSQuantizer:
    def test_forward_shapes(self):
        bsq = BSQuantizer(s1_bits=4, s2_bits=4, beta=0.1, gamma0=0.1,
                          gamma=0.1, zeta=0.1, group_size=4)
        z = torch.randn(2, 8, 8)  # 4+4=8
        bsq_loss, quantized, z_indices = bsq(z)
        assert quantized.shape == (2, 8, 8)
        assert z_indices.shape == (2, 8)

    def test_half_mode_indices(self):
        bsq = BSQuantizer(s1_bits=4, s2_bits=4, beta=0.1, gamma0=0.1,
                          gamma=0.1, zeta=0.1, group_size=4)
        z = torch.randn(2, 8, 8)
        bsq_loss, quantized, z_indices = bsq(z, half=True)
        assert isinstance(z_indices, list)
        assert len(z_indices) == 2
        assert z_indices[0].shape == (2, 8)  # s1 indices
        assert z_indices[1].shape == (2, 8)  # s2 indices


# ---------------------------------------------------------------------------
# HierarchicalEmbedding
# ---------------------------------------------------------------------------

class TestHierarchicalEmbedding:
    def test_output_shape(self):
        emb = HierarchicalEmbedding(s1_bits=4, s2_bits=4, d_model=32)
        s1_ids = torch.randint(0, 16, (2, 8))
        s2_ids = torch.randint(0, 16, (2, 8))
        out = emb([s1_ids, s2_ids])
        assert out.shape == (2, 8, 32)

    def test_split_token_correctness(self):
        emb = HierarchicalEmbedding(s1_bits=4, s2_bits=4, d_model=32)
        # Compose a token: s1=3, s2=7 -> token = (3 << 4) | 7 = 55
        token = torch.tensor([[55]])
        s1, s2 = emb.split_token(token, s2_bits=4)
        assert s1.item() == 3
        assert s2.item() == 7

    def test_composite_token_input(self):
        emb = HierarchicalEmbedding(s1_bits=4, s2_bits=4, d_model=32)
        # composite token ids (range 0 to 2^8-1)
        tokens = torch.randint(0, 256, (2, 8))
        out = emb(tokens)
        assert out.shape == (2, 8, 32)


# ---------------------------------------------------------------------------
# TemporalEmbedding
# ---------------------------------------------------------------------------

class TestTemporalEmbedding:
    def test_output_shape_fixed(self):
        te = TemporalEmbedding(d_model=32, learn_pe=False)
        # x has 5 time features: minute, hour, weekday, day, month
        x = torch.stack([
            torch.randint(0, 60, (2, 8)),    # minute
            torch.randint(0, 24, (2, 8)),    # hour
            torch.randint(0, 7, (2, 8)),     # weekday
            torch.randint(1, 32, (2, 8)),    # day
            torch.randint(1, 13, (2, 8)),    # month
        ], dim=-1)  # (2, 8, 5)
        out = te(x)
        assert out.shape == (2, 8, 32)

    def test_output_shape_learnable(self):
        te = TemporalEmbedding(d_model=32, learn_pe=True)
        x = torch.stack([
            torch.randint(0, 60, (2, 8)),
            torch.randint(0, 24, (2, 8)),
            torch.randint(0, 7, (2, 8)),
            torch.randint(1, 32, (2, 8)),
            torch.randint(1, 13, (2, 8)),
        ], dim=-1)
        out = te(x)
        assert out.shape == (2, 8, 32)


# ---------------------------------------------------------------------------
# FixedEmbedding
# ---------------------------------------------------------------------------

class TestFixedEmbedding:
    def test_output_shape(self):
        emb = FixedEmbedding(c_in=60, d_model=32)
        x = torch.randint(0, 60, (2, 8))
        out = emb(x)
        assert out.shape == (2, 8, 32)

    def test_weights_are_non_trainable(self):
        emb = FixedEmbedding(c_in=24, d_model=16)
        assert not emb.emb.weight.requires_grad

    def test_output_is_detached(self):
        emb = FixedEmbedding(c_in=24, d_model=16)
        x = torch.randint(0, 24, (2, 4))
        out = emb(x)
        assert not out.requires_grad
