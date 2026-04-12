"""
Unit tests for KronosTokenizer, Kronos, and shared building blocks.

All tests are offline — no model downloads, no HuggingFace calls.
Models are constructed from scratch with minimal dimensions.

The suite includes a regression test for the `requires_grad` typo fix:
before the fix, ``w.require_grad = False`` silently created an unused
attribute instead of disabling gradient tracking on the tensor.
"""

import pytest
import torch

from model.module import (
    DualHead,
    FixedEmbedding,
    HierarchicalEmbedding,
    RMSNorm,
    RotaryPositionalEmbedding,
    TemporalEmbedding,
    TransformerBlock,
)

# ---------------------------------------------------------------------------
# Shapes shared across tests
# ---------------------------------------------------------------------------
B = 2  # batch size
T = 8  # sequence length
D_IN = 6  # OHLCV + amount
D_MODEL = 32
N_HEADS = 2
FF_DIM = 64
S1_BITS = 4
S2_BITS = 4  # must equal S1_BITS: indices_to_bits uses codebook_dim//2 for both halves
S1_VOCAB = 2**S1_BITS  # 16
S2_VOCAB = 2**S2_BITS  # 16


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(D_MODEL)
        x = torch.randn(B, T, D_MODEL)
        assert norm(x).shape == (B, T, D_MODEL)

    def test_normalized_rms_approx_one(self):
        norm = RMSNorm(D_MODEL)
        # With weight=1 and large random input, RMS of output ≈ 1
        x = torch.randn(B, T, D_MODEL) * 10
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, abs=0.05)


class TestRotaryPositionalEmbedding:
    def test_output_shapes_preserved(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.randn(B, N_HEADS, T, 16)
        k = torch.randn(B, N_HEADS, T, 16)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_cache_reused_for_same_seq_len(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.randn(B, N_HEADS, T, 16)
        k = torch.randn(B, N_HEADS, T, 16)
        rope(q, k)
        cos_first = rope.cos_cached
        rope(q, k)
        assert rope.cos_cached is cos_first  # same object, no recompute


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(D_MODEL, N_HEADS, FF_DIM)
        x = torch.randn(B, T, D_MODEL)
        assert block(x).shape == (B, T, D_MODEL)

    def test_output_shape_with_padding_mask(self):
        block = TransformerBlock(D_MODEL, N_HEADS, FF_DIM)
        x = torch.randn(B, T, D_MODEL)
        # padding_mask: True = masked out
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, -2:] = True  # mask last 2 tokens in first batch item
        assert block(x, key_padding_mask=mask).shape == (B, T, D_MODEL)


class TestDualHead:
    @pytest.fixture(scope="class")
    def head(self):
        return DualHead(S1_BITS, S2_BITS, D_MODEL)

    def test_forward_s1_shape(self, head):
        x = torch.randn(B, T, D_MODEL)
        out = head(x)
        assert out.shape == (B, T, S1_VOCAB)

    def test_cond_forward_s2_shape(self, head):
        x2 = torch.randn(B, T, D_MODEL)
        out = head.cond_forward(x2)
        assert out.shape == (B, T, S2_VOCAB)

    def test_compute_loss_scalar(self, head):
        s1_logits = torch.randn(B, T, S1_VOCAB)
        s2_logits = torch.randn(B, T, S2_VOCAB)
        s1_targets = torch.randint(0, S1_VOCAB, (B, T))
        s2_targets = torch.randint(0, S2_VOCAB, (B, T))
        loss, ce_s1, ce_s2 = head.compute_loss(
            s1_logits, s2_logits, s1_targets, s2_targets
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_compute_loss_with_padding_mask(self, head):
        s1_logits = torch.randn(B, T, S1_VOCAB)
        s2_logits = torch.randn(B, T, S2_VOCAB)
        s1_targets = torch.randint(0, S1_VOCAB, (B, T))
        s2_targets = torch.randint(0, S2_VOCAB, (B, T))
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[0, -2:] = 1  # mask 2 padding positions
        loss, _, _ = head.compute_loss(
            s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=mask
        )
        assert torch.isfinite(loss)


class TestHierarchicalEmbedding:
    @pytest.fixture(scope="class")
    def emb(self):
        return HierarchicalEmbedding(S1_BITS, S2_BITS, D_MODEL)

    def test_forward_tuple_input(self, emb):
        s1 = torch.randint(0, S1_VOCAB, (B, T))
        s2 = torch.randint(0, S2_VOCAB, (B, T))
        out = emb([s1, s2])
        assert out.shape == (B, T, D_MODEL)

    def test_forward_composite_input(self, emb):
        """Composite token IDs should be split identically to explicit (s1, s2) inputs."""
        s1 = torch.randint(0, S1_VOCAB, (B, T))
        s2 = torch.randint(0, S2_VOCAB, (B, T))
        composite = (s1 << S2_BITS) | s2
        out_split = emb([s1, s2])
        out_composite = emb(composite)
        assert torch.allclose(out_split, out_composite)

    def test_split_token_roundtrip(self, emb):
        """split_token must recover s1 and s2 from a composite token."""
        s1_orig = torch.randint(0, S1_VOCAB, (B, T))
        s2_orig = torch.randint(0, S2_VOCAB, (B, T))
        composite = (s1_orig << S2_BITS) | s2_orig
        s1_rec, s2_rec = emb.split_token(composite, S2_BITS)
        assert torch.equal(s1_orig, s1_rec)
        assert torch.equal(s2_orig, s2_rec)


class TestFixedEmbedding:
    def test_requires_grad_fix(self):
        """
        Regression: before the fix ``w.require_grad = False`` created a spurious
        attribute without actually disabling gradients.  After the fix
        ``w.requires_grad = False`` is used, and the resulting embedding weight
        has requires_grad=False.
        """
        emb = FixedEmbedding(c_in=60, d_model=D_MODEL)
        assert not emb.emb.weight.requires_grad

    def test_output_shape(self):
        emb = FixedEmbedding(c_in=60, d_model=D_MODEL)
        x = torch.randint(0, 60, (B, T))
        out = emb(x)
        assert out.shape == (B, T, D_MODEL)

    def test_output_is_detached(self):
        """FixedEmbedding.forward calls .detach() — output must not require grad."""
        emb = FixedEmbedding(c_in=60, d_model=D_MODEL)
        x = torch.randint(0, 60, (B, T))
        out = emb(x)
        assert not out.requires_grad


def make_stamp(batch: int, seq: int) -> torch.Tensor:
    """Build a valid temporal stamp (B, T, 5) respecting each embedding's vocab size.

    Feature order: [minute(0-59), hour(0-23), weekday(0-6), day(0-31), month(0-12)]
    """
    stamp = torch.stack(
        [
            torch.randint(0, 60, (batch, seq)),  # minute
            torch.randint(0, 24, (batch, seq)),  # hour
            torch.randint(0, 7, (batch, seq)),  # weekday
            torch.randint(0, 32, (batch, seq)),  # day
            torch.randint(0, 13, (batch, seq)),  # month
        ],
        dim=-1,
    )
    return stamp


class TestTemporalEmbedding:
    def test_learnable_shape(self):
        te = TemporalEmbedding(D_MODEL, learn_pe=True)
        out = te(make_stamp(B, T))
        assert out.shape == (B, T, D_MODEL)

    def test_fixed_shape(self):
        te = TemporalEmbedding(D_MODEL, learn_pe=False)
        out = te(make_stamp(B, T))
        assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# KronosTokenizer
# ---------------------------------------------------------------------------


class TestKronosTokenizer:
    def test_forward_output_shapes(self, tokenizer):
        x = torch.randn(B, T, D_IN)
        (z_pre, z), bsq_loss, quantized, z_indices = tokenizer(x)
        assert z_pre.shape == (B, T, D_IN)
        assert z.shape == (B, T, D_IN)
        assert bsq_loss.shape == ()
        assert quantized.shape == (B, T, S1_BITS + S2_BITS)

    def test_encode_shapes(self, tokenizer):
        x = torch.randn(B, T, D_IN)
        z_indices = tokenizer.encode(x, half=True)
        assert isinstance(z_indices, list) and len(z_indices) == 2
        s1_idx, s2_idx = z_indices
        assert s1_idx.shape == (B, T)
        assert s2_idx.shape == (B, T)

    def test_encode_index_bounds(self, tokenizer):
        """Encoded s1/s2 indices must be in their respective vocab ranges."""
        x = torch.randn(B, T, D_IN)
        s1_idx, s2_idx = tokenizer.encode(x, half=True)
        assert s1_idx.min() >= 0 and s1_idx.max() < S1_VOCAB
        assert s2_idx.min() >= 0 and s2_idx.max() < S2_VOCAB

    def test_decode_shapes(self, tokenizer):
        x = torch.randn(B, T, D_IN)
        indices = tokenizer.encode(x, half=True)
        recon = tokenizer.decode(indices, half=True)
        assert recon.shape == (B, T, D_IN)

    def test_loss_is_finite(self, tokenizer):
        x = torch.randn(B, T, D_IN)
        _, bsq_loss, _, _ = tokenizer(x)
        assert torch.isfinite(bsq_loss)


# ---------------------------------------------------------------------------
# Kronos
# ---------------------------------------------------------------------------


class TestKronos:
    def test_forward_output_shapes(self, kronos):
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        s1_logits, s2_logits = kronos(s1_ids, s2_ids)
        assert s1_logits.shape == (B, T, S1_VOCAB)
        assert s2_logits.shape == (B, T, S2_VOCAB)

    def test_forward_with_stamp(self, kronos):
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        s1_logits, s2_logits = kronos(s1_ids, s2_ids, stamp=make_stamp(B, T))
        assert s1_logits.shape == (B, T, S1_VOCAB)
        assert s2_logits.shape == (B, T, S2_VOCAB)

    def test_forward_with_padding_mask(self, kronos):
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, -2:] = True
        s1_logits, s2_logits = kronos(s1_ids, s2_ids, padding_mask=mask)
        assert s1_logits.shape == (B, T, S1_VOCAB)
        assert s2_logits.shape == (B, T, S2_VOCAB)

    def test_decode_s1_shapes(self, kronos):
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        s1_logits, context = kronos.decode_s1(s1_ids, s2_ids)
        assert s1_logits.shape == (B, T, S1_VOCAB)
        assert context.shape == (B, T, D_MODEL)

    def test_decode_s2_shapes(self, kronos):
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        _, context = kronos.decode_s1(s1_ids, s2_ids)
        s2_logits = kronos.decode_s2(context, s1_ids)
        assert s2_logits.shape == (B, T, S2_VOCAB)

    def test_logits_are_finite(self, kronos):
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        s1_logits, s2_logits = kronos(s1_ids, s2_ids)
        assert torch.isfinite(s1_logits).all()
        assert torch.isfinite(s2_logits).all()

    def test_teacher_forcing_same_shape(self, kronos):
        """teacher_forcing=True should yield the same output shapes."""
        s1_ids = torch.randint(0, S1_VOCAB, (B, T))
        s2_ids = torch.randint(0, S2_VOCAB, (B, T))
        s1_targets = torch.randint(0, S1_VOCAB, (B, T))
        s1_logits, s2_logits = kronos(
            s1_ids, s2_ids, use_teacher_forcing=True, s1_targets=s1_targets
        )
        assert s1_logits.shape == (B, T, S1_VOCAB)
        assert s2_logits.shape == (B, T, S2_VOCAB)
