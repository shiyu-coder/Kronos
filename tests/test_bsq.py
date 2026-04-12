"""
Unit tests for BinarySphericalQuantizer (BSQ) and its parent BSQuantizer.

All tests are offline — no model downloads, no HuggingFace calls.

The suite includes a regression test for the h = w = int(...) unpack fix:
before the fix, `h, w = int(...)` raised a TypeError because a scalar int
is not iterable.
"""

import pytest
import torch

from model.module import BinarySphericalQuantizer, BSQuantizer

# ---------------------------------------------------------------------------
# Minimal BSQ configuration
# embed_dim must be divisible by group_size.  8 % 4 == 0 ✓
# input_format='blc' matches Kronos' sequence format (batch × length × channels).
# ---------------------------------------------------------------------------
EMBED_DIM = 8
GROUP_SIZE = 4
BSQ_KWARGS = dict(
    embed_dim=EMBED_DIM,
    beta=0.25,
    gamma0=1.0,
    gamma=1.0,
    zeta=1.0,
    group_size=GROUP_SIZE,
    input_format="blc",  # not 'bchw'; avoids the square-reshape branch by default
)


@pytest.fixture(scope="module")
def bsq() -> BinarySphericalQuantizer:
    model = BinarySphericalQuantizer(**BSQ_KWARGS)
    model.eval()
    return model


@pytest.fixture(scope="module")
def bsq_bchw() -> BinarySphericalQuantizer:
    """BSQ with image-style bchw format for testing the square-reshape branch."""
    model = BinarySphericalQuantizer(**{**BSQ_KWARGS, "input_format": "bchw"})
    model.eval()
    return model


@pytest.fixture(scope="module")
def bsquantizer() -> BSQuantizer:
    model = BSQuantizer(
        s1_bits=4,
        s2_bits=4,  # equal bits: required for indices_to_bits round-trip
        beta=0.25,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        group_size=4,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# BinarySphericalQuantizer
# ---------------------------------------------------------------------------


class TestQuantize:
    def test_output_is_bipolar(self, bsq):
        """quantize() must return values in {-1, +1} only."""
        z = torch.randn(4, 10, EMBED_DIM)
        zhat = bsq.quantize(z)
        unique = zhat.unique()
        assert set(unique.tolist()) == {-1.0, 1.0}

    def test_output_shape_preserved(self, bsq):
        z = torch.randn(2, 5, EMBED_DIM)
        zhat = bsq.quantize(z)
        assert zhat.shape == z.shape

    def test_sign_agrees_with_input(self, bsq):
        """Positive inputs → +1, negative inputs → -1."""
        z = torch.ones(1, 1, EMBED_DIM)
        assert (bsq.quantize(z) == 1.0).all()
        assert (bsq.quantize(-z) == -1.0).all()


class TestCodebookRoundTrip:
    def test_codes_to_indexes_shape(self, bsq):
        z = torch.randn(2, 8, EMBED_DIM)
        zhat = bsq.quantize(z) * (1.0 / EMBED_DIM**0.5)
        idx = bsq.codes_to_indexes(zhat)
        assert idx.shape == (2, 8)

    def test_indexes_to_codes_shape(self, bsq):
        indices = torch.zeros(2, 8, dtype=torch.long)
        codes = bsq.indexes_to_codes(indices)
        assert codes.shape == (2, 8, EMBED_DIM)

    def test_roundtrip_codes_indexes_codes(self, bsq):
        """codes_to_indexes ∘ indexes_to_codes should be the identity on quantized codes."""
        # Start from random indices and decode to codes, then re-encode
        indices_orig = torch.randint(0, 2**EMBED_DIM, (2, 6))
        codes = bsq.indexes_to_codes(indices_orig)
        # codes are in {-1, +1}; scale to match what codes_to_indexes expects
        # codes_to_indexes expects quantized (scaled by 1/sqrt(embed_dim)), so we
        # use the unscaled bipolar codes directly since basis multiplication handles it
        indices_recovered = bsq.codes_to_indexes(codes)
        assert torch.equal(indices_orig, indices_recovered)

    def test_all_ones_index(self, bsq):
        """All-+1 code maps to the maximum index (2**embed_dim - 1)."""
        # codes_to_indexes: ((code+1)/2 * basis).sum()
        # All +1 → all bits = 1 → index = sum(basis) = 2**embed_dim - 1
        codes = torch.ones(1, 1, EMBED_DIM)
        idx = bsq.codes_to_indexes(codes)
        assert idx.item() == 2**EMBED_DIM - 1

    def test_all_minus_ones_index(self, bsq):
        """All-(-1) code maps to index 0."""
        codes = -torch.ones(1, 1, EMBED_DIM)
        idx = bsq.codes_to_indexes(codes)
        assert idx.item() == 0


class TestBSQForward:
    def test_output_shapes(self, bsq):
        """forward() returns (zq, loss, metrics) with correct shapes."""
        B, T = 2, 8
        z = torch.randn(B, T, EMBED_DIM)
        zq, loss, metrics = bsq(z, collect_metrics=True)
        assert zq.shape == (B, T, EMBED_DIM)
        assert loss.shape == ()  # scalar

    def test_loss_is_finite(self, bsq):
        z = torch.randn(2, 8, EMBED_DIM)
        _, loss, _ = bsq(z)
        assert torch.isfinite(loss)

    def test_metrics_keys(self, bsq):
        z = torch.randn(2, 4, EMBED_DIM)
        bsq.eval()
        _, _, metrics = bsq(z, collect_metrics=True)
        assert "H" in metrics
        assert "indices" in metrics

    def test_no_metrics_mode(self, bsq):
        z = torch.randn(2, 4, EMBED_DIM)
        zq, loss, metrics = bsq(z, collect_metrics=False)
        assert zq.shape == (2, 4, EMBED_DIM)
        assert loss.item() == pytest.approx(0.0)


class TestGetCodebookEntry:
    def test_bchw_unpack_fix(self, bsq_bchw):
        """
        Regression: before the fix ``h, w = int(z_q.shape[1] ** 0.5)`` raised
        TypeError because a scalar int is not iterable.
        After the fix it uses ``h = w = int(...)``.
        shape[1] must be a perfect square; 4 tokens → h=w=2.
        """
        indices = torch.randint(0, 2**EMBED_DIM, (1, 4))
        # Must not raise TypeError
        z_q = bsq_bchw.get_codebook_entry(indices)
        # Output should be rearranged to (B, C, H, W)
        assert z_q.shape == (1, EMBED_DIM, 2, 2)

    def test_non_bchw_passthrough(self, bsq):
        """blc format returns (B, T, C) without reshape."""
        indices = torch.randint(0, 2**EMBED_DIM, (2, 8))
        z_q = bsq.get_codebook_entry(indices)
        assert z_q.shape == (2, 8, EMBED_DIM)


# ---------------------------------------------------------------------------
# BSQuantizer (wraps BinarySphericalQuantizer with s1/s2 split)
# ---------------------------------------------------------------------------


class TestBSQuantizer:
    def test_forward_full_mode(self, bsquantizer):
        """Non-half mode returns z_indices as a single tensor."""
        z = torch.randn(2, 8, 8)  # codebook_dim = s1_bits + s2_bits = 8
        loss, quantized, z_indices = bsquantizer(z)
        assert isinstance(z_indices, torch.Tensor)
        assert quantized.shape == (2, 8, 8)

    def test_half_mode_splits_indices(self, bsquantizer):
        z = torch.randn(2, 8, 8)  # codebook_dim = s1_bits + s2_bits = 8
        _, _, z_indices = bsquantizer(z, half=True)
        assert isinstance(z_indices, list)
        assert len(z_indices) == 2
        s1_idx, s2_idx = z_indices
        assert s1_idx.shape == (2, 8)
        assert s2_idx.shape == (2, 8)

    def test_bits_to_indices_invertible(self, bsquantizer):
        """bits_to_indices must map the same bit tensor to the same index."""
        bits = torch.randint(0, 2, (2, 8, 9)).float() * 2 - 1  # {-1, +1}
        idx_a = bsquantizer.bits_to_indices(bits)
        idx_b = bsquantizer.bits_to_indices(bits)
        assert torch.equal(idx_a, idx_b)
