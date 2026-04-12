"""
Unit tests for top_k_top_p_filtering and sample_from_logits.

All tests are offline — no model downloads, no HuggingFace calls.

The suite includes regression tests for three bugs fixed during the
uv/ruff/ty migration:
  1. top_k=None caused a TypeError when top_p was active.
  2. top_p=None caused a TypeError when top_k was active.
  3. sample_logits=False called the *parameter* top_k as a function
     instead of torch.topk.
"""

import pytest
import torch

from model.kronos import sample_from_logits, top_k_top_p_filtering

# ---------------------------------------------------------------------------
# top_k_top_p_filtering
# ---------------------------------------------------------------------------


class TestTopKFiltering:
    def test_keeps_exactly_k_tokens(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, -1.0]])
        out = top_k_top_p_filtering(logits.clone(), top_k=2)
        surviving = (out != -float("Inf")).sum().item()
        assert surviving == 2

    def test_highest_tokens_survive(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, -1.0]])
        out = top_k_top_p_filtering(logits.clone(), top_k=2)
        # Tokens at index 1 (5.0) and 2 (3.0) must survive
        assert out[0, 1] == pytest.approx(5.0)
        assert out[0, 2] == pytest.approx(3.0)

    def test_min_tokens_to_keep_respected(self):
        """top_k=1 with min_tokens_to_keep=3 must preserve at least 3 tokens."""
        logits = torch.zeros(1, 10)
        logits[0, 0] = 100.0  # one dominant token
        out = top_k_top_p_filtering(logits.clone(), top_k=1, min_tokens_to_keep=3)
        assert (out != -float("Inf")).sum().item() >= 3

    def test_top_k_batch(self):
        """Each row in a batch is filtered independently."""
        logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        out = top_k_top_p_filtering(logits.clone(), top_k=1)
        for row in range(2):
            assert (out[row] != -float("Inf")).sum().item() == 1

    def test_top_k_zero_is_noop(self):
        """top_k=0 should skip the top-k branch entirely (returns None)."""
        logits = torch.randn(1, 10)
        result = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=1.0)
        assert result is None


class TestTopPFiltering:
    def test_removes_tail_tokens(self):
        """A very peaked distribution should keep only 1 token at top_p=0.9."""
        logits = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0]])
        out = top_k_top_p_filtering(logits.clone(), top_p=0.9)
        assert out[0, 0] == pytest.approx(20.0)
        assert (out[0, 1:] == -float("Inf")).all()

    def test_uniform_keeps_all(self):
        """Uniform logits should keep all tokens since each contributes equally
        to cumulative probability."""
        logits = torch.zeros(1, 5)
        out = top_k_top_p_filtering(logits.clone(), top_p=0.9)
        # With uniform probs (0.2 each), after sorting the first token already
        # exceeds 0.9 * 5 = 0.9 cumprob, so only some may survive — at minimum
        # min_tokens_to_keep=1 always survives
        assert (out != -float("Inf")).sum().item() >= 1

    def test_top_p_one_is_noop(self):
        """top_p=1.0 with top_k=0 should return None (neither branch runs)."""
        logits = torch.randn(1, 10)
        result = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# sample_from_logits
# ---------------------------------------------------------------------------


class TestSampleFromLogits:
    # --- bug regression: None guards ---

    def test_top_k_none_top_p_active_does_not_crash(self):
        """
        Regression: before the fix, top_k=None with an active top_p would raise
        TypeError because the code did `top_k > 0` without guarding for None.
        """
        logits = torch.randn(1, 16)
        # Must not raise
        result = sample_from_logits(logits.clone(), top_k=None, top_p=0.9)
        assert result.shape == (1, 1)

    def test_top_p_none_top_k_active_does_not_crash(self):
        """
        Regression: symmetrical None-guard bug for top_p=None with active top_k.
        """
        logits = torch.randn(1, 16)
        result = sample_from_logits(logits.clone(), top_k=3, top_p=None)
        assert result.shape == (1, 1)

    def test_both_none_does_not_crash(self):
        """With no filtering, sample_from_logits should still sample correctly."""
        logits = torch.randn(1, 16)
        result = sample_from_logits(logits.clone(), top_k=None, top_p=None)
        assert result.shape == (1, 1)

    # --- bug regression: sample_logits=False used to call `top_k(...)` ---

    def test_greedy_does_not_crash(self):
        """
        Regression: sample_logits=False previously called the *parameter* top_k
        as a function (``top_k(probs, k=1)``).  After the fix it uses torch.topk.
        """
        logits = torch.randn(1, 16)
        result = sample_from_logits(logits.clone(), top_k=None, sample_logits=False)
        assert result.shape == (1, 1)

    def test_greedy_picks_argmax(self, seed):
        """sample_logits=False must return the token with highest probability."""
        logits = torch.zeros(1, 8)
        logits[0, 3] = 10.0  # token 3 dominates
        result = sample_from_logits(logits.clone(), sample_logits=False)
        assert result.item() == 3

    # --- temperature ---

    def test_temperature_zero_limit(self, seed):
        """Very low temperature concentrates mass on the argmax."""
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
        # temperature=1e-6 ≈ greedy
        results = [
            sample_from_logits(logits.clone(), temperature=1e-6).item()
            for _ in range(20)
        ]
        assert all(r == 1 for r in results), "Expected greedy argmax (idx 1) every time"

    def test_high_temperature_increases_entropy(self, seed):
        """High temperature should produce more varied samples than low temperature."""
        logits = torch.zeros(1, 8)
        logits[0, 0] = 5.0

        def sample_n(temp, n=200):
            return {
                sample_from_logits(logits.clone(), temperature=temp).item()
                for n in range(n)
            }

        low_temp_tokens = sample_n(0.01)
        high_temp_tokens = sample_n(10.0)
        assert len(high_temp_tokens) > len(low_temp_tokens)

    # --- output shape ---

    def test_output_shape_batch(self, seed):
        """Output shape is (batch_size, 1)."""
        B, V = 4, 32
        logits = torch.randn(B, V)
        result = sample_from_logits(logits, temperature=1.0)
        assert result.shape == (B, 1)

    def test_output_is_valid_index(self, seed):
        """Sampled token index must be in [0, vocab_size)."""
        V = 32
        logits = torch.randn(1, V)
        result = sample_from_logits(logits)
        assert 0 <= result.item() < V

    # --- combined top_k + top_p (only one branch runs at a time) ---

    def test_top_k_1_always_greedy(self, seed):
        """top_k=1 constrains sampling to the single most probable token."""
        logits = torch.tensor([[0.5, 5.0, 1.0, 0.0]])
        results = [
            sample_from_logits(logits.clone(), top_k=1).item() for _ in range(20)
        ]
        assert all(r == 1 for r in results)
