"""Regression test for key_padding_mask handling in the RoPE attention modules.

``key_padding_mask`` follows the convention used throughout the rest of the
codebase (see ``DualHead.compute_loss``): 0 marks a real token, a nonzero
value marks padding. A correctly masked attention layer must ignore the
content of padded key/value positions entirely, so overwriting them with
garbage must not change the output at the valid query positions.
"""
import torch

from model.module import MultiHeadAttentionWithRoPE, MultiHeadCrossAttentionWithRoPE


def _padding_mask(batch_size: int, seq_len: int, n_valid: int) -> torch.Tensor:
    """0 for the first ``n_valid`` positions, 1 (padding) after that."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask[:, n_valid:] = 1
    return mask


def test_self_attention_ignores_padded_key_value_content():
    torch.manual_seed(0)
    d_model, n_heads, seq_len, n_valid = 16, 4, 6, 4

    attn = MultiHeadAttentionWithRoPE(d_model, n_heads)
    attn.eval()

    x = torch.randn(1, seq_len, d_model)
    mask = _padding_mask(1, seq_len, n_valid)

    with torch.no_grad():
        out_a = attn(x, key_padding_mask=mask)

        x_perturbed = x.clone()
        x_perturbed[:, n_valid:, :] = torch.randn_like(x_perturbed[:, n_valid:, :]) * 100
        out_b = attn(x_perturbed, key_padding_mask=mask)

    # Only the padded positions' content changed; the valid query positions'
    # output must be identical if padding is excluded from attention.
    torch.testing.assert_close(out_a[:, :n_valid, :], out_b[:, :n_valid, :])


def test_self_attention_stays_causal_with_a_padding_mask():
    """Padding support must not come at the cost of causal masking."""
    torch.manual_seed(0)
    d_model, n_heads, seq_len, n_valid = 16, 4, 6, 4

    attn = MultiHeadAttentionWithRoPE(d_model, n_heads)
    attn.eval()

    x = torch.randn(1, seq_len, d_model)
    mask = _padding_mask(1, seq_len, n_valid)

    with torch.no_grad():
        out_a = attn(x, key_padding_mask=mask)

        # Changing a later valid position (index 1, still < n_valid) must not change
        # an earlier query position's (index 0) output.
        x_perturbed = x.clone()
        x_perturbed[:, 1, :] = torch.randn(d_model) * 100
        out_b = attn(x_perturbed, key_padding_mask=mask)

    torch.testing.assert_close(out_a[:, 0, :], out_b[:, 0, :])


def test_cross_attention_ignores_padded_key_value_content():
    torch.manual_seed(0)
    d_model, n_heads, seq_len, n_valid = 16, 4, 6, 4

    attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
    attn.eval()

    # DependencyAwareLayer, the only caller, always passes a query and a
    # key/value of the same length (sibling_embed and hidden_states come
    # from the same sequence), so this test matches real usage.
    query = torch.randn(1, seq_len, d_model)
    key = torch.randn(1, seq_len, d_model)
    value = torch.randn(1, seq_len, d_model)
    mask = _padding_mask(1, seq_len, n_valid)

    with torch.no_grad():
        out_a = attn(query, key, value, key_padding_mask=mask)

        key_perturbed = key.clone()
        value_perturbed = value.clone()
        key_perturbed[:, n_valid:, :] = torch.randn_like(key_perturbed[:, n_valid:, :]) * 100
        value_perturbed[:, n_valid:, :] = torch.randn_like(value_perturbed[:, n_valid:, :]) * 100
        out_b = attn(query, key_perturbed, value_perturbed, key_padding_mask=mask)

    torch.testing.assert_close(out_a, out_b)
