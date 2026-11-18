import torch

from model.module import MultiHeadAttentionWithRoPE, MultiHeadCrossAttentionWithRoPE


def _padding_mask(batch_size, seq_len, n_valid):
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
        x_p = x.clone()
        x_p[:, n_valid:] = torch.randn_like(x_p[:, n_valid:]) * 100
        out_b = attn(x_p, key_padding_mask=mask)
    torch.testing.assert_close(out_a[:, :n_valid], out_b[:, :n_valid])


def test_self_attention_stays_causal_with_a_padding_mask():
    torch.manual_seed(0)
    d_model, n_heads, seq_len, n_valid = 16, 4, 6, 4
    attn = MultiHeadAttentionWithRoPE(d_model, n_heads)
    attn.eval()
    x = torch.randn(1, seq_len, d_model)
    mask = _padding_mask(1, seq_len, n_valid)
    with torch.no_grad():
        out_a = attn(x, key_padding_mask=mask)
        x_p = x.clone()
        x_p[:, 1] = torch.randn(d_model) * 100
        out_b = attn(x_p, key_padding_mask=mask)
    torch.testing.assert_close(out_a[:, 0], out_b[:, 0])


def test_cross_attention_ignores_padded_key_value_content():
    torch.manual_seed(0)
    d_model, n_heads, seq_len, n_valid = 16, 4, 6, 4
    attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
    attn.eval()
    query = torch.randn(1, seq_len, d_model)
    key = torch.randn(1, seq_len, d_model)
    value = torch.randn(1, seq_len, d_model)
    mask = _padding_mask(1, seq_len, n_valid)
    with torch.no_grad():
        out_a = attn(query, key, value, key_padding_mask=mask)
        key_p, value_p = key.clone(), value.clone()
        key_p[:, n_valid:] = torch.randn_like(key_p[:, n_valid:]) * 100
        value_p[:, n_valid:] = torch.randn_like(value_p[:, n_valid:]) * 100
        out_b = attn(query, key_p, value_p, key_padding_mask=mask)
    torch.testing.assert_close(out_a, out_b)


def test_cross_attention_ignores_padded_key_value_content_train_mode():
    torch.manual_seed(0)
    d_model, n_heads, seq_len, n_valid = 16, 4, 6, 4
    attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
    attn.train()
    query = torch.randn(1, seq_len, d_model)
    key = torch.randn(1, seq_len, d_model)
    value = torch.randn(1, seq_len, d_model)
    mask = _padding_mask(1, seq_len, n_valid)
    with torch.no_grad():
        out_a = attn(query, key, value, key_padding_mask=mask)
        key_p, value_p = key.clone(), value.clone()
        key_p[:, n_valid:] = torch.randn_like(key_p[:, n_valid:]) * 100
        value_p[:, n_valid:] = torch.randn_like(value_p[:, n_valid:]) * 100
        out_b = attn(query, key_p, value_p, key_padding_mask=mask)
    torch.testing.assert_close(out_a, out_b)
