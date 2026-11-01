import torch

from model import Kronos
from model.module import RotaryPositionalEmbedding


def manual_rotate(x, positions, inv_freq):
    """RoPE-rotate [B, H, L, D] by the given integer positions."""
    freqs = torch.einsum("l,j->lj", positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def test_full_length_rotation_is_unchanged():
    """q_len == k_len (self-attention and training cross-attention)."""
    rotary = RotaryPositionalEmbedding(dim=8)
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn(2, 3, 7, 8)
    q_rot, k_rot = rotary(q, k)
    positions = torch.arange(7)
    assert torch.allclose(q_rot, manual_rotate(q, positions, rotary.inv_freq), atol=1e-6)
    assert torch.allclose(k_rot, manual_rotate(k, positions, rotary.inv_freq), atol=1e-6)


def test_single_query_gets_last_position_and_keys_keep_theirs():
    """decode_s2 passes one query against the whole context (q_len=1)."""
    rotary = RotaryPositionalEmbedding(dim=8)
    seq_len = 10
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, seq_len, 8)
    q_rot, k_rot = rotary(q, k)
    assert torch.allclose(q_rot, manual_rotate(q, torch.tensor([seq_len - 1]), rotary.inv_freq), atol=1e-6)
    assert torch.allclose(k_rot, manual_rotate(k, torch.arange(seq_len), rotary.inv_freq), atol=1e-6)


def test_decode_s2_matches_full_length_dependency_layer():
    """The single-query decode_s2 path must agree with the full-length
    computation used during training; only the last position is consumed."""
    torch.manual_seed(123)
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=False,
    ).eval()

    batch_size, seq_len = 2, 12
    s1_ids = torch.randint(0, 16, (batch_size, seq_len))
    s2_ids = torch.randint(0, 16, (batch_size, seq_len))
    stamp = torch.randint(0, 5, (batch_size, seq_len, 5)).float()

    with torch.no_grad():
        _, context = model.decode_s1(s1_ids, s2_ids, stamp)
        next_s1 = torch.randint(0, 16, (batch_size, 1))

        s2_logits = model.decode_s2(context, next_s1)[:, -1:, :]

        # Reference: run the dependency layer with a full-length query that
        # carries the sampled s1 embedding at every position, as in training.
        sibling_full = model.embedding.emb_s1(next_s1).expand(batch_size, seq_len, model.embedding.d_model)
        reference = model.head.cond_forward(model.dep_layer(context, sibling_full))[:, -1:, :]

    assert torch.allclose(s2_logits, reference, atol=1e-5)
