import pytest
import torch


@pytest.fixture
def small_tokenizer_config():
    """Minimal KronosTokenizer config for fast offline tests."""
    return dict(
        d_in=6,          # OHLCV + amount
        d_model=32,
        n_heads=2,
        ff_dim=64,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        s1_bits=4,
        s2_bits=4,       # even total (8) so half-decode works correctly
        beta=0.1,
        gamma0=0.1,
        gamma=0.1,
        zeta=0.1,
        group_size=4,
    )


@pytest.fixture
def small_model_config():
    """Minimal Kronos model config for fast offline tests."""
    return dict(
        s1_bits=4,
        s2_bits=4,
        n_layers=2,
        d_model=32,
        n_heads=2,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )


@pytest.fixture
def sample_ohlcv_data():
    """Small random OHLCV tensor (batch=2, seq_len=8, features=6)."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 6)
