"""
Shared fixtures for offline unit tests.  No network access required.
"""

import pytest
import torch

from model.kronos import Kronos, KronosTokenizer

# ---------------------------------------------------------------------------
# Minimal dimensions that satisfy all internal constraints:
#   - d_model divisible by n_heads  (32 / 2 = 16 ✓)
#   - (s1_bits + s2_bits) divisible by group_size  (4+5 = 9, 9%9 = 0 ✓)
# ---------------------------------------------------------------------------
TOKENIZER_CFG = dict(
    d_in=6,
    d_model=32,
    n_heads=2,
    ff_dim=64,
    n_enc_layers=2,
    n_dec_layers=2,
    ffn_dropout_p=0.0,
    attn_dropout_p=0.0,
    resid_dropout_p=0.0,
    # s1_bits == s2_bits required: indices_to_bits uses codebook_dim//2 as the
    # mask length for both halves, so an odd or asymmetric split breaks decode.
    s1_bits=4,
    s2_bits=4,
    beta=0.25,
    gamma0=1.0,
    gamma=1.0,
    zeta=1.0,
    group_size=4,  # embed_dim (8) must be divisible by group_size
)

KRONOS_CFG = dict(
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


@pytest.fixture(scope="module")
def tokenizer() -> KronosTokenizer:
    model = KronosTokenizer(**TOKENIZER_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def kronos() -> Kronos:
    model = Kronos(**KRONOS_CFG)
    model.eval()
    return model


@pytest.fixture
def seed():
    """Reset torch RNG before each test that needs it."""
    torch.manual_seed(0)
