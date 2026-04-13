"""Unit tests for KronosTokenizer encode/decode and forward pass."""

import pytest
import torch

from model.kronos import KronosTokenizer


class TestTokenizerEncodeDecode:
    """Encode/decode roundtrip and shape validation."""

    @pytest.fixture
    def tokenizer(self, small_tokenizer_config):
        torch.manual_seed(42)
        tok = KronosTokenizer(**small_tokenizer_config)
        tok.eval()
        return tok

    def test_encode_decode_roundtrip(self, tokenizer, sample_ohlcv_data):
        """Encode then decode should reconstruct approximately."""
        x = sample_ohlcv_data
        with torch.no_grad():
            indices = tokenizer.encode(x, half=False)
            reconstructed = tokenizer.decode(indices, half=False)
        assert reconstructed.shape == x.shape
        # Reconstruction should be finite
        assert torch.isfinite(reconstructed).all()

    def test_encode_half_mode(self, tokenizer, sample_ohlcv_data):
        """half=True should return a list of two index tensors."""
        x = sample_ohlcv_data
        with torch.no_grad():
            indices = tokenizer.encode(x, half=True)
        assert isinstance(indices, list)
        assert len(indices) == 2
        # Each should have shape (batch, seq_len)
        assert indices[0].shape == (2, 8)
        assert indices[1].shape == (2, 8)

    def test_decode_half_mode(self):
        """Decode with half=True from half-encoded indices (requires even codebook_dim)."""
        # Use even codebook_dim (s1_bits=4, s2_bits=4 -> codebook_dim=8) so
        # indices_to_bits half-split produces correct width for post_quant_embed.
        torch.manual_seed(42)
        cfg = dict(
            d_in=6, d_model=32, n_heads=2, ff_dim=64,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4,
            beta=0.1, gamma0=0.1, gamma=0.1, zeta=0.1, group_size=4,
        )
        tok = KronosTokenizer(**cfg)
        tok.eval()
        x = torch.randn(2, 8, 6)
        with torch.no_grad():
            indices = tok.encode(x, half=True)
            reconstructed = tok.decode(indices, half=True)
        assert reconstructed.shape == x.shape
        assert torch.isfinite(reconstructed).all()

    def test_shape_various_seq_lengths(self, small_tokenizer_config):
        """Various sequence lengths should produce correct output shapes."""
        torch.manual_seed(42)
        tok = KronosTokenizer(**small_tokenizer_config)
        tok.eval()

        for seq_len in [1, 4, 16, 32]:
            x = torch.randn(1, seq_len, 6)
            with torch.no_grad():
                indices = tok.encode(x)
            assert indices.shape == (1, seq_len), f"Failed for seq_len={seq_len}"

    def test_batch_size_independence(self, small_tokenizer_config):
        """Different batch sizes should work correctly."""
        torch.manual_seed(42)
        tok = KronosTokenizer(**small_tokenizer_config)
        tok.eval()

        for batch_size in [1, 3, 5]:
            x = torch.randn(batch_size, 8, 6)
            with torch.no_grad():
                indices = tok.encode(x)
            assert indices.shape[0] == batch_size


class TestTokenizerForward:
    """Test full forward pass returns expected structure."""

    @pytest.fixture
    def tokenizer(self, small_tokenizer_config):
        torch.manual_seed(42)
        tok = KronosTokenizer(**small_tokenizer_config)
        tok.eval()
        return tok

    def test_forward_returns_tuple_structure(self, tokenizer, sample_ohlcv_data):
        """Forward should return ((z_pre, z), bsq_loss, quantized, z_indices)."""
        x = sample_ohlcv_data
        result = tokenizer(x)
        assert len(result) == 4

        (z_pre, z), bsq_loss, quantized, z_indices = result

        # z_pre and z should have same shape as input
        assert z_pre.shape == x.shape
        assert z.shape == x.shape

        # bsq_loss should be a scalar
        assert bsq_loss.dim() == 0

        # quantized shape: (batch, seq, codebook_dim)
        codebook_dim = tokenizer.s1_bits + tokenizer.s2_bits  # 4+4=8
        assert quantized.shape == (2, 8, codebook_dim)

        # z_indices shape: (batch, seq)
        assert z_indices.shape == (2, 8)

    def test_forward_loss_is_finite(self, tokenizer, sample_ohlcv_data):
        result = tokenizer(sample_ohlcv_data)
        _, bsq_loss, _, _ = result
        assert torch.isfinite(bsq_loss)

    def test_forward_train_vs_eval(self, small_tokenizer_config, sample_ohlcv_data):
        """Train and eval modes should both work without error."""
        torch.manual_seed(42)
        tok = KronosTokenizer(**small_tokenizer_config)

        tok.train()
        result_train = tok(sample_ohlcv_data)
        assert len(result_train) == 4

        tok.eval()
        result_eval = tok(sample_ohlcv_data)
        assert len(result_eval) == 4
