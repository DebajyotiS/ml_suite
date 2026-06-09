"""Tests for ml_suite.models.transformer.heads and decoders."""

import pytest
import torch
from torch import nn

from ml_suite.models.transformer.decoders import PatchDecoderND, QuerySetDecoder, TokenDecoder
from ml_suite.models.transformer.heads import (
    ClassificationHead,
    PooledHead,
    RegressionHead,
    TokenwiseHead,
    make_head_mlp,
)


# ---------------------------------------------------------------------------
# make_head_mlp
# ---------------------------------------------------------------------------


def test_make_head_mlp_single_layer_is_linear():
    """A 1-layer MLP with no dropout should be a plain nn.Linear."""
    mlp = make_head_mlp(input_dim=16, output_dim=4)
    assert isinstance(mlp, nn.Linear)


def test_make_head_mlp_single_layer_output_shape():
    """Single-layer head should map (batch, tokens, input) → (batch, tokens, output)."""
    mlp = make_head_mlp(input_dim=16, output_dim=4)
    x = torch.randn(2, 8, 16)
    assert mlp(x).shape == (2, 8, 4)


def test_make_head_mlp_multi_layer_output_shape():
    """Multi-layer head should produce the correct output shape."""
    mlp = make_head_mlp(input_dim=16, output_dim=4, num_layers=3, hidden_dim=32)
    x = torch.randn(2, 8, 16)
    assert mlp(x).shape == (2, 8, 4)


def test_make_head_mlp_with_dropout_wraps_in_sequential():
    """Single-layer head with dropout should return nn.Sequential."""
    mlp = make_head_mlp(input_dim=16, output_dim=4, dropout=0.1)
    assert isinstance(mlp, nn.Sequential)


def test_make_head_mlp_rejects_non_positive_input_dim():
    """Non-positive input_dim must be rejected."""
    with pytest.raises(ValueError, match="input_dim must be positive"):
        make_head_mlp(input_dim=0, output_dim=4)


def test_make_head_mlp_rejects_non_positive_output_dim():
    """Non-positive output_dim must be rejected."""
    with pytest.raises(ValueError, match="output_dim must be positive"):
        make_head_mlp(input_dim=16, output_dim=0)


def test_make_head_mlp_rejects_invalid_dropout():
    """Dropout >= 1.0 must be rejected."""
    with pytest.raises(ValueError, match="dropout must be in"):
        make_head_mlp(input_dim=16, output_dim=4, dropout=1.0)


def test_make_head_mlp_rejects_num_layers_less_than_one():
    """num_layers < 1 must be rejected."""
    with pytest.raises(ValueError, match="num_layers must be at least 1"):
        make_head_mlp(input_dim=16, output_dim=4, num_layers=0)


def test_make_head_mlp_rejects_non_positive_hidden_dim():
    """Non-positive hidden_dim must be rejected."""
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        make_head_mlp(input_dim=16, output_dim=4, num_layers=2, hidden_dim=0)


# ---------------------------------------------------------------------------
# TokenwiseHead
# ---------------------------------------------------------------------------


def test_tokenwise_head_output_shape():
    """TokenwiseHead should map (batch, tokens, embed) → (batch, tokens, output)."""
    head = TokenwiseHead(embedding_dim=16, output_dim=4)
    x = torch.randn(2, 8, 16)
    assert head(x).shape == (2, 8, 4)


def test_tokenwise_head_multi_layer():
    """Multi-layer TokenwiseHead should produce the correct output shape."""
    head = TokenwiseHead(embedding_dim=16, output_dim=4, num_layers=3, hidden_dim=32)
    x = torch.randn(2, 8, 16)
    assert head(x).shape == (2, 8, 4)


# ---------------------------------------------------------------------------
# PooledHead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pooling", ["mean", "max", "cls", "last"])
def test_pooled_head_output_shape(pooling):
    """PooledHead should map (batch, tokens, embed) → (batch, output)."""
    head = PooledHead(embedding_dim=16, output_dim=4, pooling=pooling)
    x = torch.randn(2, 8, 16)
    assert head(x).shape == (2, 4)


def test_pooled_head_with_mask():
    """PooledHead with mask should produce the correct output shape."""
    head = PooledHead(embedding_dim=16, output_dim=4, pooling="mean")
    x = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 6:] = False
    assert head(x, mask=mask).shape == (2, 4)


# ---------------------------------------------------------------------------
# ClassificationHead / RegressionHead
# ---------------------------------------------------------------------------


def test_classification_head_is_pooled_head():
    """ClassificationHead should be a subclass of PooledHead."""
    head = ClassificationHead(embedding_dim=16, output_dim=10)
    assert isinstance(head, PooledHead)


def test_classification_head_output_shape():
    """ClassificationHead output should be (batch, num_classes)."""
    head = ClassificationHead(embedding_dim=16, output_dim=10)
    x = torch.randn(2, 8, 16)
    assert head(x).shape == (2, 10)


def test_regression_head_is_pooled_head():
    """RegressionHead should be a subclass of PooledHead."""
    head = RegressionHead(embedding_dim=16, output_dim=1)
    assert isinstance(head, PooledHead)


def test_regression_head_output_shape():
    """RegressionHead output should be (batch, output_dim)."""
    head = RegressionHead(embedding_dim=16, output_dim=3)
    x = torch.randn(2, 8, 16)
    assert head(x).shape == (2, 3)


# ---------------------------------------------------------------------------
# TokenDecoder
# ---------------------------------------------------------------------------


def test_token_decoder_is_tokenwise_head():
    """TokenDecoder should be a subclass of TokenwiseHead."""
    decoder = TokenDecoder(embedding_dim=16, output_dim=4)
    assert isinstance(decoder, TokenwiseHead)


def test_token_decoder_output_shape():
    """TokenDecoder should preserve token dimension."""
    decoder = TokenDecoder(embedding_dim=16, output_dim=4)
    x = torch.randn(2, 8, 16)
    assert decoder(x).shape == (2, 8, 4)


# ---------------------------------------------------------------------------
# PatchDecoderND
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dim, grid_shape, patch_size, out_channels, expected_shape",
    [
        (1, (8,), 4, 3, (2, 3, 32)),
        (2, (4, 4), 4, 3, (2, 3, 16, 16)),
        (3, (4, 4, 4), 2, 3, (2, 3, 8, 8, 8)),
    ],
)
def test_patch_decoder_output_shape(
    input_dim, grid_shape, patch_size, out_channels, expected_shape
):
    """PatchDecoderND should reconstruct the correct spatial grid."""
    decoder = PatchDecoderND(
        input_dim=input_dim,
        embedding_dim=16,
        out_channels=out_channels,
        patch_size=patch_size,
    )
    num_tokens = 1
    for s in grid_shape:
        num_tokens *= s

    tokens = torch.randn(2, num_tokens, 16)
    out = decoder(tokens, grid_shape=grid_shape)
    assert out.shape == expected_shape


def test_patch_decoder_rejects_invalid_input_dim():
    """input_dim outside {1, 2, 3} must be rejected."""
    with pytest.raises(ValueError, match="input_dim must be 1, 2, or 3"):
        PatchDecoderND(input_dim=4, embedding_dim=16, out_channels=3, patch_size=4)


def test_patch_decoder_rejects_non_positive_out_channels():
    """Non-positive out_channels must be rejected."""
    with pytest.raises(ValueError, match="out_channels must be positive"):
        PatchDecoderND(input_dim=2, embedding_dim=16, out_channels=0, patch_size=4)


def test_patch_decoder_rejects_wrong_grid_shape_length():
    """grid_shape length must match input_dim."""
    decoder = PatchDecoderND(input_dim=2, embedding_dim=16, out_channels=3, patch_size=4)
    tokens = torch.randn(2, 8, 16)
    with pytest.raises(ValueError, match="grid_shape must have length"):
        decoder(tokens, grid_shape=(8,))


def test_patch_decoder_rejects_token_count_grid_mismatch():
    """Mismatch between token count and grid shape must be rejected."""
    decoder = PatchDecoderND(input_dim=2, embedding_dim=16, out_channels=3, patch_size=4)
    tokens = torch.randn(2, 9, 16)  # 9 tokens but grid (4, 4) = 16
    with pytest.raises(ValueError, match="Number of tokens"):
        decoder(tokens, grid_shape=(4, 4))


def test_patch_decoder_rejects_wrong_rank_tokens():
    """Non-3D token tensor must be rejected."""
    decoder = PatchDecoderND(input_dim=2, embedding_dim=16, out_channels=3, patch_size=4)
    tokens = torch.randn(2, 16)
    with pytest.raises(ValueError, match="must have shape"):
        decoder(tokens, grid_shape=(4, 4))


# ---------------------------------------------------------------------------
# QuerySetDecoder
# ---------------------------------------------------------------------------


def test_query_set_decoder_output_shape():
    """QuerySetDecoder should produce (batch, num_queries, output_dim)."""
    decoder = QuerySetDecoder(latent_dim=16, output_dim=4, num_queries=8, query_dim=16)
    latent = torch.randn(2, 16)
    out = decoder(latent)
    assert out.shape == (2, 8, 4)


def test_query_set_decoder_rejects_non_positive_latent_dim():
    """Non-positive latent_dim must be rejected."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        QuerySetDecoder(latent_dim=0, output_dim=4, num_queries=8, query_dim=16)


def test_query_set_decoder_rejects_non_positive_num_queries():
    """Non-positive num_queries must be rejected."""
    with pytest.raises(ValueError, match="num_queries must be positive"):
        QuerySetDecoder(latent_dim=16, output_dim=4, num_queries=0, query_dim=16)


def test_query_set_decoder_rejects_wrong_rank_latent():
    """Non-2D latent must be rejected."""
    decoder = QuerySetDecoder(latent_dim=16, output_dim=4, num_queries=8, query_dim=16)
    latent = torch.randn(2, 4, 16)
    with pytest.raises(ValueError, match="must have shape"):
        decoder(latent)


def test_query_set_decoder_queries_are_learnable_parameter():
    """queries should be an nn.Parameter."""
    decoder = QuerySetDecoder(latent_dim=16, output_dim=4, num_queries=8, query_dim=16)
    assert isinstance(decoder.queries, torch.nn.Parameter)
    assert decoder.queries.shape == (8, 16)
