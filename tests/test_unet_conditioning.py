import pytest
import torch

from ml_suite.models.unet.conditioning import (
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    ConditioningBuilder,
)


def test_sinusoidal_time_embedding_shapes():
    emb = SinusoidalTimeEmbedding(7)
    t = torch.arange(4)

    assert emb(t).shape == (4, 7)
    assert emb(t[:, None]).shape == (4, 7)


def test_sinusoidal_time_embedding_rejects_bad_shape():
    emb = SinusoidalTimeEmbedding(8)

    with pytest.raises(ValueError, match="time"):
        emb(torch.randn(2, 3))


@pytest.mark.parametrize("embedding_type", ["sinusoidal", "learned"])
def test_time_embedding_mlp_shapes(embedding_type):
    mlp = TimeEmbeddingMLP(condition_dim=16, embedding_type=embedding_type)
    assert mlp(torch.arange(4)).shape == (4, 16)


def test_conditioning_builder_time_only():
    builder = ConditioningBuilder(condition_dim=16, time_conditioning=True)

    condition = builder(
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        time=torch.arange(4),
    )

    assert condition.shape == (4, 16)


def test_conditioning_builder_class_and_global_context():
    builder = ConditioningBuilder(
        condition_dim=16,
        num_classes=5,
        global_context_dim=3,
    )

    condition = builder(
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        class_labels=torch.tensor([0, 1, 2, 3]),
        global_context=torch.randn(4, 3),
    )

    assert condition.shape == (4, 16)


def test_conditioning_builder_missing_required_inputs_raise():
    builder = ConditioningBuilder(
        condition_dim=16,
        time_conditioning=True,
        num_classes=5,
        global_context_dim=3,
    )

    with pytest.raises(ValueError, match="time"):
        builder(batch_size=4, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match="class_labels"):
        builder(
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time=torch.arange(4),
        )

    with pytest.raises(ValueError, match="global_context"):
        builder(
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time=torch.arange(4),
            class_labels=torch.tensor([0, 1, 2, 3]),
        )


def test_conditioning_builder_rejects_unconfigured_inputs():
    builder = ConditioningBuilder(condition_dim=16)

    with pytest.raises(ValueError, match="time"):
        builder(
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time=torch.arange(4),
        )

    with pytest.raises(ValueError, match="class_labels"):
        builder(
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            class_labels=torch.tensor([0, 1, 2, 3]),
        )

    with pytest.raises(ValueError, match="global_context"):
        builder(
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            global_context=torch.randn(4, 3),
        )
