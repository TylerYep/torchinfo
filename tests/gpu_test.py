import pytest
import torch

from torchinfo import summary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU must be enabled.")
class TestGPU:
    """GPU-only tests."""

    @staticmethod
    def test_single_layer_network_on_gpu() -> None:
        model = torch.nn.Linear(2, 5)
        model.cuda()

        results = summary(model, input_size=(1, 2))

        assert results.total_params == 15
        assert results.trainable_params == 15

    @staticmethod
    def test_single_layer_network_on_gpu_device() -> None:
        model = torch.nn.Linear(2, 5)

        results = summary(model, input_size=(1, 2), device="cuda")

        assert results.total_params == 15
        assert results.trainable_params == 15
