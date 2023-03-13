import pytest
import torch

from tests.fixtures.models import MultiDeviceModel, SingleInputNet
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

    @staticmethod
    def test_input_size_half_precision() -> None:
        # run this test case in gpu since
        # half precision is not supported in pytorch v1.12
        test = torch.nn.Linear(2, 5).half().to(torch.device("cuda"))
        with pytest.warns(
            UserWarning,
            match=(
                "Half precision is not supported with input_size parameter, and "
                "may output incorrect results. Try passing input_data directly."
            ),
        ):
            summary(test, dtypes=[torch.float16], input_size=(10, 2), device="cuda")

    @staticmethod
    def test_device() -> None:
        model = SingleInputNet()
        # input_size
        summary(model, input_size=(5, 1, 28, 28), device="cuda")

        # input_data
        model = SingleInputNet()
        input_data = torch.randn(5, 1, 28, 28)
        summary(model, input_data=input_data)
        summary(model, input_data=input_data, device="cuda")
        summary(model, input_data=input_data.to("cuda"))
        summary(model, input_data=input_data.to("cuda"), device=torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.device_count() >= 2, reason="Only relevant for multi-GPU"
)
class TestMultiGPU:
    """multi-GPU-only tests"""

    @staticmethod
    def test_model_stays_on_device_if_gpu() -> None:
        model = torch.nn.Linear(10, 10).to("cuda:1")
        summary(model)
        model_parameter = next(model.parameters())
        assert model_parameter.device == torch.device("cuda:1")

    @staticmethod
    def test_different_model_parts_on_different_devices() -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10).to(1), torch.nn.Linear(10, 10).to(0)
        )
        summary(model)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Need CUDA to test parallelism."
)
def test_device_parallelism() -> None:
    model = MultiDeviceModel("cpu", "cuda")
    input_data = torch.randn(10)
    summary(model, input_data=input_data)
    assert not next(model.net1.parameters()).is_cuda
    assert next(model.net2.parameters()).is_cuda
