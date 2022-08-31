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
