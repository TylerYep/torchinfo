import pytest
import torch

from tests.fixtures.models import LinearModel, LSTMNet, SingleInputNet
from torchinfo import summary
from torchinfo.model_statistics import ModelStatistics

if not torch.cuda.is_available():
    pytest.skip("Cuda is not available to test half models", allow_module_level=True)


def test_single_input_half() -> None:
    model = SingleInputNet()
    model.half()

    input_data = torch.randn((2, 1, 28, 28), dtype=torch.float16, device="cuda")
    results = summary(model, input_data=input_data)

    assert ModelStatistics.to_megabytes(results.total_param_bytes) - (0.11 / 2) < 0.01
    assert ModelStatistics.to_megabytes(results.total_output_bytes), (0.09 / 2) < 0.01


def test_linear_model_half() -> None:
    x = torch.randn((64, 128))

    model = LinearModel()
    results = summary(model, input_data=x)

    model.half()
    x = x.type(torch.float16)
    results_half = summary(model, input_data=x)

    assert (
        ModelStatistics.to_megabytes(results_half.total_param_bytes)
        - ModelStatistics.to_megabytes(results.total_param_bytes) / 2
        < 0.01
    )
    assert (
        ModelStatistics.to_megabytes(results_half.total_output_bytes)
        - ModelStatistics.to_megabytes(results.total_output_bytes) / 2
        < 0.01
    )


def test_lstm_half() -> None:
    model = LSTMNet()
    model.half()
    results = summary(
        model,
        input_size=(1, 100),
        dtypes=[torch.long],
        col_width=20,
        col_names=("kernel_size", "output_size", "num_params", "mult_adds"),
        row_settings=("var_names",),
    )

    assert ModelStatistics.to_megabytes(results.total_param_bytes) - (15.14 / 2) < 0.01
    assert ModelStatistics.to_megabytes(results.total_output_bytes) - (0.67 / 2) < 0.01
