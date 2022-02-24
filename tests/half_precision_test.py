import pytest
import torch
from pytest import approx

from tests.fixtures.models import LinearModel, LSTMNet, SingleInputNet
from torchinfo import summary
from torchinfo.model_statistics import ModelStatistics


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda must be available to test half models."
)
class TestHalfPrecision:
    """
    Iterate on these tests by using the following Google Colab link:
    https://colab.research.google.com/drive/1e_86DcAL6Q0r1OkjFcOuYKQRJaKsIro8
    """

    @staticmethod
    def test_single_input_half() -> None:
        model = SingleInputNet()
        model.half()

        input_data = torch.randn((2, 1, 28, 28), dtype=torch.float16, device="cuda")
        results = summary(model, input_data=input_data)

        assert ModelStatistics.to_megabytes(results.total_param_bytes) == 0.04368
        assert ModelStatistics.to_megabytes(results.total_output_bytes) == 0.0568

    @staticmethod
    def test_linear_model_half() -> None:
        x = torch.randn((64, 128))

        model = LinearModel()
        results = summary(model, input_data=x)

        model.half()
        x = x.type(torch.float16)
        results_half = summary(model, input_data=x)

        assert ModelStatistics.to_megabytes(results_half.total_param_bytes) == approx(
            ModelStatistics.to_megabytes(results.total_param_bytes) / 2
        )
        assert ModelStatistics.to_megabytes(results_half.total_output_bytes) == approx(
            ModelStatistics.to_megabytes(results.total_output_bytes) / 2
        )

    @staticmethod
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

        assert ModelStatistics.to_megabytes(results.total_param_bytes) == 7.56916
        assert ModelStatistics.to_megabytes(results.total_output_bytes) == 0.3328
