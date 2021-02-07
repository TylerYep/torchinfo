""" unit_test/output_test.py """
import sys
import warnings

import pytest
import torch
import torchvision

from conftest import verify_output
from fixtures.models import (
    ContainerModule,
    EdgeCaseModel,
    EmptyModule,
    LSTMNet,
    MultipleInputNetDifferentDtypes,
    SingleInputNet,
)
from torchinfo import summary


class TestOutputString:
    """ Tests for output string. """

    @staticmethod
    def test_string_result() -> None:
        results = summary(SingleInputNet(), input_size=(16, 1, 28, 28), verbose=0)
        result_str = str(results) + "\n"

        with open(
            "unit_test/test_output/single_input.out", encoding="utf-8"
        ) as output_file:
            expected = output_file.read()
        assert result_str == expected

    @staticmethod
    def test_single_input(capsys: pytest.CaptureFixture[str]) -> None:
        model = SingleInputNet()

        summary(model, input_size=(16, 1, 28, 28), depth=1)

        verify_output(capsys, "unit_test/test_output/single_input.out")

    @staticmethod
    def test_single_input_all_cols(capsys: pytest.CaptureFixture[str]) -> None:
        model = SingleInputNet()
        col_names = (
            "kernel_size",
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
        )
        input_shape = (7, 1, 28, 28)
        summary(
            model, input_size=input_shape, depth=1, col_names=col_names, col_width=20
        )
        verify_output(capsys, "unit_test/test_output/single_input_all.out")

        summary(
            model,
            input_data=torch.randn(*input_shape),
            depth=1,
            col_names=col_names,
            col_width=20,
        )
        verify_output(capsys, "unit_test/test_output/single_input_all.out")

        summary(
            model,
            input_size=(1, 28, 28),
            depth=1,
            col_names=col_names,
            col_width=20,
            batch_dim=0,
        )
        verify_output(capsys, "unit_test/test_output/single_input_batch_dim.out")

    @staticmethod
    def test_basic_summary(capsys: pytest.CaptureFixture[str]) -> None:
        model = SingleInputNet()

        summary(model)

        verify_output(capsys, "unit_test/test_output/basic_summary.out")

    @staticmethod
    def test_lstm_out(capsys: pytest.CaptureFixture[str]) -> None:
        summary(
            LSTMNet(),
            input_size=(1, 100),
            dtypes=[torch.long],
            verbose=2,
            col_width=20,
            col_names=("kernel_size", "output_size", "num_params", "mult_adds"),
        )

        if sys.version_info < (3, 7):
            try:
                verify_output(capsys, "unit_test/test_output/lstm.out")
            except AssertionError:
                warnings.warn(
                    "LSTM verbose output is not determininstic because dictionaries "
                    "are not necessarily ordered in versions before Python 3.7."
                )
        else:
            verify_output(capsys, "unit_test/test_output/lstm.out")

    @staticmethod
    def test_frozen_layers_out(capsys: pytest.CaptureFixture[str]) -> None:
        model = torchvision.models.resnet18()
        for ind, param in enumerate(model.parameters()):
            if ind < 30:
                param.requires_grad = False

        summary(
            model,
            input_size=(1, 3, 64, 64),
            depth=3,
            col_names=("output_size", "num_params", "kernel_size", "mult_adds"),
        )

        verify_output(capsys, "unit_test/test_output/frozen_layers.out")

    @staticmethod
    def test_resnet_out(capsys: pytest.CaptureFixture[str]) -> None:
        model = torchvision.models.resnet152()

        summary(model, (1, 3, 224, 224), depth=3)

        verify_output(capsys, "unit_test/test_output/resnet152.out")

    @staticmethod
    def test_dict_out(capsys: pytest.CaptureFixture[str]) -> None:
        model = MultipleInputNetDifferentDtypes()
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        summary(
            model, input_data={"x1": input_data, "x2": other_input_data},
        )

        verify_output(capsys, "unit_test/test_output/dict_input.out")


class TestEdgeCaseOutputString:
    """ Tests for edge case output strings. """

    @staticmethod
    def test_exception_output(capsys: pytest.CaptureFixture[str]) -> None:
        input_size = (1, 1, 28, 28)
        summary(EdgeCaseModel(throw_error=False), input_size=input_size)
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), input_size=input_size)

        verify_output(capsys, "unit_test/test_output/exception.out")

    @staticmethod
    def test_container_output(capsys: pytest.CaptureFixture[str]) -> None:
        summary(ContainerModule(), input_size=(1, 5), depth=4)

        verify_output(capsys, "unit_test/test_output/container.out")

    @staticmethod
    def test_empty_module(capsys: pytest.CaptureFixture[str]) -> None:
        summary(EmptyModule())

        verify_output(capsys, "unit_test/test_output/empty_module.out")
