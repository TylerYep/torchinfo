""" unit_test/output_test.py """
# pylint: disable=no-self-use

import sys
import warnings

import pytest
import torch
import torchvision
from _pytest.capture import CaptureFixture

from conftest import verify_output
from fixtures.models import (
    ContainerModule,
    EdgeCaseModel,
    EmptyModule,
    LSTMNet,
    SingleInputNet,
)
from torchsummary.torchsummary import summary


class TestOutputString:
    """ Tests for output string. """

    def test_string_result(self) -> None:
        results = summary(SingleInputNet(), (1, 28, 28), verbose=0)

        result_str = str(results) + "\n"

        with open(
            "unit_test/test_output/single_input.out", encoding="utf-8"
        ) as output_file:
            expected = output_file.read()
        assert result_str == expected

    def test_single_input(self, capsys: CaptureFixture) -> None:
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(model, input_shape, depth=1)

        verify_output(capsys, "unit_test/test_output/single_input.out")

    def test_single_input_batch_dim(self, capsys: CaptureFixture) -> None:
        model = SingleInputNet()
        input_shape = (7, 1, 28, 28)
        summary(model, input_shape, depth=1, batch_dim=None)
        verify_output(capsys, "unit_test/test_output/single_input_batch_dim.out")

        input_data = torch.randn(*input_shape)
        summary(model, input_data, depth=1, batch_dim=None)
        verify_output(capsys, "unit_test/test_output/single_input_batch_dim.out")

    def test_basic_summary(self, capsys: CaptureFixture) -> None:
        model = SingleInputNet()

        summary(model)

        verify_output(capsys, "unit_test/test_output/basic_summary.out")

    def test_single_input_with_kernel_macs(self, capsys: CaptureFixture) -> None:
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(
            model,
            input_shape,
            depth=1,
            col_names=("kernel_size", "output_size", "num_params", "mult_adds"),
            col_width=20,
        )

        verify_output(capsys, "unit_test/test_output/single_input_all.out")

    def test_lstm_out(self, capsys: CaptureFixture) -> None:
        summary(
            LSTMNet(),
            (100,),
            dtypes=[torch.long],
            branching=False,
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

    def test_frozen_layers_out(self, capsys: CaptureFixture) -> None:
        model = torchvision.models.resnet18()
        input_shape = (3, 64, 64)
        for ind, param in enumerate(model.parameters()):
            if ind < 30:
                param.requires_grad = False

        summary(
            model,
            input_shape,
            depth=3,
            col_names=("output_size", "num_params", "kernel_size", "mult_adds"),
        )

        verify_output(capsys, "unit_test/test_output/frozen_layers.out")

    def test_resnet_out(self, capsys: CaptureFixture) -> None:
        model = torchvision.models.resnet152()

        summary(model, (3, 224, 224), depth=3)

        verify_output(capsys, "unit_test/test_output/resnet152.out")

    def test_exception_output(self, capsys: CaptureFixture) -> None:
        summary(EdgeCaseModel(throw_error=False), (1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), (1, 28, 28))

        verify_output(capsys, "unit_test/test_output/exception.out")

    def test_container_output(self, capsys: CaptureFixture) -> None:
        summary(ContainerModule(), (5,), depth=4)

        verify_output(capsys, "unit_test/test_output/container.out")

    def test_empty_module(self, capsys: CaptureFixture) -> None:
        summary(EmptyModule())

        verify_output(capsys, "unit_test/test_output/empty_module.out")
