import torch
import torchvision

from fixtures.models import LSTMNet, SingleInputNet
from torchsummary.torchsummary import summary


class TestOutputString:
    @staticmethod
    def test_single_input(capsys):
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(model, input_shape, depth=1)

        verify_output(capsys, "unit_test/test_output/single_input.out")

    @staticmethod
    def test_single_input_with_kernel_macs(capsys):
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(
            model,
            input_shape,
            depth=1,
            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
            col_width=20,
        )

        verify_output(capsys, "unit_test/test_output/single_input_all.out")

    @staticmethod
    def test_lstm_out(capsys):
        summary(
            LSTMNet(),
            (100,),
            dtypes=[torch.long],
            branching=False,
            verbose=2,
            col_width=20,
            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
        )

        verify_output(capsys, "unit_test/test_output/lstm.out")

    @staticmethod
    def test_frozen_layers_out(capsys):
        model = torchvision.models.resnet18()
        input_shape = (3, 64, 64)
        for ind, param in enumerate(model.parameters()):
            if ind < 30:
                param.requires_grad = False

        summary(
            model,
            input_shape,
            depth=3,
            col_names=["output_size", "num_params", "kernel_size", "mult_adds"],
        )

        verify_output(capsys, "unit_test/test_output/frozen_layers.out")

    @staticmethod
    def test_resnet_out(capsys):
        model = torchvision.models.resnet152()

        summary(model, (3, 224, 224), depth=3)

        verify_output(capsys, "unit_test/test_output/resnet152.out")


def verify_output(capsys, filename):
    captured = capsys.readouterr().out
    with capsys.disabled():
        with open(filename) as output_file:
            expected = output_file.read()
    assert captured == expected
