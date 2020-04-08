import pytest
import numpy as np
import torchvision
import torch

from torchsummary.torchsummary import summary
from fixtures.models import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes, \
    LSTMNet, RecursiveNet, NetWithArgs, CustomModule


class TestModels:
    @staticmethod
    def test_single_input():
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        results = summary(model, input_shape)

        assert results.total_params == 21840
        assert results.trainable_params == 21840

    @staticmethod
    def test_multiple_input():
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)

        results = summary(model, [input1, input2])

        assert results.total_params == 31120
        assert results.trainable_params == 31120

    @staticmethod
    def test_single_layer_network():
        model = torch.nn.Linear(2, 5)
        input_shape = (1, 2)

        results = summary(model, input_shape)

        assert results.total_params == 15
        assert results.trainable_params == 15

    @staticmethod
    def test_single_layer_network_on_gpu():
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
        input_shape = (1, 2)

        results = summary(model, input_shape)

        assert results.total_params == 15
        assert results.trainable_params == 15

    @staticmethod
    def test_multiple_input_types():
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]

        results = summary(model, [input1, input2], dtypes=dtypes)

        assert results.total_params == 31120
        assert results.trainable_params == 31120

    @staticmethod
    def test_lstm():
        results = summary(LSTMNet(), (100,), dtypes=[torch.long])

        assert len(results.summary_list) == 3, 'Should find 3 layers'

    @staticmethod
    def test_recursive():
        results = summary(RecursiveNet(), (64, 28, 28))
        second_layer = results.summary_list[1]

        assert len(results.summary_list) == 2, 'Should find 2 layers'
        assert second_layer.num_params_to_str() == '(recursive)', \
            'should not count the second layer again'
        assert results.total_params == 36928
        assert results.trainable_params == 36928
        assert results.total_mult_adds == 57802752

    @staticmethod
    def test_model_with_args():
        summary(NetWithArgs(), (64, 28, 28), "args1", args2="args2")

    @staticmethod
    def test_resnet():
        # According to https://arxiv.org/abs/1605.07146, resnet50 has ~25.6 M trainable params.
        # Let's make sure we count them correctly
        model = torchvision.models.resnet50()
        results = summary(model, (3, 224, 224))

        np.testing.assert_approx_equal(25.6e6, results.total_params, significant=3)

    @staticmethod
    def test_custom_modules():
        test = CustomModule(2, 3)

        summary(test, [(2,)])
        summary(test, (2,))
        summary(test, [2, ])
        with pytest.raises(AssertionError):
            summary(test, [(3, 0)])

    @staticmethod
    def test_input_tensor():
        input_data = torch.randn(5, 1, 28, 28)

        metrics = summary(SingleInputNet(), input_data)

        assert metrics.input_size == [torch.Size([5, 1, 28, 28])]

    @staticmethod
    def test_multiple_input_tensor():
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        metrics = summary(MultipleInputNetDifferentDtypes(), input_data, other_input_data)

        assert metrics.input_size == [torch.Size([1, 300])]


class TestOutputString:
    @staticmethod
    def test_single_input(capsys):
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(model, input_shape, max_depth=1)

        verify_output(capsys, 'unit_test/test_output/single_input.out')

    @staticmethod
    def test_single_input_with_kernel_macs(capsys):
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(model,
                input_shape,
                max_depth=1,
                col_names=['kernel_size', 'output_size', 'num_params', 'mult_adds'],
                col_width=20)

        verify_output(capsys, 'unit_test/test_output/single_input_all.out')

    @staticmethod
    def test_lstm_out(capsys):
        summary(LSTMNet(), (100,),
                dtypes=[torch.long],
                use_branching=False,
                verbose=2,
                col_width=20,
                col_names=['kernel_size', 'output_size', 'num_params', 'mult_adds'])

        verify_output(capsys, 'unit_test/test_output/lstm.out')

    @staticmethod
    def test_frozen_layers_out(capsys):
        model = torchvision.models.resnet18()
        input_shape = (3, 64, 64)
        for ind, param in enumerate(model.parameters()):
            if ind < 30:
                param.requires_grad = False

        summary(model, input_shape,
                max_depth=3,
                col_names=['output_size', 'num_params', 'kernel_size', 'mult_adds'])

        verify_output(capsys, 'unit_test/test_output/frozen_layers.out')

    @staticmethod
    def test_resnet_out(capsys):
        model = torchvision.models.resnet152()

        summary(model, (3, 224, 224), max_depth=3)

        verify_output(capsys, 'unit_test/test_output/resnet152.out')


def verify_output(capsys, filename):
    captured = capsys.readouterr().out
    with capsys.disabled():
        with open(filename) as output_file:
            expected = output_file.read()
    assert captured == expected
