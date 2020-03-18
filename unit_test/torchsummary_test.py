import pytest
import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from torchsummary.torchsummary import summary
from fixtures.models import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes, \
    LSTMNet, RecursiveNet, NetWithArgs, CustomModule


class TestModels:
    @staticmethod
    def test_single_input():
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        _, (total_params, trainable_params) = summary(model, input_shape)

        assert total_params == 21840
        assert trainable_params == 21840

    @staticmethod
    def test_multiple_input():
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)

        _, (total_params, trainable_params) = summary(model, [input1, input2])

        assert total_params == 31120
        assert trainable_params == 31120

    @staticmethod
    def test_single_layer_network():
        model = torch.nn.Linear(2, 5)
        input_shape = (1, 2)

        _, (total_params, trainable_params) = summary(model, input_shape)

        assert total_params == 15
        assert trainable_params == 15

    @staticmethod
    def test_single_layer_network_on_gpu():
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
        input_shape = (1, 2)

        _, (total_params, trainable_params) = summary(model, input_shape)

        assert total_params == 15
        assert trainable_params == 15

    @staticmethod
    def test_multiple_input_types():
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]

        _, (total_params, trainable_params) = summary(model, [input1, input2], dtypes=dtypes)

        assert total_params == 31120
        assert trainable_params == 31120

    @staticmethod
    def test_lstm():
        summary_dict, _ = summary(LSTMNet(), (100,), dtypes=[torch.long]) # [length, batch_size]
        assert len(summary_dict) == 3, 'Should find 3 layers'

    @staticmethod
    def test_recursive():
        summary_dict, (total_params, trainable_params) = summary(RecursiveNet(), (64, 28, 28))
        second_layer = tuple(summary_dict.items())[1]

        assert len(summary_dict) == 2, 'Should find 2 layers'
        assert second_layer[1].num_params == '(recursive)', 'should not count the second layer again'
        assert total_params == 36928
        # assert df_total['Totals']['Mult-Adds'] == 57802752

    @staticmethod
    def test_model_with_args():
        summary(NetWithArgs(), (64, 28, 28), "args1", args2="args2")

    @staticmethod
    def test_resnet():
        model = torchvision.models.resnet50()
        _, (total_params, trainable_params) = summary(model, (3, 224, 224))
        # According to https://arxiv.org/abs/1605.07146, resnet50 has ~25.6 M trainable params.
        # Let's make sure we count them correctly

        np.testing.assert_approx_equal(25.6e6, total_params, significant=3)

    @staticmethod
    def test_custom_modules():
        test = CustomModule(2, 3)

        summary(test, [(2,)])
        summary(test, (2,))
        summary(test, [2,])
        with pytest.raises(AssertionError):
            summary(test, [(3, 0)])


class TestOutputString:
    @staticmethod
    def test_single_input(capsys):
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        summary(model, input_shape, max_depth=1)

        captured = capsys.readouterr().out
        with capsys.disabled():
            with open('unit_test/test_output/single_input.out') as output_file:
                expected = output_file.read()
        assert captured == expected

    @staticmethod
    def test_resnet_out(capsys):
        model = torchvision.models.resnet152()
        summary(model, (3, 224, 224), max_depth=3)

        captured = capsys.readouterr().out
        with capsys.disabled():
            with open('unit_test/test_output/resnet152.out') as output_file:
                expected = output_file.read()
        assert captured == expected

    # @staticmethod
    # def test_lstm_out(capsys):
    #     summary_dict, _ = summary(LSTMNet(), (100,), dtypes=[torch.long]) # [length, batch_size]

    #     captured = capsys.readouterr().out
    #     with capsys.disabled():
    #         with open('unit_test/test_output/lstm.out') as output_file:
    #             expected = output_file.read()
    #     assert captured == expected
