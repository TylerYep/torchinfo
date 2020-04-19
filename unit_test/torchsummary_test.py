import numpy as np
import pytest
import torch
import torchvision

from fixtures.models import (
    CustomModule,
    FunctionalNet,
    LSTMNet,
    MultipleInputNet,
    MultipleInputNetDifferentDtypes,
    NetWithArgs,
    RecursiveNet,
    SiameseNets,
    SingleInputNet,
)
from torchsummary.torchsummary import summary


class TestModels:
    @staticmethod
    def test_string_result():
        results = summary(SingleInputNet(), (1, 28, 28), verbose=0)

        result_str = str(results) + "\n"

        with open("unit_test/test_output/single_input.out") as output_file:
            expected = output_file.read()
        assert result_str == expected

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

        assert len(results.summary_list) == 3, "Should find 3 layers"

    @staticmethod
    def test_recursive():
        results = summary(RecursiveNet(), (64, 28, 28))
        second_layer = results.summary_list[1]

        assert len(results.summary_list) == 6, "Should find 6 layers"
        assert (
            second_layer.num_params_to_str() == "(recursive)"
        ), "should not count the second layer again"
        assert results.total_params == 36928
        assert results.trainable_params == 36928
        assert results.total_mult_adds == 173408256

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
        summary(test, [2])
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

    @staticmethod
    def test_siamese_net():
        metrics = summary(SiameseNets(), [(1, 88, 88), (1, 88, 88)])

        assert round(metrics.to_megabytes(metrics.total_input), 2) == 0.06

    @staticmethod
    def test_functional_layers():
        summary(FunctionalNet(), (1, 28, 28))

        # todo assert that Maxpool functional layer is detected!

    @staticmethod
    def test_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(SingleInputNet(), (1, 28, 28), device=device)

        input_data = torch.randn(5, 1, 28, 28)
        summary(SingleInputNet(), input_data)
        summary(SingleInputNet(), input_data, device=device)

        summary(SingleInputNet(), input_data.to(device))
        summary(SingleInputNet(), input_data.to(device), device=torch.device("cpu"))
