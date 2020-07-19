""" unit_test/torchsummary_test.py """
# pylint: disable=no-self-use

import numpy as np
import pytest
import torch
import torchvision

from fixtures.models import (
    ContainerModule,
    CustomModule,
    EdgeCaseModel,
    FunctionalNet,
    LSTMNet,
    MultipleInputNetDifferentDtypes,
    PackPaddedLSTM,
    RecursiveNet,
    ReturnDict,
    SiameseNets,
    SingleInputNet,
)
from torchsummary.torchsummary import summary


class TestModels:
    """ Test torchsummary on many different models. """

    def test_string_result(self):
        results = summary(SingleInputNet(), (1, 28, 28), verbose=0)

        result_str = str(results) + "\n"

        with open("unit_test/test_output/single_input.out", encoding="utf-8") as output_file:
            expected = output_file.read()
        assert result_str == expected

    def test_single_input(self):
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        results = summary(model, input_shape)

        assert results.total_params == 21840
        assert results.trainable_params == 21840

    def test_input_tensor(self):
        input_data = torch.randn(5, 1, 28, 28)

        metrics = summary(SingleInputNet(), input_data)

        assert metrics.input_size == [torch.Size([5, 1, 28, 28])]

    def test_single_layer_network(self):
        model = torch.nn.Linear(2, 5)
        input_shape = (1, 2)

        results = summary(model, input_shape)

        assert results.total_params == 15
        assert results.trainable_params == 15

    def test_single_layer_network_on_gpu(self):
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
        input_shape = (1, 2)

        results = summary(model, input_shape)

        assert results.total_params == 15
        assert results.trainable_params == 15

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        if torch.cuda.is_available():
            dtypes = [torch.cuda.FloatTensor, torch.cuda.LongTensor]
        else:
            dtypes = [torch.FloatTensor, torch.LongTensor]

        results = summary(model, [input1, input2], dtypes=dtypes)

        assert results.total_params == 31120
        assert results.trainable_params == 31120

    def test_lstm(self):
        results = summary(LSTMNet(), (100,), dtypes=[torch.long])

        assert len(results.summary_list) == 3, "Should find 3 layers"

    def test_recursive(self):
        results = summary(RecursiveNet(), (64, 28, 28))
        second_layer = results.summary_list[1]

        assert len(results.summary_list) == 6, "Should find 6 layers"
        assert (
            second_layer.num_params_to_str() == "(recursive)"
        ), "should not count the second layer again"
        assert results.total_params == 36928
        assert results.trainable_params == 36928
        assert results.total_mult_adds == 173408256

    def test_model_with_args(self):
        summary(RecursiveNet(), (64, 28, 28), "args1", args2="args2")

    def test_resnet(self):
        # According to https://arxiv.org/abs/1605.07146, resnet50 has ~25.6 M trainable params.
        # Let's make sure we count them correctly
        model = torchvision.models.resnet50()
        results = summary(model, (3, 224, 224))

        np.testing.assert_approx_equal(25.6e6, results.total_params, significant=3)

    def test_input_size_possibilities(self):
        test = CustomModule(2, 3)

        summary(test, [(2,)])
        summary(test, ((2,),))
        summary(test, (2,))
        summary(test, [2])
        with pytest.raises(AssertionError):
            summary(test, [(3, 0)])
        with pytest.raises(TypeError):
            summary(test, {0: 1})
        with pytest.raises(TypeError):
            summary(test, "hello")

    def test_multiple_input_tensor_args(self):
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        metrics = summary(MultipleInputNetDifferentDtypes(), input_data, other_input_data)

        assert metrics.input_size == [torch.Size([1, 300])]

    def test_multiple_input_tensor_list(self):
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        metrics = summary(MultipleInputNetDifferentDtypes(), [input_data, other_input_data])

        assert metrics.input_size == [torch.Size([1, 300]), torch.Size([1, 300])]

    def test_siamese_net(self):
        metrics = summary(SiameseNets(), [(1, 88, 88), (1, 88, 88)])

        assert round(metrics.to_bytes(metrics.total_input), 2) == 0.06

    def test_functional_layers(self):
        summary(FunctionalNet(), (1, 28, 28))
        # Should assert that Maxpool functional layer is detected!
        # We don't handle functional layers yet.

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(SingleInputNet(), (1, 28, 28), device=device)

        input_data = torch.randn(5, 1, 28, 28)
        summary(SingleInputNet(), input_data)
        summary(SingleInputNet(), input_data, device=device)

        summary(SingleInputNet(), input_data.to(device))
        summary(SingleInputNet(), input_data.to(device), device=torch.device("cpu"))

    def test_return_dict(self):
        input_size = [torch.Size([1, 28, 28]), [12]]

        metrics = summary(ReturnDict(), input_size, col_width=65)

        assert metrics.input_size == [(1, 28, 28), [12]]

    def test_exceptions(self):
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), (1, 28, 28))
        with pytest.raises(TypeError):
            summary(EdgeCaseModel(return_str=True), (1, 28, 28))

    def test_pack_padded(self):
        x = torch.ones([20, 128]).long()
        # fmt: off
        y = torch.Tensor([
            13, 12, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ]).long()
        # fmt: on

        summary(PackPaddedLSTM(), x, y)

    def test_containers(self):
        summary(ContainerModule(), (5,))
