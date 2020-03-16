import torch

from src.torchsummary import summary, summary_string
from fixtures.models import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes

gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"

class TestModels:
    @staticmethod
    def test_single_input():
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        total_params, trainable_params = summary(model, input_shape)

        assert total_params == 21840
        assert trainable_params == 21840

    @staticmethod
    def test_multiple_input():
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)

        total_params, trainable_params = summary(model, [input1, input2])

        assert total_params == 31120
        assert trainable_params == 31120

    @staticmethod
    def test_single_layer_network():
        model = torch.nn.Linear(2, 5)
        input_shape = (1, 2)

        total_params, trainable_params = summary(model, input_shape)

        assert total_params == 15
        assert trainable_params == 15

    @staticmethod
    def test_single_layer_network_on_gpu():
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
        input_shape = (1, 2)

        total_params, trainable_params = summary(model, input_shape)

        assert total_params == 15
        assert trainable_params == 15

    @staticmethod
    def test_multiple_input_types():
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]

        total_params, trainable_params = summary(model, [input1, input2], dtypes=dtypes)

        assert total_params == 31120
        assert trainable_params == 31120


class TestOutputString:
    @staticmethod
    def test_single_input():
        model = SingleInputNet()
        input_shape = (1, 28, 28)

        result, _ = summary_string(model, input_shape)

        assert type(result) == str
