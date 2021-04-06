""" tests/torchinfo_test.py """
import torch
import torchvision

from fixtures.models import (
    AutoEncoder,
    ContainerModule,
    CustomModule,
    LSTMNet,
    MultipleInputNetDifferentDtypes,
    NamedTuple,
    PackPaddedLSTM,
    RecursiveNet,
    ReturnDict,
    SiameseNets,
    SingleInputNet,
)
from torchinfo import summary


class TestModels:
    """ Test torchinfo on many different models. """

    @staticmethod
    def test_single_input() -> None:
        model = SingleInputNet()

        # input_size keyword arg intentionally omitted.
        results = summary(model, (2, 1, 28, 28))

        assert len(results.summary_list) == 5, "Should find 6 layers"
        assert results.total_params == 21840
        assert results.trainable_params == 21840

    @staticmethod
    def test_input_tensor() -> None:
        metrics = summary(SingleInputNet(), input_data=torch.randn(5, 1, 28, 28))

        assert metrics.input_size == [torch.Size([5, 1, 28, 28])]

    @staticmethod
    def test_batch_size_optimization() -> None:
        model = SingleInputNet()

        # batch size intentionally omitted.
        results = summary(model, (1, 28, 28), batch_dim=0)

        assert len(results.summary_list) == 5, "Should find 6 layers"
        assert results.total_params == 21840
        assert results.trainable_params == 21840

    @staticmethod
    def test_single_layer_network() -> None:
        model = torch.nn.Linear(2, 5)

        results = summary(model, input_size=(1, 2))

        assert results.total_params == 15
        assert results.trainable_params == 15

    @staticmethod
    def test_single_layer_network_on_gpu() -> None:
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()

        results = summary(model, input_size=(1, 2))

        assert results.total_params == 15
        assert results.trainable_params == 15

    @staticmethod
    def test_multiple_input_types() -> None:
        model = MultipleInputNetDifferentDtypes()
        input_size = (1, 300)
        if torch.cuda.is_available():
            dtypes = [
                torch.cuda.FloatTensor,  # type: ignore[attr-defined]
                torch.cuda.LongTensor,  # type: ignore[attr-defined]
            ]
        else:
            dtypes = [torch.FloatTensor, torch.LongTensor]

        results = summary(model, input_size=[input_size, input_size], dtypes=dtypes)

        assert results.total_params == 31120
        assert results.trainable_params == 31120

    @staticmethod
    def test_lstm() -> None:
        results = summary(LSTMNet(), input_size=(100, 1), dtypes=[torch.long])

        assert len(results.summary_list) == 3, "Should find 3 layers"

    @staticmethod
    def test_lstm_custom_batch_size() -> None:
        # batch_size intentionally omitted.
        results = summary(LSTMNet(), (100,), dtypes=[torch.long], batch_dim=1)

        assert len(results.summary_list) == 3, "Should find 3 layers"

    @staticmethod
    def test_recursive() -> None:
        results = summary(RecursiveNet(), input_size=(1, 64, 28, 28))
        second_layer = results.summary_list[1]

        assert len(results.summary_list) == 6, "Should find 6 layers"
        assert (
            second_layer.num_params_to_str() == "(recursive)"
        ), "should not count the second layer again"
        assert results.total_params == 36928
        assert results.trainable_params == 36928
        assert results.total_mult_adds == 173408256

    @staticmethod
    def test_resnet() -> None:
        # According to https://arxiv.org/abs/1605.07146,
        # resnet50 has ~25.6 M trainable params.
        model = torchvision.models.resnet50()
        results = summary(model, input_size=(2, 3, 224, 224))

        assert results.total_params == 25557032  # close to 25.6e6

    @staticmethod
    def test_siamese_net() -> None:
        metrics = summary(SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)])

        assert round(metrics.to_bytes(metrics.total_input), 2) == 0.06

    @staticmethod
    def test_fasterrcnn() -> None:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained_backbone=False
        )
        results = summary(model, input_size=(1, 3, 112, 112))

        assert results.total_params == 38410902

    @staticmethod
    def test_device() -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SingleInputNet()
        # input_size
        summary(model, input_size=(5, 1, 28, 28), device=device)

        # input_data
        input_data = torch.randn(5, 1, 28, 28)
        summary(model, input_data=input_data)
        summary(model, input_data=input_data, device=device)
        summary(model, input_data=input_data.to(device))
        summary(model, input_data=input_data.to(device), device=torch.device("cpu"))

    @staticmethod
    def test_pack_padded() -> None:
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

        summary(PackPaddedLSTM(), input_data=x, lengths=y, device='cpu')


class TestEdgeCaseModels:
    """ Test torchinfo on different edge case models. """

    @staticmethod
    def test_model_with_args() -> None:
        summary(
            RecursiveNet(), input_size=(1, 64, 28, 28), args1="args1", args2="args2"
        )

    @staticmethod
    def test_input_size_possibilities() -> None:
        test = CustomModule(2, 3)

        summary(test, input_size=[(2,)])
        summary(test, input_size=((2,),))
        summary(test, input_size=(2,))
        summary(test, input_size=[2])

    @staticmethod
    def test_multiple_input_tensor_args() -> None:
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        metrics = summary(
            MultipleInputNetDifferentDtypes(),
            input_data=input_data,
            x2=other_input_data,
        )

        assert metrics.input_size == [torch.Size([1, 300])]

    @staticmethod
    def test_multiple_input_tensor_dict() -> None:
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        metrics = summary(
            MultipleInputNetDifferentDtypes(),
            input_data={"x1": input_data, "x2": other_input_data},
        )

        assert metrics.input_size == [torch.Size([1, 300]), torch.Size([1, 300])]

    @staticmethod
    def test_multiple_input_tensor_list() -> None:
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        metrics = summary(
            MultipleInputNetDifferentDtypes(), input_data=[input_data, other_input_data]
        )

        assert metrics.input_size == [torch.Size([1, 300]), torch.Size([1, 300])]

    @staticmethod
    def test_namedtuple() -> None:
        model = NamedTuple()
        input_size = [(2, 1, 28, 28), (2, 1, 28, 28)]
        named_tuple = model.Point(*input_size)
        summary(model, input_size=input_size, z=named_tuple, device='cpu')

    @staticmethod
    def test_return_dict() -> None:
        input_size = [torch.Size([1, 28, 28]), [12]]

        metrics = summary(
            ReturnDict(), input_size=input_size, col_width=65, batch_dim=0
        )

        assert metrics.input_size == [(1, 28, 28), [12]]

    @staticmethod
    def test_containers() -> None:
        summary(ContainerModule(), input_size=(5,))

    @staticmethod
    def test_eval_order_doesnt_matter() -> None:
        input_size = (1, 3, 224, 224)
        input_tensor = torch.ones(input_size)

        model1 = torchvision.models.resnet18(pretrained=True)
        model1.eval()
        summary(model1, input_size=input_size, device='cpu')
        with torch.no_grad():
            output1 = model1(input_tensor)

        model2 = torchvision.models.resnet18(pretrained=True)
        summary(model2, input_size=input_size, device='cpu')
        model2.eval()
        with torch.no_grad():
            output2 = model2(input_tensor)

        assert torch.all(torch.eq(output1, output2))

    @staticmethod
    def test_autoencoder() -> None:
        model = AutoEncoder()
        summary(model, input_size=(1, 3, 64, 64))
