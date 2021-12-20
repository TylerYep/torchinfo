""" tests/torchinfo_test.py """
import pytest
import torch
import torchvision  # type: ignore[import]
from torch import nn

from tests.conftest import verify_output_str
from tests.fixtures.genotype import GenotypeNetwork  # type: ignore[attr-defined]
from tests.fixtures.models import (
    AutoEncoder,
    ContainerModule,
    CustomParameter,
    DictParameter,
    EmptyModule,
    LinearModel,
    LSTMNet,
    MixedTrainableParameters,
    ModuleDictModel,
    MultipleInputNetDifferentDtypes,
    NamedTuple,
    PackPaddedLSTM,
    ParameterListModel,
    PartialJITModel,
    RecursiveNet,
    ReturnDict,
    SiameseNets,
    SingleInputNet,
)
from tests.fixtures.tmva_net import TMVANet  # type: ignore[attr-defined]
from torchinfo import ALL_COLUMN_SETTINGS, summary


def test_basic_summary() -> None:
    model = SingleInputNet()

    summary(model)


def test_string_result() -> None:
    results = summary(SingleInputNet(), input_size=(16, 1, 28, 28))
    result_str = f"{results}\n"

    verify_output_str(result_str, "tests/test_output/string_result.out")


def test_single_input() -> None:
    model = SingleInputNet()

    # input_size keyword arg intentionally omitted.
    results = summary(model, (2, 1, 28, 28))

    assert len(results.summary_list) == 6, "Should find 6 layers"
    assert results.total_params == 21840
    assert results.trainable_params == 21840


def test_input_tensor() -> None:
    metrics = summary(SingleInputNet(), input_data=torch.randn(5, 1, 28, 28))

    assert metrics.input_size == torch.Size([5, 1, 28, 28])


def test_batch_size_optimization() -> None:
    model = SingleInputNet()

    # batch size intentionally omitted.
    results = summary(model, (1, 28, 28), batch_dim=0)

    assert len(results.summary_list) == 6, "Should find 6 layers"
    assert results.total_params == 21840
    assert results.trainable_params == 21840


def test_single_linear_layer() -> None:
    model = torch.nn.Linear(2, 5)

    results = summary(model)
    results = summary(model, input_size=(1, 2))

    assert results.total_params == 15
    assert results.trainable_params == 15


def test_single_layer_network_on_gpu() -> None:
    model = torch.nn.Linear(2, 5)
    if torch.cuda.is_available():
        model.cuda()

    results = summary(model, input_size=(1, 2))

    assert results.total_params == 15
    assert results.trainable_params == 15


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


def test_single_input_all_cols() -> None:
    model = SingleInputNet()
    col_names = ("kernel_size", "input_size", "output_size", "num_params", "mult_adds")
    input_shape = (7, 1, 28, 28)
    summary(
        model,
        input_data=torch.randn(*input_shape),
        depth=1,
        col_names=col_names,
        col_width=20,
    )


def test_single_input_batch_dim() -> None:
    model = SingleInputNet()
    col_names = ("kernel_size", "input_size", "output_size", "num_params", "mult_adds")
    summary(
        model,
        input_size=(1, 28, 28),
        depth=1,
        col_names=col_names,
        col_width=20,
        batch_dim=0,
    )


def test_frozen_layers() -> None:
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


def test_resnet18_depth_consistency() -> None:
    model = torchvision.models.resnet18()

    for depth in range(1, 3):
        summary(model, (1, 3, 64, 64), depth=depth, cache_forward_pass=True)


def test_resnet152() -> None:
    model = torchvision.models.resnet152()

    summary(model, (1, 3, 224, 224), depth=3)


def test_dict_input() -> None:
    # TODO: expand this test to handle intermediate dict layers.
    model = MultipleInputNetDifferentDtypes()
    input_data = torch.randn(1, 300)
    other_input_data = torch.randn(1, 300).long()

    summary(model, input_data={"x1": input_data, "x2": other_input_data})


def test_row_settings() -> None:
    model = SingleInputNet()

    summary(model, input_size=(16, 1, 28, 28), row_settings=("var_names",))


def test_jit() -> None:
    model = LinearModel()
    model_jit = torch.jit.script(model)
    x = torch.randn(64, 128)

    regular_model = summary(model, input_data=x)
    jit_model = summary(model_jit, input_data=x)

    assert len(regular_model.summary_list) == len(jit_model.summary_list)


def test_partial_jit() -> None:
    model_jit = torch.jit.script(PartialJITModel())

    summary(model_jit, input_data=torch.randn(2, 1, 28, 28))


def test_custom_parameter() -> None:
    model = CustomParameter(8, 4)

    summary(model, input_size=(1,))


def test_parameter_list() -> None:
    model = ParameterListModel()

    summary(
        model,
        input_size=(100, 100),
        verbose=2,
        col_names=ALL_COLUMN_SETTINGS,
        col_width=15,
    )


def test_dict_parameters_1() -> None:
    model = DictParameter()

    input_data = {256: torch.randn(10, 1), 512: [torch.randn(10, 1)]}
    summary(model, input_data={"x": input_data, "scale_factor": 5})


def test_dict_parameters_2() -> None:
    model = DictParameter()

    input_data = {256: torch.randn(10, 1), 512: [torch.randn(10, 1)]}
    summary(model, input_data={"x": input_data}, scale_factor=5)


def test_dict_parameters_3() -> None:
    model = DictParameter()

    input_data = {256: torch.randn(10, 1), 512: [torch.randn(10, 1)]}
    summary(model, input_data=[input_data], scale_factor=5)


def test_lstm() -> None:
    # results = summary(LSTMNet(), input_size=(100, 1), dtypes=[torch.long])
    results = summary(
        LSTMNet(),
        input_size=(1, 100),
        dtypes=[torch.long],
        verbose=2,
        col_width=20,
        col_names=("kernel_size", "output_size", "num_params", "mult_adds"),
        row_settings=("var_names",),
    )

    assert len(results.summary_list) == 4, "Should find 4 layers"


def test_lstm_custom_batch_size() -> None:
    # batch_size intentionally omitted.
    results = summary(LSTMNet(), (100,), dtypes=[torch.long], batch_dim=1)

    assert len(results.summary_list) == 4, "Should find 4 layers"


def test_recursive() -> None:
    results = summary(RecursiveNet(), input_size=(1, 64, 28, 28))
    second_layer = results.summary_list[2]

    assert len(results.summary_list) == 7, "Should find 7 layers"
    assert (
        second_layer.num_params_to_str(reached_max_depth=False) == "(recursive)"
    ), "should not count the second layer again"
    assert results.total_params == 36928
    assert results.trainable_params == 36928
    assert results.total_mult_adds == 173709312


def test_resnet() -> None:
    # According to https://arxiv.org/abs/1605.07146,
    # resnet50 has ~25.6 M trainable params.
    model = torchvision.models.resnet50()
    results = summary(model, input_size=(2, 3, 224, 224))

    assert results.total_params == 25557032  # close to 25.6e6
    assert results.total_mult_adds == sum(
        layer.macs for layer in results.summary_list if layer.is_leaf_layer
    )


def test_siamese_net() -> None:
    metrics = summary(SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)])

    assert round(metrics.float_to_megabytes(metrics.total_input), 2) == 0.25


def test_container() -> None:
    summary(ContainerModule(), input_size=(1, 5), depth=4)


def test_empty_module() -> None:
    summary(EmptyModule())


@pytest.mark.skip
def test_fasterrcnn() -> None:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained_backbone=False
    )
    results = summary(model, input_size=(1, 3, 112, 112))

    assert results.total_params == 41755286


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

    summary(PackPaddedLSTM(), input_data=x, lengths=y, device="cpu")


def test_module_dict() -> None:
    summary(
        ModuleDictModel(),
        input_data=torch.randn(1, 10, 3, 3),
        layer_type="conv",
        activation_type="lrelu",
    )


def test_model_with_args() -> None:
    summary(RecursiveNet(), input_size=(1, 64, 28, 28), args1="args1", args2="args2")


def test_input_size_possibilities() -> None:
    test = CustomParameter(2, 3)

    summary(test, input_size=[(2,)])
    summary(test, input_size=((2,),))
    summary(test, input_size=(2,))
    summary(test, input_size=[2])


def test_multiple_input_tensor_args() -> None:
    input_data = torch.randn(1, 300)
    other_input_data = torch.randn(1, 300).long()

    metrics = summary(
        MultipleInputNetDifferentDtypes(), input_data=input_data, x2=other_input_data
    )

    assert metrics.input_size == torch.Size([1, 300])


def test_multiple_input_tensor_dict() -> None:
    input_data = torch.randn(1, 300)
    other_input_data = torch.randn(1, 300).long()

    metrics = summary(
        MultipleInputNetDifferentDtypes(),
        input_data={"x1": input_data, "x2": other_input_data},
    )

    assert metrics.input_size == {
        "x1": torch.Size([1, 300]),
        "x2": torch.Size([1, 300]),
    }


def test_multiple_input_tensor_list() -> None:
    input_data = torch.randn(1, 300)
    other_input_data = torch.randn(1, 300).long()

    metrics = summary(
        MultipleInputNetDifferentDtypes(), input_data=[input_data, other_input_data]
    )

    assert metrics.input_size == [torch.Size([1, 300]), torch.Size([1, 300])]


def test_namedtuple() -> None:
    model = NamedTuple()
    input_size = [(2, 1, 28, 28), (2, 1, 28, 28)]
    named_tuple = model.Point(*input_size)
    summary(model, input_size=input_size, z=named_tuple, device="cpu")


def test_return_dict() -> None:
    input_size = [torch.Size([1, 28, 28]), [12]]

    metrics = summary(ReturnDict(), input_size=input_size, col_width=65, batch_dim=0)

    assert metrics.input_size == [(1, 28, 28), [12]]


def test_containers() -> None:
    summary(ContainerModule(), input_size=(5,))


def test_eval_order_doesnt_matter() -> None:
    input_size = (1, 3, 224, 224)
    input_tensor = torch.ones(input_size)

    model1 = torchvision.models.resnet18(pretrained=True)
    model1.eval()
    summary(model1, input_size=input_size, device="cpu")
    with torch.inference_mode():  # type: ignore[no-untyped-call]
        output1 = model1(input_tensor)

    model2 = torchvision.models.resnet18(pretrained=True)
    summary(model2, input_size=input_size, device="cpu")
    model2.eval()
    with torch.inference_mode():  # type: ignore[no-untyped-call]
        output2 = model2(input_tensor)

    assert torch.all(torch.eq(output1, output2))


def test_autoencoder() -> None:
    model = AutoEncoder()
    summary(model, input_size=(1, 3, 64, 64))


def test_genotype() -> None:
    model = GenotypeNetwork()

    x = summary(model, (2, 3, 32, 32), depth=3, cache_forward_pass=True)
    y = summary(model, (2, 3, 32, 32), depth=7, cache_forward_pass=True)

    assert x.total_params == y.total_params, (x, y)


def test_tmva_net_column_totals() -> None:
    for depth in (1, 3, 5):
        results = summary(
            TMVANet(n_classes=4, n_frames=5),
            input_data=[
                torch.randn(1, 1, 5, 256, 64),
                torch.randn(1, 1, 5, 256, 256),
                torch.randn(1, 1, 5, 256, 64),
            ],
            col_names=["output_size", "num_params", "mult_adds"],
            depth=depth,
            cache_forward_pass=True,
        )

        assert results.total_params == sum(
            layer.num_params for layer in results.summary_list if layer.is_leaf_layer
        )
        assert results.total_mult_adds == sum(
            layer.macs for layer in results.summary_list if layer.is_leaf_layer
        )


def test_reusing_activation_layers() -> None:
    act = nn.LeakyReLU(inplace=True)
    model1 = nn.Sequential(act, nn.Identity(), act, nn.Identity(), act)  # type: ignore[no-untyped-call] # noqa
    model2 = nn.Sequential(
        nn.LeakyReLU(inplace=True),
        nn.Identity(),  # type: ignore[no-untyped-call]
        nn.LeakyReLU(inplace=True),
        nn.Identity(),  # type: ignore[no-untyped-call]
        nn.LeakyReLU(inplace=True),
    )

    result_1 = summary(model1)
    result_2 = summary(model2)

    assert len(result_1.summary_list) == len(result_2.summary_list) == 6


def test_mixed_trainable_parameters() -> None:
    result = summary(MixedTrainableParameters())

    assert result.trainable_params == 10
    assert result.total_params == 20
