from typing import Any

import torch
from torch import nn
from torch.nn.utils import prune

from tests.conftest import verify_output_str
from tests.fixtures.models import (
    AutoEncoder,
    ContainerModule,
    CustomParameter,
    DictParameter,
    EmptyModule,
    FakePrunedLayerModel,
    LinearModel,
    LSTMNet,
    MixedTrainableParameters,
    ModuleDictModel,
    MultipleInputNetDifferentDtypes,
    NamedTuple,
    PackPaddedLSTM,
    ParameterListModel,
    PartialJITModel,
    PrunedLayerNameModel,
    RecursiveNet,
    ReturnDict,
    ReuseLinear,
    ReuseLinearExtended,
    ReuseReLU,
    SiameseNets,
    SingleInputNet,
)
from torchinfo import ColumnSettings, summary
from torchinfo.enums import Verbosity


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


def test_pruning() -> None:
    model = SingleInputNet()
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(  # type: ignore[no-untyped-call]
                module, "weight", 0.5
            )
    results = summary(model, input_size=(16, 1, 28, 28))

    assert results.total_params == 10965
    assert results.total_mult_adds == 3957600


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
        col_names=list(ColumnSettings),
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
        verbose=Verbosity.VERBOSE,
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


def test_siamese_net() -> None:
    metrics = summary(SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)])

    assert round(metrics.float_to_megabytes(metrics.total_input), 2) == 0.25


def test_container() -> None:
    summary(ContainerModule(), input_size=(1, 5), depth=4)


def test_empty_module() -> None:
    summary(EmptyModule())


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

    summary(PackPaddedLSTM(), input_data=x, lengths=y)


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
    summary(model, input_size=input_size, z=named_tuple)


def test_return_dict() -> None:
    input_size = [torch.Size([1, 28, 28]), [12]]

    metrics = summary(ReturnDict(), input_size=input_size, col_width=65, batch_dim=0)

    assert metrics.input_size == [(1, 28, 28), [12]]


def test_containers() -> None:
    summary(ContainerModule(), input_size=(5,))


def test_autoencoder() -> None:
    model = AutoEncoder()
    summary(model, input_size=(1, 3, 64, 64))


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


def test_too_many_linear() -> None:
    net = ReuseLinear()
    summary(net, (2, 10))


def test_too_many_linear_plus_existing_hooks() -> None:
    a, b = False, False

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        del module, inputs
        nonlocal a
        a = True

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        del module, inputs, outputs
        nonlocal b
        b = True

    net = ReuseLinearExtended()
    result_1 = summary(net, (2, 10))

    net = ReuseLinearExtended()
    net.linear.register_forward_pre_hook(pre_hook)
    net.linear.register_forward_hook(hook)

    result_2 = summary(net, (2, 10))

    assert a is True
    assert b is True
    assert str(result_1) == str(result_2)


def test_too_many_relus() -> None:
    summary(ReuseReLU(), (4, 4, 64, 64))


def test_pruned_adversary() -> None:
    model = PrunedLayerNameModel(8, 4)

    results = summary(model, input_size=(1,))

    assert results.total_params == 32

    second_model = FakePrunedLayerModel(8, 4)

    results = summary(second_model, input_size=(1,))

    assert results.total_params == 32  # should be 64
