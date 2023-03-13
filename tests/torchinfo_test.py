from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from tests.conftest import verify_output_str
from tests.fixtures.models import (
    AutoEncoder,
    CNNModuleList,
    ContainerModule,
    ConvLayerA,
    ConvLayerB,
    CustomParameter,
    DictParameter,
    EdgecaseInputOutputModel,
    EmptyModule,
    FakePrunedLayerModel,
    HighlyNestedDictModel,
    InsideModel,
    LinearModel,
    LSTMNet,
    MixedTrainable,
    MixedTrainableParameters,
    ModuleDictModel,
    MultipleInputNetDifferentDtypes,
    NamedTuple,
    NumpyModel,
    PackPaddedLSTM,
    ParameterFCNet,
    ParameterListModel,
    PartialJITModel,
    PrunedLayerNameModel,
    RecursiveNet,
    RecursiveWithMissingLayers,
    RegisterParameter,
    ReturnDict,
    ReuseLinear,
    ReuseLinearExtended,
    ReuseReLU,
    SiameseNets,
    SimpleRNN,
    SingleInputNet,
    UninitializedParameterModel,
)
from torchinfo import ColumnSettings, summary
from torchinfo.enums import Units, Verbosity


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


def test_training_mode() -> None:
    summary(torch.nn.Linear(2, 5), input_data=torch.zeros(1, 2), mode="train")


def test_uninitialized_tensor() -> None:
    model = UninitializedParameterModel()
    summary(model)
    summary(model, input_data=torch.randn(2, 2))


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
    input_shape = (7, 1, 28, 28)
    summary(
        model,
        input_data=torch.randn(*input_shape),
        depth=1,
        col_names=list(ColumnSettings),
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


def test_formatting_options() -> None:
    model = SingleInputNet()

    results = summary(model, input_size=(16, 1, 28, 28), verbose=0)
    results.formatting.macs_units = Units.NONE
    print(results)

    results.formatting.params_units = Units.TERABYTES
    results.formatting.macs_units = Units.TERABYTES
    print(results)


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
        col_width=20,
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
    summary(ContainerModule(), input_size=(5,))


def test_empty_module() -> None:
    summary(EmptyModule())


def test_device() -> None:
    model = SingleInputNet()
    # input_size
    summary(model, input_size=(5, 1, 28, 28), device="cpu")

    # input_data
    input_data = torch.randn(5, 1, 28, 28)
    summary(model, input_data=input_data)
    summary(model, input_data=input_data, device="cpu")
    summary(model, input_data=input_data.to("cpu"))
    summary(model, input_data=input_data.to("cpu"), device=torch.device("cpu"))


def test_pack_padded() -> None:
    # use explicit device=cpu
    # see: https://github.com/pytorch/pytorch/issues/43227
    device = torch.device("cpu")

    x = torch.ones([20, 128]).long().to(device)
    # fmt: off

    y = torch.Tensor([
        13, 12, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
    ]).long().to(device)
    # fmt: on

    summary(PackPaddedLSTM(), input_data=x, lengths=y, device=device)


def test_module_dict() -> None:
    summary(
        ModuleDictModel(),
        input_data=torch.randn(1, 10, 3, 3),
        layer_type="conv",
        activation_type="lrelu",
    )
    summary(
        ModuleDictModel(),
        input_data=torch.randn(1, 10, 3, 3),
        layer_type="pool",
        activation_type="prelu",
    )


def test_highly_nested_dict_model() -> None:
    """
    Test the following three if-clauses
    from LayerInfo.calculate_size.extract_tensor: 1, 2, 4, 5
    (starts counting from 1)
    """
    model = HighlyNestedDictModel()
    summary(model, input_data=torch.ones(10))


def test_edgecase_input_output_model() -> None:
    """
    Test the following two if-clauses
    from LayerInfo.calculate_size.extract_tensor: 3
    (starts counting from 1) as well as the final return.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgecaseInputOutputModel().to(device)
    summary(model, input_data=[{}])


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

    # explicitly use cpu to prevent mixed device
    # when cuda is available
    summary(model, input_size=input_size, z=named_tuple, device=torch.device("cpu"))


def test_numpy_model() -> None:
    model = NumpyModel()
    summary(model, input_data=np.ones(3, dtype=np.float32))


def test_return_dict() -> None:
    input_size = [torch.Size([1, 28, 28]), [12]]

    metrics = summary(ReturnDict(), input_size=input_size, col_width=65, batch_dim=0)

    assert metrics.input_size == [(1, 28, 28), [12]]


def test_autoencoder() -> None:
    model = AutoEncoder()
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=(
            ColumnSettings.OUTPUT_SIZE,
            ColumnSettings.NUM_PARAMS,
            ColumnSettings.KERNEL_SIZE,
        ),
    )


def test_reusing_activation_layers() -> None:
    act = nn.LeakyReLU(inplace=True)
    model1 = nn.Sequential(act, nn.Identity(), act, nn.Identity(), act)
    model2 = nn.Sequential(
        nn.LeakyReLU(inplace=True),
        nn.Identity(),
        nn.LeakyReLU(inplace=True),
        nn.Identity(),
        nn.LeakyReLU(inplace=True),
    )

    result_1 = summary(model1)
    result_2 = summary(model2)

    assert len(result_1.summary_list) == len(result_2.summary_list) == 6


def test_mixed_trainable_parameters() -> None:
    result = summary(MixedTrainableParameters(), verbose=Verbosity.VERBOSE)

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


def test_trainable_column() -> None:
    summary(
        MixedTrainable(),
        input_size=(1, 1, 1),
        col_names=("kernel_size", "input_size", "output_size", "trainable"),
    )


def test_empty_module_list() -> None:
    summary(nn.ModuleList())


def test_single_parameter_model() -> None:
    class ParameterA(nn.Module):  # pylint: disable=too-few-public-methods
        """A model with one parameter."""

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1024))

    class ParameterB(nn.Module):  # pylint: disable=too-few-public-methods
        """A model with one parameter and one Conv2d layer."""

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1024))
            self.conv = nn.Conv2d(3, 6, 3)

    summary(ParameterA())
    summary(ParameterB())
    summary(ParameterA(), verbose=2)
    summary(ParameterB(), verbose=2)


def test_register_parameter() -> None:
    model = RegisterParameter(nn.Linear(2, 2), nn.Linear(2, 2))
    result = summary(model)
    expected = sum(w.numel() for w in model.parameters() if w.requires_grad)
    assert result.total_params == expected == 14


def test_parameters_with_other_layers() -> None:
    input_data = torch.randn(3, 128)
    summary(ParameterFCNet(128, 64, 32), input_data=input_data, verbose=2)
    summary(ParameterFCNet(128, 64), input_data=input_data, verbose=2)


def test_nested_leftover_params() -> None:
    x = torch.zeros(100, 2)
    result = summary(InsideModel(), input_data=[x], row_settings=("var_names",))
    expected = sum(p.numel() for p in InsideModel().parameters() if p.requires_grad)
    assert result.total_params == expected == 8


def test_recursive_with_missing_layers() -> None:
    summary(
        RecursiveWithMissingLayers(),
        input_data=[torch.rand((2, 3, 128, 128))],
        row_settings=("depth", "var_names"),
    )


def test_cnn_module_list() -> None:
    summary(CNNModuleList(ConvLayerA), input_size=[1, 1, 10])
    summary(CNNModuleList(ConvLayerB), input_size=[1, 1, 10])


def test_hide_recursive_layers() -> None:
    model = SimpleRNN()
    summary(model, input_size=(2, 3))
    summary(model, input_size=(2, 3), row_settings=("depth", "hide_recursive_layers"))


def test_hide_recursive_layers_outside_loop() -> None:
    model = SimpleRNN(repeat_outside_loop=True)
    summary(model, input_size=(2, 3))
    summary(model, input_size=(2, 3), row_settings=("depth", "hide_recursive_layers"))
