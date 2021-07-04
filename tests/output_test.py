""" tests/output_test.py """
import pytest
import torch
import torchvision  # type: ignore[import]

from conftest import verify_output_str
from fixtures.models import (
    ContainerModule,
    CustomParameter,
    EdgeCaseModel,
    EmptyModule,
    LinearModel,
    MultipleInputNetDifferentDtypes,
    ParameterListModel,
    PartialJITModel,
    SingleInputNet,
)
from torchinfo import ALL_COLUMN_SETTINGS, summary


class TestOutputString:
    """Tests for output string."""

    @staticmethod
    def test_string_result() -> None:
        results = summary(SingleInputNet(), input_size=(16, 1, 28, 28))
        result_str = f"{results}\n"

        verify_output_str(result_str, "tests/test_output/string_result.out")

    @staticmethod
    def test_single_input_all_cols() -> None:
        model = SingleInputNet()
        col_names = (
            "kernel_size",
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
        )
        input_shape = (7, 1, 28, 28)
        summary(
            model,
            input_data=torch.randn(*input_shape),
            depth=1,
            col_names=col_names,
            col_width=20,
        )

    @staticmethod
    def test_single_input_batch_dim() -> None:
        model = SingleInputNet()
        col_names = (
            "kernel_size",
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
        )
        summary(
            model,
            input_size=(1, 28, 28),
            depth=1,
            col_names=col_names,
            col_width=20,
            batch_dim=0,
        )

    @staticmethod
    def test_basic_summary() -> None:
        model = SingleInputNet()

        summary(model)

    @staticmethod
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

    @staticmethod
    def test_resnet18_depth_consistency() -> None:
        model = torchvision.models.resnet18()

        summary(model, (1, 3, 64, 64), depth=1)
        summary(model, (1, 3, 64, 64), depth=2)

    @staticmethod
    def test_resnet152() -> None:
        model = torchvision.models.resnet152()

        summary(model, (1, 3, 224, 224), depth=3)

    @staticmethod
    def test_dict_input() -> None:
        # TODO: expand this test to handle intermediate dict layers.
        model = MultipleInputNetDifferentDtypes()
        input_data = torch.randn(1, 300)
        other_input_data = torch.randn(1, 300).long()

        summary(model, input_data={"x1": input_data, "x2": other_input_data})

    @staticmethod
    def test_row_settings() -> None:
        model = SingleInputNet()

        summary(model, input_size=(16, 1, 28, 28), row_settings=("var_names",))

    @staticmethod
    def test_jit() -> None:
        model = LinearModel()
        model_jit = torch.jit.script(model)
        x = torch.randn(64, 128)

        regular_model = summary(model, input_data=x)
        jit_model = summary(model_jit, input_data=x)

        assert len(regular_model.summary_list) == len(jit_model.summary_list)

    @staticmethod
    def test_partial_jit() -> None:
        model_jit = torch.jit.script(PartialJITModel())

        summary(model_jit, input_data=torch.randn(2, 1, 28, 28))

    @staticmethod
    def test_custom_parameter() -> None:
        model = CustomParameter(8, 4)

        summary(model, input_size=(1,))

    @staticmethod
    def test_parameter_list() -> None:
        model = ParameterListModel()

        summary(
            model,
            input_size=(100, 100),
            verbose=2,
            col_names=ALL_COLUMN_SETTINGS,
            col_width=15,
        )


class TestEdgeCaseOutputString:
    """Tests for edge case output strings."""

    @staticmethod
    def test_exception() -> None:
        input_size = (1, 1, 28, 28)
        summary(EdgeCaseModel(throw_error=False), input_size=input_size)
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), input_size=input_size)

    @staticmethod
    def test_container() -> None:
        summary(ContainerModule(), input_size=(1, 5), depth=4)

    @staticmethod
    def test_empty_module() -> None:
        summary(EmptyModule())
