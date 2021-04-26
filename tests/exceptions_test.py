""" tests/torchinfo_test.py """
import pytest
import torch

from fixtures.models import CustomModule, EdgeCaseModel, IdentityModel
from torchinfo import summary


class TestExceptions:
    """Test torchinfo on various edge cases."""

    @staticmethod
    def test_invalid_user_params() -> None:
        test = IdentityModel()

        with pytest.raises(ValueError):
            summary(test, verbose=4)
        with pytest.raises(ValueError):
            summary(test, input_size=(1, 28, 28), col_names=("invalid_name",))
        with pytest.raises(ValueError):
            summary(test, col_width=0)
        with pytest.raises(ValueError):
            summary(test, row_settings=("invalid_name",))
        with pytest.raises(ValueError):
            summary(test, col_names=("output_size",))
        with pytest.raises(RuntimeError):
            summary(test, (1, 28, 28), torch.randn(1, 28, 28))

    @staticmethod
    def test_incorrect_model_forward() -> None:
        # Warning: these tests always raise RuntimeError.
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), input_size=(5, 1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(return_str=True), input_size=(5, 1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(return_class=True), input_size=(5, 1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(
                EdgeCaseModel(throw_error=True),
                input_data=[[[torch.randn(1, 28, 28)]]],  # type: ignore[list-item]
            )

    @staticmethod
    def test_input_size_possibilities() -> None:
        test = CustomModule(2, 3)

        with pytest.raises(ValueError):
            summary(test, input_size=[(3, 0)])
        with pytest.raises(TypeError):
            summary(test, input_size={0: 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            summary(test, input_size="hello")
