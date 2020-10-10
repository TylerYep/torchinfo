""" unit_test/torchinfo_test.py """
import pytest
import torch

from fixtures.models import CustomModule, EdgeCaseModel, Identity
from torchinfo import summary


class TestExceptions:
    """ Test torchinfo on various edge cases. """

    @staticmethod
    def test_invalid_user_params() -> None:
        test = Identity()

        with pytest.raises(ValueError):
            summary(test, verbose=4)
        with pytest.raises(ValueError):
            summary(test, (1, 28, 28), col_names=("invalid_name",))
        with pytest.raises(ValueError):
            summary(test, col_names=("output_size",))

    @staticmethod
    def test_incorrect_model_forward() -> None:
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), (1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(return_str=True), (1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(return_class=True), (1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), [[[torch.randn(1, 28, 28)]]])

    @staticmethod
    def test_input_size_possibilities() -> None:
        test = CustomModule(2, 3)

        with pytest.raises(ValueError):
            summary(test, [(3, 0)])
        with pytest.raises(TypeError):
            summary(test, {0: 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            summary(test, "hello")
