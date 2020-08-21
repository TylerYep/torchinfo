""" unit_test/torchsummary_test.py """
# pylint: disable=no-self-use
import pytest
import torch

from fixtures.models import CustomModule, EdgeCaseModel, Identity
from torchsummary.torchsummary import summary


class TestExceptions:
    """ Test torchsummary on various edge cases. """

    def test_invalid_user_params(self) -> None:
        test = Identity()

        with pytest.raises(ValueError):
            summary(test, verbose=4)
        with pytest.raises(ValueError):
            summary(test, (1, 28, 28), col_names=("invalid_name",))
        with pytest.raises(ValueError):
            summary(test, col_names=("output_size",))

    def test_incorrect_model_forward(self) -> None:
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), (1, 28, 28))
        with pytest.raises(TypeError):
            summary(EdgeCaseModel(return_str=True), (1, 28, 28))
        with pytest.raises(TypeError):
            summary(EdgeCaseModel(return_class=True), (1, 28, 28))
        with pytest.raises(RuntimeError):
            summary(EdgeCaseModel(throw_error=True), [[[torch.randn(1, 28, 28)]]])

    def test_input_size_possibilities(self) -> None:
        test = CustomModule(2, 3)

        with pytest.raises(ValueError):
            summary(test, [(3, 0)])
        with pytest.raises(TypeError):
            summary(test, {0: 1})  # type: ignore
        with pytest.raises(TypeError):
            summary(test, "hello")
