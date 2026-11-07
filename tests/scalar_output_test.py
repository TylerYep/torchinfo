import pytest
import torch
from torch import nn

from torchinfo import summary


@pytest.mark.parametrize("input_shape", [(), (4,), (2, 3)])
@pytest.mark.parametrize("use_input_size", [False, True])
def test_prelu_output_shapes(
    input_shape: tuple[int, ...], use_input_size: bool
) -> None:
    model = nn.PReLU()
    inputs = torch.full(input_shape, -1.0)
    assert model(inputs).shape == inputs.shape

    if use_input_size:
        result = summary(model, input_size=[input_shape], verbose=0)
    else:
        result = summary(model, input_data=inputs, verbose=0)

    layer = result.summary_list[0]
    assert layer.executed
    assert layer.input_size == list(input_shape)
    assert layer.output_size == list(input_shape)
    assert result.total_params == result.trainable_params == 1
    assert result.total_mult_adds == (input_shape[0] if input_shape else 1)
    assert result.total_param_bytes == model.weight.element_size()
    assert result.total_output_bytes == 2 * inputs.numel() * inputs.element_size()


class AffineMean(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.mean(inputs) * self.weight + self.bias


def test_parameterized_scalar_reduction() -> None:
    model = AffineMean()
    inputs = torch.ones(2, 3)
    assert model(inputs).shape == torch.Size([])

    result = summary(model, input_data=inputs, verbose=0)

    layer = result.summary_list[0]
    assert layer.executed
    assert layer.input_size == [2, 3]
    assert layer.output_size == []
    assert result.total_params == result.trainable_params == 2
    assert result.total_mult_adds == 2
    assert result.total_param_bytes == 2 * inputs.element_size()
    assert result.total_output_bytes == 2 * inputs.element_size()
