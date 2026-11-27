from collections import UserDict
from collections.abc import Callable, Mapping
from types import MappingProxyType

import pytest
import torch
from torch import nn
from torch.nn import functional

from torchinfo import summary

MappingFactory = Callable[[dict[str, torch.Tensor]], Mapping[str, torch.Tensor]]


class OutputLinear(nn.Module):
    def __init__(self, factory: MappingFactory | None, use_child: bool) -> None:
        super().__init__()
        self.factory = factory
        self.use_child = use_child
        if use_child:
            self.linear = nn.Linear(3, 2)
        else:
            self.weight = nn.Parameter(torch.ones(2, 3))
            self.bias = nn.Parameter(torch.zeros(2))

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor | Mapping[str, torch.Tensor]:
        output: torch.Tensor = (
            self.linear(inputs)
            if self.use_child
            else functional.linear(inputs, self.weight, self.bias)
        )
        return output if self.factory is None else self.factory({"output": output})


@pytest.mark.parametrize("factory", [None, dict, UserDict, MappingProxyType])
@pytest.mark.parametrize("batch_dim", [None, 0, 1])
@pytest.mark.parametrize("use_child", [False, True])
def test_mapping_output_batch_dimensions(
    factory: MappingFactory | None, batch_dim: int | None, use_child: bool
) -> None:
    model = OutputLinear(factory, use_child)
    expected_shape = [4, 2]
    inputs = torch.ones(4, 3)
    if batch_dim is not None:
        expected_shape.insert(batch_dim, 1)
        inputs = inputs.unsqueeze(batch_dim)
    output = model(inputs)
    tensor_output = output if isinstance(output, torch.Tensor) else output["output"]
    assert list(tensor_output.shape) == expected_shape

    result = summary(
        model,
        input_size=(4, 3),
        batch_dim=batch_dim,
        device=torch.device("cpu"),
        verbose=0,
    )

    assert result.total_mult_adds == 32
    assert result.total_params == result.trainable_params == 8
    assert result.total_param_bytes == 32
    assert result.total_output_bytes == 64
    assert all(layer.executed for layer in result.summary_list)
    assert all(layer.output_size == expected_shape for layer in result.summary_list)


class OutputScale(nn.Module):
    def __init__(self, factory: MappingFactory | None) -> None:
        super().__init__()
        self.factory = factory
        self.weight = nn.Parameter(torch.ones(3))

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor | Mapping[str, torch.Tensor]:
        output = inputs * self.weight
        return output if self.factory is None else self.factory({"output": output})


@pytest.mark.parametrize("factory", [dict, UserDict, MappingProxyType])
@pytest.mark.parametrize("batch_dim", [0, 1])
def test_mapping_output_with_direct_weight(
    factory: MappingFactory, batch_dim: int
) -> None:
    inputs = torch.ones(4, 3).unsqueeze(batch_dim)
    model = OutputScale(factory)
    output = model(inputs)
    assert isinstance(output, Mapping)
    torch.testing.assert_close(output["output"], inputs)
    expected = summary(
        OutputScale(None), input_data=inputs, batch_dim=batch_dim, verbose=0
    )
    actual = summary(model, input_data=inputs, batch_dim=batch_dim, verbose=0)

    assert actual.summary_list[0].output_size == list(inputs.shape)
    assert actual.total_params == actual.trainable_params == 3
    assert actual.total_param_bytes == 12
    assert actual.total_output_bytes == 96
    assert actual.total_input == expected.total_input
    assert actual.total_mult_adds == expected.total_mult_adds
