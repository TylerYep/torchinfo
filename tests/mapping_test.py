from collections import UserDict
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from torchinfo import ModelStatistics, summary


MappingFactory = Callable[[dict[str, Any]], Mapping[str, Any]]
MAPPING_TYPES = [dict, UserDict, MappingProxyType, BatchEncoding]


class KeywordModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, *, bias: torch.Tensor
    ) -> torch.Tensor:
        result: torch.Tensor = self.linear(x * mask) + bias
        return result


class MappingLayer(nn.Module):
    def __init__(self, factory: MappingFactory, mapping_output: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.factory = factory
        self.mapping_output = mapping_output

    def forward(self, payload: Mapping[str, torch.Tensor]) -> Any:
        output = self.linear(payload["x"])
        return self.factory({"scores": output}) if self.mapping_output else output


class NestedModel(nn.Module):
    def __init__(self, factory: MappingFactory, mapping_output: bool) -> None:
        super().__init__()
        self.layer = MappingLayer(factory, mapping_output)

    def forward(self, payload: Mapping[str, torch.Tensor]) -> Any:
        return self.layer(payload)


def assert_same_statistics(actual: ModelStatistics, expected: ModelStatistics) -> None:
    assert actual.input_size == expected.input_size
    for field in (
        "total_input",
        "total_mult_adds",
        "total_params",
        "trainable_params",
        "total_param_bytes",
        "total_output_bytes",
    ):
        assert getattr(actual, field) == getattr(expected, field)
    assert [layer.input_size for layer in actual.summary_list] == [
        layer.input_size for layer in expected.summary_list
    ]
    assert [layer.output_size for layer in actual.summary_list] == [
        layer.output_size for layer in expected.summary_list
    ]
    assert str(actual) == str(expected)


@pytest.mark.parametrize("factory", MAPPING_TYPES)
def test_mapping_keyword_inputs(factory: MappingFactory) -> None:
    model = KeywordModel()
    inputs = {"x": torch.ones(2, 3), "mask": torch.ones(2, 3)}
    bias = torch.ones(2)
    mapped = factory(inputs)
    torch.testing.assert_close(model(**mapped, bias=bias), model(**inputs, bias=bias))
    expected = summary(model, input_data=inputs, bias=bias, verbose=0)
    actual = summary(model, input_data=mapped, bias=bias, verbose=0)
    assert_same_statistics(actual, expected)
    assert all(mapped[key] is value for key, value in inputs.items())


@pytest.mark.parametrize("factory", MAPPING_TYPES)
@pytest.mark.parametrize("mapping_output", [False, True])
@pytest.mark.parametrize("batch_dim", [None, 0])
def test_nested_mapping_statistics(
    factory: MappingFactory, mapping_output: bool, batch_dim: int | None
) -> None:
    inputs = {"x": torch.ones(2, 3)}
    expected = summary(
        NestedModel(dict, mapping_output),
        input_data={"payload": inputs},
        batch_dim=batch_dim,
        verbose=0,
    )
    actual = summary(
        NestedModel(factory, mapping_output),
        input_data={"payload": factory(inputs)},
        batch_dim=batch_dim,
        verbose=0,
    )
    assert_same_statistics(actual, expected)


@pytest.mark.parametrize("sequence_type", [list, tuple])
def test_sequence_inputs_still_work(sequence_type: type[Any]) -> None:
    model = KeywordModel()
    data = [torch.ones(2, 3), torch.ones(2, 3)]
    bias = torch.ones(2)
    result = summary(model, input_data=sequence_type(data), bias=bias, verbose=0)
    assert result.total_params == 8
    assert result.total_mult_adds == 16
    assert result.summary_list[-1].output_size == [2, 2]
