import sys
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
from torch import nn

from torchinfo import summary
from torchinfo.torchinfo import get_total_memory_used


class MetadataProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False)

    def forward(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        if modality != "image":
            raise ValueError("Expected image features")
        result: torch.Tensor = self.projection(features)
        return result


@pytest.mark.parametrize("container", [list, tuple])
def test_positional_metadata_matches_mapping(container: Any) -> None:
    model = MetadataProjection().eval()
    features = torch.ones(2, 3)
    inputs = container((features, "image"))
    original = features.clone()
    with torch.no_grad():
        expected_output = model(*inputs)
    mapped = summary(
        model,
        input_data={"features": features, "modality": "image"},
        device=torch.device("cpu"),
        mode="same",
        verbose=0,
    )
    positional = summary(
        model, input_data=inputs, device=torch.device("cpu"), mode="same", verbose=0
    )
    assert positional.total_input == mapped.total_input
    assert positional.total_input == sys.getsizeof(features.untyped_storage())
    assert positional.total_params == mapped.total_params == 6
    assert positional.summary_list[0].output_size == list(expected_output.shape)
    assert inputs[0] is features and inputs[1] == "image"
    assert torch.equal(features, original)


class ScaledProjection(nn.Module):
    def forward(self, features: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        assert len(options["labels"]) == features.shape[0]
        result: torch.Tensor = features * options["scale"]
        return result


def test_nested_metadata_summary_counts_only_tensor_memory() -> None:
    features = torch.ones(2, 3)
    labels = ["cat", "dog"]
    flags = (True, None, 17)
    options = {"labels": labels, "scale": 2, "flags": flags}
    model = ScaledProjection().eval()
    assert torch.equal(model(features, options), features * 2)
    result = summary(
        model,
        input_data=(features, options),
        device=torch.device("cpu"),
        mode="same",
        verbose=0,
    )
    assert result.total_input == sys.getsizeof(features.untyped_storage())
    assert options == {"labels": ["cat", "dog"], "scale": 2, "flags": flags}
    assert options["labels"] is labels
    assert options["flags"] is flags


@pytest.mark.parametrize("container", ["list", "tuple", "mapping", "namedtuple"])
@pytest.mark.parametrize("metadata", ["image", None, 17, True, 1.5, b"image"])
def test_memory_ignores_non_tensor_leaves(metadata: Any, container: str) -> None:
    tensor = torch.ones(2, 3)
    array = np.ones((3, 2), dtype=np.float32)

    class Pair(NamedTuple):
        tensor: torch.Tensor
        metadata: Any

    batch: Any
    if container == "namedtuple":
        batch = Pair(tensor, metadata)
    elif container == "mapping":
        batch = {"tensor": tensor, "metadata": metadata}
    elif container == "tuple":
        batch = (tensor, metadata)
    else:
        batch = [tensor, metadata]
    data = {"batch": [batch, (array, {"option": metadata})]}
    expected = sys.getsizeof(tensor.untyped_storage()) + sys.getsizeof(
        torch.from_numpy(array).untyped_storage()
    )
    assert get_total_memory_used(data) == expected
