from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Union

import torch
from torch import nn
from torch.jit import ScriptModule

from .enums import ColumnSettings

DETECTED_INPUT_OUTPUT_TYPES = Union[
    Sequence[Any], Dict[Any, torch.Tensor], torch.Tensor
]


class LayerInfo:
    """Class that holds information about a layer module."""

    def __init__(
        self,
        var_name: str,
        module: nn.Module,
        depth: int,
        parent_info: LayerInfo | None = None,
    ) -> None:
        # Identifying information
        self.layer_id = id(module)
        self.module = module
        self.class_name = (
            str(module.original_name)
            if isinstance(module, ScriptModule)
            else module.__class__.__name__
        )
        # {layer name: {col_name: value_for_row}}
        self.inner_layers: dict[str, dict[ColumnSettings, Any]] = {}
        self.depth = depth
        self.depth_index: int | None = None  # set at the very end
        self.executed = False
        self.parent_info = parent_info
        self.var_name = var_name
        self.is_leaf_layer = not any(self.module.children())

        # Statistics
        self.trainable_params = 0
        self.is_recursive = False
        self.input_size: list[int] = []
        self.output_size: list[int] = []
        self.kernel_size: list[int] = []
        self.num_params = 0
        self.macs = 0

    def __repr__(self) -> str:
        return f"{self.class_name}: {self.depth}"

    @staticmethod
    def calculate_size(
        inputs: DETECTED_INPUT_OUTPUT_TYPES, batch_dim: int | None
    ) -> list[int]:
        """Set input_size or output_size using the model's inputs."""

        def nested_list_size(inputs: Sequence[Any]) -> list[int]:
            """Flattens nested list size."""
            if hasattr(inputs, "tensors"):
                return nested_list_size(inputs.tensors)  # type: ignore[attr-defined]
            if (
                isinstance(inputs, torch.Tensor)
                or not hasattr(inputs, "__getitem__")
                or not inputs
            ):
                return []
            if isinstance(inputs[0], dict):
                return nested_list_size(list(inputs[0].items()))
            if hasattr(inputs[0], "size") and callable(inputs[0].size):
                return list(inputs[0].size())
            if isinstance(inputs, (list, tuple)):
                return nested_list_size(inputs[0])
            return []

        size = []
        # pack_padded_seq and pad_packed_seq store feature into data attribute
        if isinstance(inputs, (list, tuple)) and inputs and hasattr(inputs[0], "data"):
            size = list(inputs[0].data.size())
            if batch_dim is not None:
                size = size[:batch_dim] + [1] + size[batch_dim + 1 :]

        elif isinstance(inputs, dict):
            # TODO avoid overwriting the previous size every time?
            for _, output in inputs.items():
                size = list(output.size())
                if batch_dim is not None:
                    size = [size[:batch_dim] + [1] + size[batch_dim + 1 :]]

        elif isinstance(inputs, torch.Tensor):
            size = list(inputs.size())
            if batch_dim is not None and batch_dim < len(size):
                size[batch_dim] = 1

        elif isinstance(inputs, (list, tuple)):
            size = nested_list_size(inputs)

        else:
            raise TypeError(
                "Model contains a layer with an unsupported input or output type: "
                f"{inputs}, type: {type(inputs)}"
            )

        return size

    def get_layer_name(self, show_var_name: bool, show_depth: bool) -> str:
        layer_name = self.class_name
        if show_var_name and self.var_name:
            layer_name += f" ({self.var_name})"
        if show_depth and self.depth > 0:
            layer_name += f": {self.depth}"
            if self.depth_index is not None:
                layer_name += f"-{self.depth_index}"
        return layer_name

    def get_param_count(self, name: str, param: torch.Tensor) -> tuple[int, str]:
        """
        Get count of number of params, accounting for mask.

        Masked models save parameters with the suffix "_orig" added.
        They have a buffer ending with "_mask" which has only 0s and 1s.
        If a mask exists, the sum of 1s in mask is number of params.
        """
        if name.endswith("_orig"):
            without_suffix = name[:-5]
            pruned_weights = rgetattr(self.module, f"{without_suffix}_mask")
            if pruned_weights is not None:
                parameter_count = int(torch.sum(pruned_weights))
                return parameter_count, without_suffix
        return param.nelement(), name

    def calculate_num_params(self) -> None:
        """
        Set num_params, trainable, inner_layers, and kernel_size
        using the module's parameters.
        """
        name = ""
        for name, param in self.module.named_parameters():
            cur_params, name = self.get_param_count(name, param)

            self.num_params += cur_params
            if param.requires_grad:
                self.trainable_params += cur_params

            ksize = list(param.size())
            if name == "weight":
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]
                self.kernel_size = ksize

            # RNN modules have inner weights such as weight_ih_l0
            self.inner_layers[name] = {
                ColumnSettings.KERNEL_SIZE: str(ksize),
                ColumnSettings.NUM_PARAMS: f"├─{cur_params:,}",
            }
        if self.inner_layers:
            self.inner_layers[name][
                ColumnSettings.NUM_PARAMS
            ] = f"└─{self.inner_layers[name][ColumnSettings.NUM_PARAMS][2:]}"

    def calculate_macs(self) -> None:
        """
        Set MACs using the module's parameters and layer's output size, which is
        used for computing number of operations for Conv layers.

        Please note: Returned MACs is the number of MACs for the full tensor,
        i.e., taking the batch-dimension into account.
        """
        for name, param in self.module.named_parameters():
            cur_params, name = self.get_param_count(name, param)
            if name in ("weight", "bias"):
                # ignore C when calculating Mult-Adds in ConvNd
                if "Conv" in self.class_name:
                    self.macs += int(
                        cur_params * prod(self.output_size[:1] + self.output_size[2:])
                    )
                else:
                    self.macs += self.output_size[0] * cur_params
            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name or "bias" in name:
                self.macs += prod(self.output_size[:2]) * cur_params

    def check_recursive(self, summary_list: list[LayerInfo]) -> None:
        """
        If the current module is already-used, mark as (recursive).
        Must check before adding line to the summary.
        """
        if any(self.module.named_parameters()):
            for other_layer in summary_list:
                if self.layer_id == other_layer.layer_id:
                    self.is_recursive = True

    def macs_to_str(
        self, reached_max_depth: bool, children_layers: list[LayerInfo]
    ) -> str:
        """Convert MACs to string."""
        if self.macs <= 0:
            return "--"
        if self.is_leaf_layer:
            return f"{self.macs:,}"
        if reached_max_depth:
            sum_child_macs = sum(
                child.macs for child in children_layers if child.is_leaf_layer
            )
            return f"{sum_child_macs:,}"
        return "--"

    def num_params_to_str(self, reached_max_depth: bool) -> str:
        """Convert num_params to string."""
        if self.is_recursive:
            return "(recursive)"
        if self.num_params > 0 and (reached_max_depth or self.is_leaf_layer):
            param_count_str = f"{self.num_params:,}"
            return param_count_str if self.trainable_params else f"({param_count_str})"
        return "--"


def prod(num_list: Iterable[int] | torch.Size) -> int:
    result = 1
    if isinstance(num_list, Iterable):
        for item in num_list:
            result *= prod(item) if isinstance(item, Iterable) else item
    return result


def rgetattr(module: nn.Module, attr: str) -> torch.Tensor | None:
    """Get the tensor submodule called attr from module."""
    for attr_i in attr.split("."):
        if not hasattr(module, attr_i):
            return None
        module = getattr(module, attr_i)
    assert isinstance(module, torch.Tensor)
    return module
