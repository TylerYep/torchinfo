from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.jit import ScriptModule

from .enums import ColumnSettings

try:
    from torch.nn.parameter import is_lazy
except ImportError:

    def is_lazy(param: nn.Parameter) -> bool:  # type: ignore[misc]
        del param
        return False


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
        self.children: list[LayerInfo] = []  # set at the very end
        self.executed = False
        self.parent_info = parent_info
        self.var_name = var_name
        self.is_leaf_layer = not any(self.module.children())
        self.contains_lazy_param = False

        # Statistics
        self.is_recursive = False
        self.input_size: list[int] = []
        self.output_size: list[int] = []
        self.kernel_size = self.get_kernel_size(module)
        self.trainable_params = 0
        self.num_params = 0
        self.param_bytes = 0
        self.output_bytes = 0
        self.macs = 0

    def __repr__(self) -> str:
        return f"{self.class_name}: {self.depth}"

    @property
    def trainable(self) -> str:
        """
        Checks if the module is trainable. Returns:
            "True", if all the parameters are trainable (`requires_grad=True`)
            "False" if none of the parameters are trainable.
            "Partial" if some weights are trainable, but not all.
            "--" if no module has no parameters, like Dropout.
        """
        if self.num_params == 0:
            return "--"
        if self.trainable_params == 0:
            return "False"
        if self.num_params == self.trainable_params:
            return "True"
        if self.num_params > self.trainable_params:
            return "Partial"
        raise RuntimeError("Unreachable trainable calculation.")

    @staticmethod
    def calculate_size(
        inputs: DETECTED_INPUT_OUTPUT_TYPES, batch_dim: int | None
    ) -> tuple[list[int], int]:
        """
        Set input_size or output_size using the model's inputs.
        Returns the corrected shape of `inputs` and the size of
        a single element in bytes.
        """
        if inputs is None:
            size, elem_bytes = [], 0

        # pack_padded_seq and pad_packed_seq store feature into data attribute
        elif (
            isinstance(inputs, (list, tuple))
            and inputs
            and hasattr(inputs[0], "data")
            and hasattr(inputs[0].data, "size")
        ):
            size = list(inputs[0].data.size())
            elem_bytes = inputs[0].data.element_size()
            if batch_dim is not None:
                size = size[:batch_dim] + [1] + size[batch_dim + 1 :]

        elif isinstance(inputs, dict):
            output = list(inputs.values())[-1]
            size, elem_bytes = nested_list_size(output)
            if batch_dim is not None:
                size = [size[:batch_dim] + [1] + size[batch_dim + 1 :]]

        elif isinstance(inputs, torch.Tensor):
            size = list(inputs.size())
            elem_bytes = inputs.element_size()

        elif isinstance(inputs, np.ndarray):
            inputs_ = torch.from_numpy(inputs)
            size, elem_bytes = list(inputs_.size()), inputs_.element_size()

        elif isinstance(inputs, (list, tuple)):
            size, elem_bytes = nested_list_size(inputs)
            if batch_dim is not None and batch_dim < len(size):
                size[batch_dim] = 1

        else:
            raise TypeError(
                "Model contains a layer with an unsupported input or output type: "
                f"{inputs}, type: {type(inputs)}"
            )

        return size, elem_bytes

    @staticmethod
    def get_param_count(
        module: nn.Module, name: str, param: torch.Tensor
    ) -> tuple[int, str]:
        """
        Get count of number of params, accounting for mask.

        Masked models save parameters with the suffix "_orig" added.
        They have a buffer ending with "_mask" which has only 0s and 1s.
        If a mask exists, the sum of 1s in mask is number of params.
        """
        if name.endswith("_orig"):
            without_suffix = name[:-5]
            pruned_weights = rgetattr(module, f"{without_suffix}_mask")
            if pruned_weights is not None:
                parameter_count = int(torch.sum(pruned_weights))
                return parameter_count, without_suffix
        return param.nelement(), name

    @staticmethod
    def get_kernel_size(module: nn.Module) -> int | list[int] | None:
        if hasattr(module, "kernel_size"):
            k = module.kernel_size
            kernel_size: int | list[int]
            if isinstance(k, Iterable):
                kernel_size = list(k)
            elif isinstance(k, int):
                kernel_size = int(k)
            else:
                raise TypeError(f"kernel_size has an unexpected type: {type(k)}")
            return kernel_size
        return None

    def get_layer_name(self, show_var_name: bool, show_depth: bool) -> str:
        layer_name = self.class_name
        if show_var_name and self.var_name:
            layer_name += f" ({self.var_name})"
        if show_depth and self.depth > 0:
            layer_name += f": {self.depth}"
            if self.depth_index is not None:
                layer_name += f"-{self.depth_index}"
        return layer_name

    def calculate_num_params(self) -> None:
        """
        Set num_params, trainable, inner_layers, and kernel_size
        using the module's parameters.
        """
        self.num_params = 0
        self.param_bytes = 0
        self.trainable_params = 0
        self.inner_layers = {}

        final_name = ""
        for name, param in self.module.named_parameters():
            if is_lazy(param):
                self.contains_lazy_param = True
                continue
            cur_params, name = self.get_param_count(self.module, name, param)
            self.param_bytes += param.element_size() * cur_params

            self.num_params += cur_params
            if param.requires_grad:
                self.trainable_params += cur_params

            # kernel_size for inner layer parameters
            ksize = list(param.size())
            if name == "weight":
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]

            # RNN modules have inner weights such as weight_ih_l0
            # Don't show parameters for the overall model, show for individual layers
            if self.parent_info is not None or "." not in name:
                self.inner_layers[name] = {
                    ColumnSettings.KERNEL_SIZE: str(ksize),
                    ColumnSettings.NUM_PARAMS: f"├─{cur_params:,}",
                }
                final_name = name
        # Fix the final row to display more nicely
        if self.inner_layers:
            self.inner_layers[final_name][
                ColumnSettings.NUM_PARAMS
            ] = f"└─{self.inner_layers[final_name][ColumnSettings.NUM_PARAMS][2:]}"

    def calculate_macs(self) -> None:
        """
        Set MACs using the module's parameters and layer's output size, which is
        used for computing number of operations for Conv layers.

        Please note: Returned MACs is the number of MACs for the full tensor,
        i.e., taking the batch-dimension into account.
        """
        for name, param in self.module.named_parameters():
            cur_params, name = self.get_param_count(self.module, name, param)
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

    def check_recursive(self, layer_ids: set[int]) -> None:
        """
        If the current module is already-used, mark as (recursive).
        Must check before adding line to the summary.
        """
        if self.layer_id in layer_ids:
            self.is_recursive = True

    def macs_to_str(self, reached_max_depth: bool) -> str:
        """Convert MACs to string."""
        if self.macs <= 0:
            return "--"
        if self.is_leaf_layer:
            return f"{self.macs:,}"
        if reached_max_depth:
            sum_child_macs = sum(
                child.macs for child in self.children if child.is_leaf_layer
            )
            return f"{sum_child_macs:,}"
        return "--"

    def num_params_to_str(self, reached_max_depth: bool) -> str:
        """Convert num_params to string."""
        if self.num_params == 0:
            return "--"
        if self.is_recursive:
            return "(recursive)"
        if reached_max_depth or self.is_leaf_layer:
            param_count_str = f"{self.num_params:,}"
            return param_count_str if self.trainable_params else f"({param_count_str})"
        leftover_params = self.leftover_params()
        return f"{leftover_params:,}" if leftover_params > 0 else "--"

    def params_percent(
        self, total_params: int, reached_max_depth: bool, precision: int = 2
    ) -> str:
        """Convert num_params to string."""
        spacing = 5
        zero = f"{' ' * spacing}--"
        if total_params == 0:
            return zero
        if self.is_recursive:
            return "(recursive)"
        params = (
            self.num_params
            if reached_max_depth or self.is_leaf_layer
            else self.leftover_params()
        )
        if params == 0:
            return zero
        return f"{params / total_params:>{precision + spacing}.{precision}%}"

    def leftover_params(self) -> int:
        """
        Leftover params are the number of params this current layer has that are not
        included in the child num_param counts.
        """
        return self.num_params - sum(
            child.num_params if child.is_leaf_layer else child.leftover_params()
            for child in self.children
            if not child.is_recursive
        )

    def leftover_trainable_params(self) -> int:
        return self.trainable_params - sum(
            child.trainable_params
            if child.is_leaf_layer
            else child.leftover_trainable_params()
            for child in self.children
            if not child.is_recursive
        )


def nested_list_size(inputs: Sequence[Any] | torch.Tensor) -> tuple[list[int], int]:
    """Flattens nested list size."""
    if hasattr(inputs, "tensors"):
        size, elem_bytes = nested_list_size(inputs.tensors)
    elif isinstance(inputs, torch.Tensor):
        size, elem_bytes = list(inputs.size()), inputs.element_size()
    elif isinstance(inputs, np.ndarray):
        inputs_torch = torch.from_numpy(inputs)  # preserves dtype
        size, elem_bytes = list(inputs_torch.size()), inputs_torch.element_size()
    elif not hasattr(inputs, "__getitem__") or not inputs:
        size, elem_bytes = [], 0
    elif isinstance(inputs, dict):
        size, elem_bytes = nested_list_size(list(inputs.values()))
    elif (
        hasattr(inputs, "size")
        and callable(inputs.size)
        and hasattr(inputs, "element_size")
        and callable(inputs.element_size)
    ):
        size, elem_bytes = list(inputs.size()), inputs.element_size()
    elif isinstance(inputs, (list, tuple)):
        size, elem_bytes = nested_list_size(inputs[0])
    else:
        size, elem_bytes = [], 0

    return size, elem_bytes


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


def get_children_layers(summary_list: list[LayerInfo], index: int) -> list[LayerInfo]:
    """Fetches all of the children of a given layer."""
    num_children = 0
    for layer in summary_list[index + 1 :]:
        if layer.depth <= summary_list[index].depth:
            break
        num_children += 1
    return summary_list[index + 1 : index + 1 + num_children]
