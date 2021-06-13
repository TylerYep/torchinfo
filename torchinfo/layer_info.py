""" layer_info.py """
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.jit import ScriptModule

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
        depth_index: Optional[int] = None,
        parent_info: Optional["LayerInfo"] = None,
    ) -> None:
        # Identifying information
        self.layer_id = id(module)
        self.module = module
        self.class_name = (
            str(module.original_name)
            if isinstance(module, ScriptModule)
            else module.__class__.__name__
        )
        # {layer name: {row_name: row_value}}
        self.inner_layers: Dict[str, Dict[str, Any]] = {}
        self.depth = depth
        self.depth_index = depth_index
        self.executed = False
        self.parent_info = parent_info
        self.var_name = var_name
        self.leaf_layer = not any(self.module.children())

        # Statistics
        self.trainable = True
        self.is_recursive = False
        self.input_size: List[int] = []
        self.output_size: List[int] = []
        self.kernel_size: List[int] = []
        self.num_params = 0
        self.macs = 0

    def __repr__(self) -> str:
        return f"{self.class_name}: {self.depth}"

    @staticmethod
    def calculate_size(
        inputs: DETECTED_INPUT_OUTPUT_TYPES, batch_dim: Optional[int]
    ) -> List[int]:
        """Set input_size or output_size using the model's inputs."""

        def nested_list_size(inputs: Sequence[Any]) -> List[int]:
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

    def calculate_num_params(self) -> None:
        """
        Set num_params, trainable, inner_layers, and kernel_size
        using the module's parameters.
        """
        for name, param in self.module.named_parameters():
            self.num_params += param.nelement()
            self.trainable &= param.requires_grad

            ksize = list(param.size())
            if name == "weight":
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]
                self.kernel_size = ksize

            # RNN modules have inner weights such as weight_ih_l0
            self.inner_layers[name] = {
                "kernel_size": str(ksize),
                "num_params": f"{param.nelement():,}",
            }

    def calculate_macs(self) -> None:
        """
        Set MACs using the module's parameters and layer's output size, which is
        used for computing number of operations for Conv layers.

        Please note: Returned MACs is the number of MACs for the full tensor,
        i.e., taking the batch-dimension into account.
        """
        if self.leaf_layer:
            for name, param in self.module.named_parameters():
                if name in ("weight", "bias"):
                    # ignore C when calculating Mult-Adds in ConvNd
                    if "Conv" in self.class_name:
                        self.macs += int(
                            param.nelement()
                            * prod(self.output_size[:1] + self.output_size[2:])
                        )
                    else:
                        self.macs += self.output_size[0] * param.nelement()
                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name or "bias" in name:
                    self.macs += prod(self.output_size[:2]) * param.nelement()

    def check_recursive(self, summary_list: List["LayerInfo"]) -> None:
        """
        If the current module is already-used, mark as (recursive).
        Must check before adding line to the summary.
        """
        if any(self.module.named_parameters()):
            for other_layer in summary_list:
                if self.layer_id == other_layer.layer_id:
                    self.is_recursive = True

    def macs_to_str(self, reached_max_depth: bool) -> str:
        """Convert MACs to string."""
        if self.macs > 0 and (reached_max_depth or self.leaf_layer):
            return f"{self.macs:,}"
        return "--"

    def num_params_to_str(self, reached_max_depth: bool) -> str:
        """Convert num_params to string."""
        if self.is_recursive:
            return "(recursive)"
        if self.num_params > 0 and (reached_max_depth or self.leaf_layer):
            param_count_str = f"{self.num_params:,}"
            return param_count_str if self.trainable else f"({param_count_str})"
        return "--"


def prod(num_list: Union[Iterable[int], torch.Size]) -> int:
    result = 1
    for item in num_list:
        result *= prod(item) if isinstance(item, Iterable) else item
    return result
