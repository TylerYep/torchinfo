""" layer_info.py """
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

DETECTED_INPUT_OUTPUT_TYPES = Union[
    Sequence[Any], Dict[Any, torch.Tensor], torch.Tensor
]


class LayerInfo:
    """ Class that holds information about a layer module. """

    def __init__(
        self,
        module: nn.Module,
        depth: int,
        depth_index: Optional[int] = None,
        parent_info: Optional["LayerInfo"] = None,
    ):
        # Identifying information
        self.layer_id = id(module)
        self.module = module
        self.class_name = str(module.__class__).split(".")[-1].split("'")[0]
        self.inner_layers: Dict[str, List[int]] = {}
        self.depth = depth
        self.depth_index = depth_index
        self.executed = False
        self.parent_info = parent_info

        # Statistics
        self.trainable = True
        self.is_recursive = False
        self.input_size: List[int] = []
        self.output_size: List[int] = []
        self.kernel_size: List[int] = []
        self.num_params = 0
        self.macs = 0
        self.calculate_num_params()

    def __repr__(self) -> str:
        if self.depth_index is None:
            return f"{self.class_name}: {self.depth}"
        return f"{self.class_name}: {self.depth}-{self.depth_index}"

    @staticmethod
    def calculate_size(
        inputs: DETECTED_INPUT_OUTPUT_TYPES, batch_dim: Optional[int]
    ) -> List[int]:
        """ Set input_size or output_size using the model's inputs. """

        def nested_list_size(inputs: Sequence[Any]) -> List[int]:
            """ Flattens nested list size. """
            if hasattr(inputs[0], "size") and callable(inputs[0].size):
                return list(inputs[0].size())
            if isinstance(inputs, (list, tuple)):
                return nested_list_size(inputs[0])
            return []

        # pack_padded_seq and pad_packed_seq store feature into data attribute
        if isinstance(inputs, (list, tuple)) and len(inputs) == 0:
            size = []
        elif isinstance(inputs, (list, tuple)) and hasattr(inputs[0], "data"):
            size = list(inputs[0].data.size())
            if batch_dim is not None:
                size = size[:batch_dim] + [-1] + size[batch_dim + 1 :]

        elif isinstance(inputs, dict):
            # TODO avoid overwriting the previous size every time?
            for _, output in inputs.items():
                size = list(output.size())
                if batch_dim is not None:
                    size = [size[:batch_dim] + [-1] + size[batch_dim + 1 :]]

        elif isinstance(inputs, torch.Tensor):
            size = list(inputs.size())
            if batch_dim is not None:
                size[batch_dim] = -1

        elif isinstance(inputs, (list, tuple)):
            size = nested_list_size(inputs)

        else:
            raise TypeError(
                "Model contains a layer with an unsupported "
                "input or output type: {}".format(inputs)
            )

        return size

    def calculate_num_params(self) -> None:
        """
        Set num_params, trainable, inner_layers, and kernel_size
        using the module's parameters.
        """
        for name, param in self.module.named_parameters():
            self.num_params += param.nelement()
            self.trainable &= param.requires_grad

            if name == "weight":
                ksize = list(param.size())
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]
                self.kernel_size = ksize

            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name:
                self.inner_layers[name] = list(param.size())

    def calculate_macs(self) -> None:
        """
        Set MACs using the module's parameters and layer's output size, which is
        used for computing number of operations for Conv layers.
        """
        for name, param in self.module.named_parameters():
            if name == "weight":
                # ignore N, C when calculate Mult-Adds in ConvNd
                if "Conv" in self.class_name:
                    self.macs += int(param.nelement() * prod(self.output_size[2:]))
                else:
                    self.macs += param.nelement()
            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name:
                self.macs += param.nelement()

    def check_recursive(self, summary_list: List["LayerInfo"]) -> None:
        """
        If the current module is already-used, mark as (recursive).
        Must check before adding line to the summary.
        """
        if list(self.module.named_parameters()):
            for other_layer in summary_list:
                if self.layer_id == other_layer.layer_id:
                    self.is_recursive = True

    def macs_to_str(self, reached_max_depth: bool) -> str:
        """ Convert MACs to string. """
        if self.num_params > 0 and (
            reached_max_depth or not any(self.module.children())
        ):
            return f"{self.macs:,}"
        return "--"

    def num_params_to_str(self, reached_max_depth: bool = False) -> str:
        """ Convert num_params to string. """
        if self.is_recursive:
            return "(recursive)"
        if self.num_params > 0:
            param_count_str = f"{self.num_params:,}"
            if reached_max_depth or not any(self.module.children()):
                if not self.trainable:
                    return f"({param_count_str})"
                return param_count_str
        return "--"


def prod(num_list: Union[Iterable[Any], torch.Size]) -> int:
    result = 1
    for num in num_list:
        result *= num
    return abs(result)
