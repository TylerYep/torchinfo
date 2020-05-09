from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

DETECTED_OUTPUT_TYPES = Union[Sequence[Any], Dict[Any, torch.Tensor], torch.Tensor]


class LayerInfo:
    """ Class that holds information about a layer module. """

    def __init__(self, module: nn.Module, depth: int, depth_index: int):
        # Identifying information
        self.layer_id = id(module)
        self.module = module
        self.class_name = str(module.__class__).split(".")[-1].split("'")[0]
        self.inner_layers: Dict[str, List[int]] = {}
        self.depth = depth
        self.depth_index = depth_index

        # Statistics
        self.trainable = True
        self.is_recursive = False
        self.output_size: List[Union[int, Sequence[Any], torch.Size]] = []
        self.kernel_size: List[int] = []
        self.num_params = 0
        self.macs = 0

    def __repr__(self) -> str:
        return f"{self.class_name}: {self.depth}-{self.depth_index}"

    def calculate_output_size(self, outputs: DETECTED_OUTPUT_TYPES, batch_dim: int) -> None:
        """ Set output_size using the model's outputs. """
        if isinstance(outputs, (list, tuple)):
            try:
                self.output_size = list(outputs[0].size())
            except AttributeError:
                # pack_padded_seq and pad_packed_seq store feature into data attribute
                size = list(outputs[0].data.size())
                self.output_size = size[:batch_dim] + [-1] + size[batch_dim + 1 :]

        elif isinstance(outputs, dict):
            for _, output in outputs.items():
                size = list(output.size())
                size_with_batch = size[:batch_dim] + [-1] + size[batch_dim + 1 :]
                self.output_size.append(size_with_batch)

        elif isinstance(outputs, torch.Tensor):
            self.output_size = list(outputs.size())
            self.output_size[batch_dim] = -1

        else:
            raise TypeError(f"Model contains a layer with an unsupported output type: {outputs}")

    def calculate_num_params(self) -> None:
        """ Set num_params using the module's parameters.  """
        for name, param in self.module.named_parameters():
            self.num_params += param.nelement()
            self.trainable &= param.requires_grad

            if name == "weight":
                ksize = list(param.size())
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]
                self.kernel_size = ksize

                # ignore N, C when calculate Mult-Adds in ConvNd
                if "Conv" in self.class_name:
                    self.macs += int(param.nelement() * np.prod(self.output_size[2:]))
                else:
                    self.macs += param.nelement()

            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name:
                self.inner_layers[name] = list(param.size())
                self.macs += param.nelement()

    def check_recursive(self, summary_list: List[LayerInfo]) -> None:
        """ if the current module is already-used, mark as (recursive).
        Must check before adding line to the summary. """
        if list(self.module.named_parameters()):
            for other_layer in summary_list:
                if self.layer_id == other_layer.layer_id:
                    self.is_recursive = True

    def macs_to_str(self, reached_max_depth: bool) -> str:
        """ Convert MACs to string. """
        if self.num_params > 0 and (reached_max_depth or not any(self.module.children())):
            return f"{self.macs:,}"
        return "--"

    def num_params_to_str(self, reached_max_depth: bool = False) -> str:
        """ Convert num_params to string. """
        assert self.num_params >= 0
        if self.is_recursive:
            return "(recursive)"
        if self.num_params > 0:
            param_count_str = f"{self.num_params:,}"
            if reached_max_depth or not any(self.module.children()):
                if not self.trainable:
                    return f"({param_count_str})"
                return param_count_str
        return "--"
