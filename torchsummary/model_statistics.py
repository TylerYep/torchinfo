from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from .formatting import FormattingOptions, Verbosity
from .layer_info import LayerInfo

HEADER_TITLES = {
    "kernel_size": "Kernel Shape",
    "output_size": "Output Shape",
    "num_params": "Param #",
    "mult_adds": "Mult-Adds",
}
CORRECTED_INPUT_SIZE_TYPE = List[Union[Sequence[Any], torch.Size]]


class ModelStatistics:
    """ Class for storing results of the summary. """

    def __init__(
        self,
        summary_list: List[LayerInfo],
        input_size: CORRECTED_INPUT_SIZE_TYPE,
        formatting: FormattingOptions,
    ):
        self.summary_list = summary_list
        self.input_size = input_size
        self.total_input = sum([abs(np.prod(sz)) for sz in input_size])
        self.formatting = formatting
        self.total_params, self.trainable_params = 0, 0
        self.total_output, self.total_mult_adds = 0, 0
        for layer_info in summary_list:
            self.total_mult_adds += layer_info.macs
            if not layer_info.is_recursive:
                if layer_info.depth == formatting.max_depth or (
                    not any(layer_info.module.children())
                    and layer_info.depth < formatting.max_depth
                ):
                    self.total_params += layer_info.num_params
                    if layer_info.trainable:
                        self.trainable_params += layer_info.num_params
                if layer_info.num_params > 0 and not any(layer_info.module.children()):
                    # x2 for gradients
                    self.total_output += 2.0 * abs(np.prod(layer_info.output_size))

    @staticmethod
    def to_bytes(num: int) -> float:
        """ Converts a number (assume floats, 4 bytes each) to megabytes or gigabytes. """
        assert num >= 0
        if num >= 1e9:
            return num * 4 / (1024 ** 3)
        return num * 4 / (1024 ** 2)

    @staticmethod
    def to_readable(num: int) -> float:
        """ Converts a number to millions or billions. """
        assert num >= 0
        if num >= 1e9:
            return num / 1e9
        return num / 1e6

    def __repr__(self) -> str:
        """ Print results of the summary. """
        header_row = self.formatting.format_row("Layer (type:depth-idx)", HEADER_TITLES)
        layer_rows = self.layers_to_str()

        total_size = self.total_input + self.total_output + self.total_params
        width = self.formatting.get_total_width()
        summary_str = (
            f"{'-' * width}\n"
            f"{header_row}"
            f"{'=' * width}\n"
            f"{layer_rows}"
            f"{'=' * width}\n"
            f"Total params: {self.total_params:,}\n"
            f"Trainable params: {self.trainable_params:,}\n"
            f"Non-trainable params: {self.total_params - self.trainable_params:,}\n"
            f"Total mult-adds ({'G' if self.total_mult_adds >= 1e9 else 'M'}): "
            f"{self.to_readable(self.total_mult_adds):0.2f}\n"
            f"{'-' * width}\n"
            f"Input size (MB): {self.to_bytes(self.total_input):0.2f}\n"
            f"Forward/backward pass size (MB): {self.to_bytes(self.total_output):0.2f}\n"
            f"Params size (MB): {self.to_bytes(self.total_params):0.2f}\n"
            f"Estimated Total Size (MB): {self.to_bytes(total_size):0.2f}\n"
            f"{'-' * width}"
        )
        return summary_str

    def layer_info_to_row(self, layer_info: LayerInfo, reached_max_depth: bool = False) -> str:
        """ Convert layer_info to string representation of a row. """

        def get_start_str(depth: int) -> str:
            return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

        row_values = {
            "kernel_size": str(layer_info.kernel_size) if layer_info.kernel_size else "--",
            "output_size": str(layer_info.output_size),
            "num_params": layer_info.num_params_to_str(reached_max_depth),
            "mult_adds": layer_info.macs_to_str(reached_max_depth),
        }
        depth = layer_info.depth
        name = (get_start_str(depth) if self.formatting.use_branching else "") + str(layer_info)
        new_line = self.formatting.format_row(name, row_values)
        if self.formatting.verbose == Verbosity.VERBOSE.value:
            for inner_name, inner_shape in layer_info.inner_layers.items():
                prefix = get_start_str(depth + 1) if self.formatting.use_branching else "  "
                extra_row_values = {"kernel_size": str(inner_shape)}
                new_line += self.formatting.format_row(prefix + inner_name, extra_row_values)
        return new_line

    def layers_to_str(self) -> str:
        """ Print each layer of the model as tree or as a list. """
        if self.formatting.use_branching:
            return self._layer_tree_to_str()

        layer_rows = ""
        for layer_info in self.summary_list:
            layer_rows += self.layer_info_to_row(layer_info)
        return layer_rows

    def _layer_tree_to_str(self, left: int = 0, right: Optional[int] = None, depth: int = 1) -> str:
        """ Print each layer of the model using a fancy branching diagram. """
        if depth > self.formatting.max_depth:
            return ""
        new_left = left - 1
        new_str = ""
        if right is None:
            right = len(self.summary_list)
        for i in range(left, right):
            layer_info = self.summary_list[i]
            if layer_info.depth == depth:
                reached_max_depth = depth == self.formatting.max_depth
                new_str += self.layer_info_to_row(layer_info, reached_max_depth)
                new_str += self._layer_tree_to_str(new_left + 1, i, depth + 1)
                new_left = i
        return new_str
