""" model_statistics.py """
from typing import Any, Dict, Iterable, List, Union

import torch

from .formatting import FormattingOptions, Verbosity
from .layer_info import LayerInfo, prod

HEADER_TITLES = {
    "kernel_size": "Kernel Shape",
    "input_size": "Input Shape",
    "output_size": "Output Shape",
    "num_params": "Param #",
    "mult_adds": "Mult-Adds",
}
CORRECTED_INPUT_SIZE_TYPE = List[Union[Iterable[Any], torch.Size]]


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
        self.total_input = sum(prod(sz) for sz in input_size) if input_size else 0
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
                    self.total_output += 2 * prod(layer_info.output_size)

    def __repr__(self) -> str:
        """ Print results of the summary. """
        header_row = self.formatting.format_row("Layer (type:depth-idx)", HEADER_TITLES)
        layer_rows = self.layers_to_str()
        divider = "=" * self.formatting.get_total_width()
        summary_str = (
            "{0}\n{1}{0}\n{2}{0}"
            "\nTotal params: {3:,}\n"
            "Trainable params: {4:,}\n"
            "Non-trainable params: {5:,}\n".format(
                divider,
                header_row,
                layer_rows,
                self.total_params,
                self.trainable_params,
                self.total_params - self.trainable_params,
            )
        )
        if self.input_size:
            summary_str += (
                "Total mult-adds ({}): {:0.2f}\n"
                "{}\n"
                "Input size (MB): {:0.2f}\n"
                "Forward/backward pass size (MB): {:0.2f}\n"
                "Params size (MB): {:0.2f}\n"
                "Estimated Total Size (MB): {:0.2f}\n".format(
                    "G" if self.total_mult_adds >= 1e9 else "M",
                    self.to_readable(self.total_mult_adds),
                    divider,
                    self.to_bytes(self.total_input),
                    self.to_bytes(self.total_output),
                    self.to_bytes(self.total_params),
                    self.to_bytes(
                        self.total_input + self.total_output + self.total_params
                    ),
                )
            )
        summary_str += divider
        return summary_str

    @staticmethod
    def to_bytes(num: int) -> float:
        """ Converts a number (assume floats, 4 bytes each) to megabytes. """
        return num * 4 / (1024 ** 2)

    @staticmethod
    def to_readable(num: int) -> float:
        """ Converts a number to millions or billions. """
        if num >= 1e9:
            return num / 1e9
        return num / 1e6

    def layer_info_to_row(
        self, layer_info: LayerInfo, reached_max_depth: bool = False
    ) -> str:
        """ Convert layer_info to string representation of a row. """

        def get_start_str(depth: int) -> str:
            return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

        row_values = {
            "kernel_size": str(layer_info.kernel_size)
            if layer_info.kernel_size
            else "--",
            "input_size": str(layer_info.input_size),
            "output_size": str(layer_info.output_size),
            "num_params": layer_info.num_params_to_str(reached_max_depth),
            "mult_adds": layer_info.macs_to_str(reached_max_depth),
        }
        depth = layer_info.depth
        name = (get_start_str(depth) if self.formatting.use_branching else "") + str(
            layer_info
        )
        new_line = self.formatting.format_row(name, row_values)
        if self.formatting.verbose == Verbosity.VERBOSE.value:
            for inner_name, inner_shape in layer_info.inner_layers.items():
                prefix = (
                    get_start_str(depth + 1) if self.formatting.use_branching else "  "
                )
                extra_row_values = {"kernel_size": str(inner_shape)}
                new_line += self.formatting.format_row(
                    prefix + inner_name, extra_row_values
                )
        return new_line

    def layers_to_str(self) -> str:
        """ Print each layer of the model as tree or as a list. """
        if self.formatting.use_branching:
            return self._layer_tree_to_str()

        layer_rows = ""
        for layer_info in self.summary_list:
            layer_rows += self.layer_info_to_row(layer_info)
        return layer_rows

    def _layer_tree_to_str(self) -> str:
        """ Print each layer of the model using a fancy branching diagram. """
        new_str = ""
        current_hierarchy: Dict[int, LayerInfo] = {}

        for layer_info in self.summary_list:
            if layer_info.depth > self.formatting.max_depth:
                continue

            # create full hierarchy of current layer
            hierarchy = {}
            parent = layer_info.parent_info
            while parent is not None and parent.depth > 0:
                hierarchy[parent.depth] = parent
                parent = parent.parent_info

            # show hierarchy if it is not there already
            for d in range(1, layer_info.depth):
                if (
                    d not in current_hierarchy
                    or current_hierarchy[d].module is not hierarchy[d].module
                ):
                    new_str += self.layer_info_to_row(hierarchy[d])
                    current_hierarchy[d] = hierarchy[d]

            reached_max_depth = layer_info.depth == self.formatting.max_depth
            new_str += self.layer_info_to_row(layer_info, reached_max_depth)
            current_hierarchy[layer_info.depth] = layer_info

            # remove deeper hierarchy
            d = layer_info.depth + 1
            while d in current_hierarchy:
                current_hierarchy.pop(d)
                d += 1

        return new_str
