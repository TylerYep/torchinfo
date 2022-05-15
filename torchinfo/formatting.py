from __future__ import annotations

import math
from typing import Any

from .enums import ColumnSettings, RowSettings, Verbosity
from .layer_info import LayerInfo

HEADER_TITLES = {
    ColumnSettings.KERNEL_SIZE: "Kernel Shape",
    ColumnSettings.INPUT_SIZE: "Input Shape",
    ColumnSettings.OUTPUT_SIZE: "Output Shape",
    ColumnSettings.NUM_PARAMS: "Param #",
    ColumnSettings.MULT_ADDS: "Mult-Adds",
    ColumnSettings.TRAINABLE: "Trainable",
}


class FormattingOptions:
    """Class that holds information about formatting the table output."""

    def __init__(
        self,
        max_depth: int,
        verbose: int,
        col_names: tuple[ColumnSettings, ...],
        col_width: int,
        row_settings: set[RowSettings],
    ) -> None:
        self.max_depth = max_depth
        self.verbose = verbose
        self.col_names = col_names
        self.col_width = col_width
        self.row_settings = row_settings

        self.layer_name_width = 40
        self.ascii_only = RowSettings.ASCII_ONLY in self.row_settings
        self.show_var_name = RowSettings.VAR_NAMES in self.row_settings
        self.show_depth = RowSettings.DEPTH in self.row_settings

    @staticmethod
    def str_(val: Any) -> str:
        return str(val) if val else "--"

    @staticmethod
    def get_children_layers(
        summary_list: list[LayerInfo], layer_info: LayerInfo, index: int
    ) -> list[LayerInfo]:
        """Fetches all of the children of a given layer."""
        num_children = 0
        for layer in summary_list[index + 1 :]:
            if layer.depth <= layer_info.depth:
                break
            num_children += 1
        return summary_list[index + 1 : index + 1 + num_children]

    def get_start_str(self, depth: int) -> str:
        """This function should handle all ascii/non-ascii-related characters."""
        if depth == 0:
            return ""
        if depth == 1:
            return "+ " if self.ascii_only else "├─"
        return ("|    " if self.ascii_only else "│    ") * (depth - 1) + (
            "+ " if self.ascii_only else "└─"
        )

    def set_layer_name_width(
        self, summary_list: list[LayerInfo], align_val: int = 5
    ) -> None:
        """
        Set layer name width by taking the longest line length and rounding up to
        the nearest multiple of align_val.
        """
        max_length = 0
        for info in summary_list:
            depth_indent = info.depth * align_val + 1
            layer_title = info.get_layer_name(self.show_var_name, self.show_depth)
            max_length = max(max_length, len(layer_title) + depth_indent)
        if max_length >= self.layer_name_width:
            self.layer_name_width = math.ceil(max_length / align_val) * align_val

    def get_total_width(self) -> int:
        """Calculate the total width of all lines in the table."""
        return len(tuple(self.col_names)) * self.col_width + self.layer_name_width

    def format_row(self, layer_name: str, row_values: dict[ColumnSettings, str]) -> str:
        """Get the string representation of a single layer of the model."""
        info_to_use = [row_values.get(row_type, "") for row_type in self.col_names]
        new_line = f"{layer_name:<{self.layer_name_width}} "
        for info in info_to_use:
            new_line += f"{info:<{self.col_width}} "
        return new_line.rstrip() + "\n"

    def header_row(self) -> str:
        layer_header = ""
        if self.show_var_name:
            layer_header += " (var_name)"
        if self.show_depth:
            layer_header += ":depth-idx"
        return self.format_row(f"Layer (type{layer_header})", HEADER_TITLES)

    def layer_info_to_row(
        self,
        layer_info: LayerInfo,
        reached_max_depth: bool,
        children_layers: list[LayerInfo],
    ) -> str:
        """Convert layer_info to string representation of a row."""
        values_for_row = {
            ColumnSettings.KERNEL_SIZE: self.str_(layer_info.kernel_size),
            ColumnSettings.INPUT_SIZE: self.str_(layer_info.input_size),
            ColumnSettings.OUTPUT_SIZE: self.str_(layer_info.output_size),
            ColumnSettings.NUM_PARAMS: layer_info.num_params_to_str(reached_max_depth),
            ColumnSettings.MULT_ADDS: layer_info.macs_to_str(
                reached_max_depth, children_layers
            ),
            ColumnSettings.TRAINABLE: self.str_(layer_info.trainable),
        }
        start_str = self.get_start_str(layer_info.depth)
        layer_name = layer_info.get_layer_name(self.show_var_name, self.show_depth)
        new_line = self.format_row(f"{start_str}{layer_name}", values_for_row)

        if self.verbose == Verbosity.VERBOSE:
            for inner_name, inner_layer_info in layer_info.inner_layers.items():
                prefix = self.get_start_str(layer_info.depth + 1)
                new_line += self.format_row(f"{prefix}{inner_name}", inner_layer_info)
        return new_line

    def layers_to_str(self, summary_list: list[LayerInfo]) -> str:
        """
        Print each layer of the model using a fancy branching diagram.
        This is necessary to handle Container modules that don't have explicit parents.
        """
        new_str = ""
        current_hierarchy: dict[int, LayerInfo] = {}
        for i, layer_info in enumerate(summary_list):
            if layer_info.depth > self.max_depth:
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
                    new_str += self.layer_info_to_row(
                        hierarchy[d], reached_max_depth=False, children_layers=[]
                    )
                    current_hierarchy[d] = hierarchy[d]

            reached_max_depth = layer_info.depth == self.max_depth
            children_layers = self.get_children_layers(summary_list, layer_info, i)
            new_str += self.layer_info_to_row(
                layer_info, reached_max_depth, children_layers
            )
            current_hierarchy[layer_info.depth] = layer_info

            # remove deeper hierarchy
            d = layer_info.depth + 1
            while d in current_hierarchy:
                current_hierarchy.pop(d)
                d += 1

        return new_str
