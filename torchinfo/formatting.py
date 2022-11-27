from __future__ import annotations

import math
from typing import Any

from .enums import ColumnSettings, RowSettings, Units, Verbosity
from .layer_info import LayerInfo

HEADER_TITLES = {
    ColumnSettings.KERNEL_SIZE: "Kernel Shape",
    ColumnSettings.INPUT_SIZE: "Input Shape",
    ColumnSettings.OUTPUT_SIZE: "Output Shape",
    ColumnSettings.NUM_PARAMS: "Param #",
    ColumnSettings.PARAMS_PERCENT: "Param %",
    ColumnSettings.MULT_ADDS: "Mult-Adds",
    ColumnSettings.TRAINABLE: "Trainable",
}
CONVERSION_FACTORS = {
    Units.TERABYTES: 1e12,
    Units.GIGABYTES: 1e9,
    Units.MEGABYTES: 1e6,
    Units.NONE: 1,
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
        self.params_units = Units.NONE
        self.macs_units = Units.AUTO

        self.layer_name_width = 40
        self.ascii_only = RowSettings.ASCII_ONLY in self.row_settings
        self.show_var_name = RowSettings.VAR_NAMES in self.row_settings
        self.show_depth = RowSettings.DEPTH in self.row_settings
        self.hide_recursive_layers = (
            RowSettings.HIDE_RECURSIVE_LAYERS in self.row_settings
        )

    @staticmethod
    def str_(val: Any) -> str:
        return str(val) if val else "--"

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
        self, layer_info: LayerInfo, reached_max_depth: bool, total_params: int
    ) -> str:
        """Convert layer_info to string representation of a row."""
        values_for_row = {
            ColumnSettings.KERNEL_SIZE: self.str_(layer_info.kernel_size),
            ColumnSettings.INPUT_SIZE: self.str_(layer_info.input_size),
            ColumnSettings.OUTPUT_SIZE: self.str_(layer_info.output_size),
            ColumnSettings.NUM_PARAMS: layer_info.num_params_to_str(reached_max_depth),
            ColumnSettings.PARAMS_PERCENT: layer_info.params_percent(
                total_params, reached_max_depth
            ),
            ColumnSettings.MULT_ADDS: layer_info.macs_to_str(reached_max_depth),
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

    def layers_to_str(self, summary_list: list[LayerInfo], total_params: int) -> str:
        """
        Print each layer of the model using only current layer info.
        Container modules are already dealt with in add_missing_container_layers.
        """
        new_str = ""
        for layer_info in summary_list:
            if (
                layer_info.depth > self.max_depth
                or self.hide_recursive_layers
                and layer_info.is_recursive
            ):
                continue

            reached_max_depth = layer_info.depth == self.max_depth
            new_str += self.layer_info_to_row(
                layer_info, reached_max_depth, total_params
            )
        return new_str
