""" formatting.py """
import math
from enum import Enum, unique
from typing import Dict, Iterable, List

from .layer_info import LayerInfo

ALL_ROW_SETTINGS = ("depth", "var_names")
HEADER_TITLES = {
    "kernel_size": "Kernel Shape",
    "input_size": "Input Shape",
    "output_size": "Output Shape",
    "num_params": "Param #",
    "mult_adds": "Mult-Adds",
}


@unique
class Verbosity(Enum):
    """Contains verbosity levels."""

    QUIET, DEFAULT, VERBOSE = 0, 1, 2


class FormattingOptions:
    """Class that holds information about formatting the table output."""

    def __init__(
        self,
        max_depth: int,
        verbose: int,
        col_names: Iterable[str],
        col_width: int,
        row_settings: Iterable[str],
    ):
        self.max_depth = max_depth
        self.verbose = verbose
        self.col_names = col_names
        self.col_width = col_width
        self.row_settings = row_settings

        self.layer_name_width = 40
        self.show_var_name = "var_names" in self.row_settings
        self.show_depth = "depth" in self.row_settings

    @staticmethod
    def get_start_str(depth: int) -> str:
        return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"

    def set_layer_name_width(
        self, summary_list: List[LayerInfo], align_val: int = 5
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

    def format_row(self, layer_name: str, row_values: Dict[str, str]) -> str:
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

    def layer_info_to_row(self, layer_info: LayerInfo, reached_max_depth: bool) -> str:
        """Convert layer_info to string representation of a row."""
        row_values = {
            "kernel_size": (
                str(layer_info.kernel_size) if layer_info.kernel_size else "--"
            ),
            "input_size": str(layer_info.input_size),
            "output_size": str(layer_info.output_size),
            "num_params": layer_info.num_params_to_str(reached_max_depth),
            "mult_adds": layer_info.macs_to_str(reached_max_depth),
        }
        start_str = self.get_start_str(layer_info.depth)
        layer_name = layer_info.get_layer_name(self.show_var_name, self.show_depth)
        new_line = self.format_row(f"{start_str}{layer_name}", row_values)

        if self.verbose == Verbosity.VERBOSE.value:
            for inner_name, inner_shape in layer_info.inner_layers.items():
                prefix = self.get_start_str(layer_info.depth + 1)
                extra_row_values = {"kernel_size": str(inner_shape)}
                new_line += self.format_row(f"{prefix}{inner_name}", extra_row_values)
        return new_line

    def layers_to_str(self, summary_list: List[LayerInfo]) -> str:
        """Print each layer of the model using a fancy branching diagram."""
        new_str = ""
        current_hierarchy: Dict[int, LayerInfo] = {}
        for layer_info in summary_list:
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
                        hierarchy[d], reached_max_depth=False
                    )
                    current_hierarchy[d] = hierarchy[d]

            reached_max_depth = layer_info.depth == self.max_depth
            new_str += self.layer_info_to_row(layer_info, reached_max_depth)
            current_hierarchy[layer_info.depth] = layer_info

            # remove deeper hierarchy
            d = layer_info.depth + 1
            while d in current_hierarchy:
                current_hierarchy.pop(d)
                d += 1

        return new_str
