from __future__ import annotations

import math
from typing import Any, Iterable, List, Union

from .enums import ColumnSettings, RowSettings, Units, Verbosity
from .layer_info import LayerInfo, NamedParamInfo

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
LAYER_TYPES = Union[LayerInfo, NamedParamInfo]
LAYER_TREE_TYPE = List[Union[LAYER_TYPES, "LAYER_TREE_TYPE"]]


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
        layer_info: LAYER_TYPES,
        reached_max_depth: bool,
        total_params: int,
        vertical_bars: tuple[bool, ...],
    ) -> str:
        """Convert layer_info to string representation of a row."""
        tree_prefix = get_tree_prefix(vertical_bars, ascii_only=self.ascii_only)

        if isinstance(layer_info, LayerInfo):
            layer_name = layer_info.get_layer_name(self.show_var_name, self.show_depth)
            values_for_row = {
                ColumnSettings.KERNEL_SIZE: self.str_(layer_info.kernel_size),
                ColumnSettings.INPUT_SIZE: self.str_(layer_info.input_size),
                ColumnSettings.OUTPUT_SIZE: self.str_(layer_info.output_size),
                ColumnSettings.NUM_PARAMS: layer_info.num_params_to_str(
                    reached_max_depth
                ),
                ColumnSettings.PARAMS_PERCENT: layer_info.params_percent(
                    total_params, reached_max_depth
                ),
                ColumnSettings.MULT_ADDS: layer_info.macs_to_str(reached_max_depth),
                ColumnSettings.TRAINABLE: self.str_(layer_info.trainable),
            }

        elif isinstance(layer_info, NamedParamInfo):
            layer_name = layer_info.name
            tree_icon = get_tree_icon(
                horizontal_bar=True,
                vertical_bar=not layer_info.is_last,
                ascii_only=self.ascii_only,
            )
            values_for_row = {
                ColumnSettings.KERNEL_SIZE: self.str_(layer_info.kernel_size),
                ColumnSettings.NUM_PARAMS: f"{tree_icon}{layer_info.num_params:,}",
            }

        return self.format_row(f"{tree_prefix}{layer_name}", values_for_row)

    def layers_to_str(self, summary_list: list[LayerInfo], total_params: int) -> str:
        """
        Print each layer of the model using only current layer info.
        Container modules are already dealt with in add_missing_container_layers.
        """
        new_str = ""

        # Any layers that are going to be revealed/hidden have to be handled
        # before we calculate the layer tree, otherwise the hierarchy won't
        # be rendered correctly.

        visible_layers: list[LAYER_TYPES] = []

        for layer_info in summary_list:
            if layer_info.depth > self.max_depth:
                continue
            if self.hide_recursive_layers and layer_info.is_recursive:
                continue

            visible_layers.append(layer_info)

            if self.verbose == Verbosity.VERBOSE:
                visible_layers += layer_info.inner_layers

        layer_tree = make_layer_tree(visible_layers)

        for layer_info_, vertical_bars in iter_layer_tree(layer_tree):
            reached_max_depth = layer_info_.depth == self.max_depth
            new_str += self.layer_info_to_row(
                layer_info_, reached_max_depth, total_params, vertical_bars
            )
        return new_str


def make_layer_tree(summary_list: list[LAYER_TYPES]) -> LAYER_TREE_TYPE:
    tree, _ = _make_layer_tree(summary_list, 0, 0)
    return tree


def _make_layer_tree(
    summary_list: list[LAYER_TYPES], curr_depth: int, cursor: int
) -> tuple[LAYER_TREE_TYPE, int]:
    tree: LAYER_TREE_TYPE = []

    while cursor < len(summary_list):
        layer_info = summary_list[cursor]
        depth = layer_info.depth

        if depth < curr_depth:
            break

        if depth == curr_depth:
            tree.append(layer_info)
            cursor += 1

        if depth > curr_depth:
            assert depth == curr_depth + 1
            subtree, cursor = _make_layer_tree(summary_list, depth, cursor)
            tree.append(subtree)

    return tree, cursor


def iter_layer_tree(
    layer_tree: LAYER_TREE_TYPE,
) -> Iterable[tuple[LAYER_TYPES, tuple[bool, ...]]]:
    for layer_info, vertical_bars in _iter_layer_tree(layer_tree, ()):
        yield layer_info, vertical_bars[1:]


def _iter_layer_tree(
    layer_tree: LAYER_TREE_TYPE, vertical_bars: tuple[bool, ...]
) -> Iterable[tuple[LAYER_TYPES, tuple[bool, ...]]]:
    for i, item in enumerate(layer_tree):
        if isinstance(item, (LayerInfo, NamedParamInfo)):
            stop_vertical_bars = i

    for i, item in enumerate(layer_tree):
        if isinstance(item, (LayerInfo, NamedParamInfo)):
            yield item, vertical_bars + (i != stop_vertical_bars,)

        else:
            yield from _iter_layer_tree(item, vertical_bars + (i < stop_vertical_bars,))


def get_tree_prefix(
    vertical_bars: tuple[bool, ...], *, ascii_only: bool, indent: int = 4
) -> str:
    i_innermost = len(vertical_bars) - 1
    tree_icons = [
        get_tree_icon(
            horizontal_bar=(i == i_innermost),
            vertical_bar=vertical_bar,
            ascii_only=ascii_only,
        )
        for i, vertical_bar in enumerate(vertical_bars)
    ]
    indent_str = " " * indent
    return indent_str.join(tree_icons)


def get_tree_icon(*, horizontal_bar: bool, vertical_bar: bool, ascii_only: bool) -> str:
    icons = [
        [
            [" ", "│"],  # ascii_only=0, horizontal_bar=0
            ["└─", "├─"],  # ascii_only=0, horizontal_bar=1
        ],
        [
            [" ", "|"],  # ascii_only=1, horizontal_bar=0
            ["'--", "|--"],  # ascii_only=1, horizontal_bar=1
        ],
    ]
    return icons[ascii_only][horizontal_bar][vertical_bar]
