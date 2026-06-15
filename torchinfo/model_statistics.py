from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .enums import Units
from .formatting import CONVERSION_FACTORS, FormattingOptions

if TYPE_CHECKING:
    from .layer_info import LayerInfo


class ModelStatistics:
    """Class for storing results of the summary."""

    def __init__(
        self,
        summary_list: list[LayerInfo],
        input_size: Any,
        total_input_size: int,
        formatting: FormattingOptions,
    ) -> None:
        self.summary_list = summary_list
        self.input_size = input_size
        self.formatting = formatting
        self.total_input = total_input_size
        self.total_mult_adds = 0
        self.total_params, self.trainable_params = 0, 0
        self.total_param_bytes, self.total_output_bytes = 0, 0

        for layer_info in summary_list:
            if layer_info.is_leaf_layer:
                self.total_mult_adds += layer_info.macs
                if layer_info.num_params > 0:
                    # x2 for gradients
                    self.total_output_bytes += layer_info.output_bytes * 2

        # Parameter totals are taken from the root module(s). A module's
        # named_parameters() already deduplicates tensors shared across modules
        # (weight tying, e.g. tied embeddings / lm_head) and counts parameters of
        # submodules that weren't executed in this forward pass -- so this matches
        # `sum(p.numel() for p in model.parameters())`. Summing the per-row counts
        # instead would double-count tied weights (#322/#377).
        for layer_info in summary_list:
            if layer_info.parent_info is None:
                self.total_params += layer_info.num_params
                self.trainable_params += layer_info.trainable_params
                self.total_param_bytes += layer_info.param_bytes

        # Mark a module as "(recursive)" when every parameter it owns directly was
        # already counted by an earlier row (a fully shared/tied module), so the
        # displayed per-row counts still sum to the deduplicated total.
        seen_param_ids: set[int] = set()
        for layer_info in summary_list:
            direct_params = layer_info.get_direct_param_ids()
            contributed_new = False
            for param_id, _, _, _ in direct_params:
                if param_id not in seen_param_ids:
                    seen_param_ids.add(param_id)
                    contributed_new = True
            if not contributed_new and any(count for _, count, _, _ in direct_params):
                layer_info.is_recursive = True
        self.formatting.set_layer_name_width(summary_list)

    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        total_params = ModelStatistics.format_output_num(
            self.total_params, self.formatting.params_count_units, False
        )
        trainable_params = ModelStatistics.format_output_num(
            self.trainable_params, self.formatting.params_count_units, False
        )
        non_trainable_params = ModelStatistics.format_output_num(
            self.total_params - self.trainable_params,
            self.formatting.params_count_units,
            False,
        )
        all_layers = self.formatting.layers_to_str(self.summary_list, self.total_params)
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{all_layers}{divider}\n"
            f"Total params{total_params}\n"
            f"Trainable params{trainable_params}\n"
            f"Non-trainable params{non_trainable_params}\n"
        )
        if self.input_size:
            macs = ModelStatistics.format_output_num(
                self.total_mult_adds, self.formatting.macs_units, False
            )
            input_size = ModelStatistics.format_output_num(
                self.total_input, self.formatting.params_size_units, True
            )
            output_bytes = ModelStatistics.format_output_num(
                self.total_output_bytes, self.formatting.params_size_units, True
            )
            param_bytes = ModelStatistics.format_output_num(
                self.total_param_bytes, self.formatting.params_size_units, True
            )
            total_bytes = ModelStatistics.format_output_num(
                self.total_input + self.total_output_bytes + self.total_param_bytes,
                self.formatting.params_size_units,
                True,
            )
            summary_str += (
                f"Total mult-adds{macs}\n{divider}\n"
                f"Input size{input_size}\n"
                f"Forward/backward pass size{output_bytes}\n"
                f"Params size{param_bytes}\n"
                f"Estimated Total Size{total_bytes}\n"
            )
        summary_str += divider
        return summary_str

    @staticmethod
    def float_to_megabytes(num: int) -> float:
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num * 4 / 1e6

    @staticmethod
    def to_megabytes(num: int) -> float:
        """Converts bytes to megabytes."""
        return num / 1e6

    @staticmethod
    def to_readable(num: float, units: Units = Units.AUTO) -> tuple[Units, float]:
        """Converts a number to millions, billions, or trillions."""
        if units == Units.AUTO:
            if num >= 1e12:
                return Units.TERABYTES, num / 1e12
            if num >= 1e9:
                return Units.GIGABYTES, num / 1e9
            if num >= 1e6:
                return Units.MEGABYTES, num / 1e6
            if num >= 1e3:
                return Units.KILOBYTES, num / 1e3
            return Units.NONE, num
        return units, num / CONVERSION_FACTORS[units]

    @staticmethod
    def format_output_num(num: int, units: Units, is_bytes: bool) -> str:
        units_used, converted_num = ModelStatistics.to_readable(num, units)
        if isinstance(converted_num, float) and converted_num.is_integer():
            converted_num = int(converted_num)
        units_display = (
            ""
            if units_used == Units.NONE
            else f" ({units_used.value}{'B' if is_bytes else ''})"
        )
        fmt = "d" if isinstance(converted_num, int) else ".2f"
        return f"{units_display}: {converted_num:,{fmt}}"
