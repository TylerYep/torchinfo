from __future__ import annotations

from typing import Any

from .enums import Units
from .formatting import CONVERSION_FACTORS, FormattingOptions
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

        # TODO: Figure out why the below functions using max() are ever 0
        # (they should always be non-negative), and remove the call to max().
        # Investigation: https://github.com/TylerYep/torchinfo/pull/195
        for layer_info in summary_list:
            if layer_info.is_leaf_layer:
                self.total_mult_adds += layer_info.macs
                if layer_info.num_params > 0:
                    # x2 for gradients
                    self.total_output_bytes += layer_info.output_bytes * 2
                if layer_info.is_recursive:
                    continue
                self.total_params += max(layer_info.num_params, 0)
                self.total_param_bytes += layer_info.param_bytes
                self.trainable_params += max(layer_info.trainable_params, 0)
            else:
                if layer_info.is_recursive:
                    continue
                leftover_params = layer_info.leftover_params()
                leftover_trainable_params = layer_info.leftover_trainable_params()
                self.total_params += max(leftover_params, 0)
                self.trainable_params += max(leftover_trainable_params, 0)
        self.formatting.set_layer_name_width(summary_list)

    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        total_params = ModelStatistics.format_output_num(
            self.total_params, self.formatting.params_units
        )
        trainable_params = ModelStatistics.format_output_num(
            self.trainable_params, self.formatting.params_units
        )
        non_trainable_params = ModelStatistics.format_output_num(
            self.total_params - self.trainable_params, self.formatting.params_units
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
                self.total_mult_adds, self.formatting.macs_units
            )
            input_size = self.to_megabytes(self.total_input)
            output_bytes = self.to_megabytes(self.total_output_bytes)
            param_bytes = self.to_megabytes(self.total_param_bytes)
            total_bytes = self.to_megabytes(
                self.total_input + self.total_output_bytes + self.total_param_bytes
            )
            summary_str += (
                f"Total mult-adds{macs}\n{divider}\n"
                f"Input size (MB): {input_size:0.2f}\n"
                f"Forward/backward pass size (MB): {output_bytes:0.2f}\n"
                f"Params size (MB): {param_bytes:0.2f}\n"
                f"Estimated Total Size (MB): {total_bytes:0.2f}\n"
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
    def to_readable(num: int, units: Units = Units.AUTO) -> tuple[Units, float]:
        """Converts a number to millions, billions, or trillions."""
        if units == Units.AUTO:
            if num >= 1e12:
                return Units.TERABYTES, num / 1e12
            if num >= 1e9:
                return Units.GIGABYTES, num / 1e9
            return Units.MEGABYTES, num / 1e6
        return units, num / CONVERSION_FACTORS[units]

    @staticmethod
    def format_output_num(num: int, units: Units) -> str:
        units_used, converted_num = ModelStatistics.to_readable(num, units)
        if converted_num.is_integer():
            converted_num = int(converted_num)
        units_display = "" if units_used == Units.NONE else f" ({units_used.value})"
        fmt = "d" if isinstance(converted_num, int) else ".2f"
        return f"{units_display}: {converted_num:,{fmt}}"
