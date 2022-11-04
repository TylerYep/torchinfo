from __future__ import annotations

from typing import Any

from .formatting import FormattingOptions
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
                if layer_info.is_recursive:
                    continue
                self.total_params += layer_info.num_params
                self.total_param_bytes += layer_info.param_bytes
                self.trainable_params += layer_info.trainable_params
            else:
                if layer_info.is_recursive:
                    continue
                self.total_params += layer_info.leftover_params()
                self.trainable_params += layer_info.leftover_trainable_params()

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
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{self.formatting.layers_to_str(self.summary_list)}{divider}\n"
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
    def to_readable(num: int, units: str = "auto") -> tuple[str, float]:
        """Converts a number to millions, billions, or trillions."""
        if units == "auto":
            if num >= 1e12:
                return "T", num / 1e12
            if num >= 1e9:
                return "G", num / 1e9
            return "M", num / 1e6
        divisor = {"T": 1e12, "G": 1e9, "M": 1e6, "": 1.0}[units]
        num_conv = num / divisor
        return units, num_conv

    @staticmethod
    def format_output_num(num: int, units: str) -> str:
        units_conv, num_conv = ModelStatistics.to_readable(num, units)
        if num_conv.is_integer():
            num_conv = int(num_conv)
        if units_conv != "":
            units_conv = f" ({units_conv})"
        fmt = "d" if isinstance(num_conv, int) else ".2f"
        output = f"{units_conv}: {num_conv:,{fmt}}"
        return output
