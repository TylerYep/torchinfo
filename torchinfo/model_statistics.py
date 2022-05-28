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
                if layer_info.is_recursive:
                    continue
                self.total_params += layer_info.num_params
                self.total_param_bytes += layer_info.param_bytes
                self.trainable_params += layer_info.trainable_params
                if layer_info.num_params > 0:
                    # x2 for gradients
                    self.total_output_bytes += layer_info.output_bytes * 2
            else:
                if layer_info.is_recursive:
                    continue
                self.total_params += layer_info.leftover_params()
                self.trainable_params += layer_info.leftover_trainable_params()

        self.formatting.set_layer_name_width(summary_list)

    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{self.formatting.layers_to_str(self.summary_list)}{divider}\n"
            f"Total params: {self.total_params:,}\n"
            f"Trainable params: {self.trainable_params:,}\n"
            f"Non-trainable params: {self.total_params - self.trainable_params:,}\n"
        )
        if self.input_size:
            unit, macs = self.to_readable(self.total_mult_adds)
            input_size = self.to_megabytes(self.total_input)
            output_bytes = self.to_megabytes(self.total_output_bytes)
            param_bytes = self.to_megabytes(self.total_param_bytes)
            total_bytes = self.to_megabytes(
                self.total_input + self.total_output_bytes + self.total_param_bytes
            )
            summary_str += (
                f"Total mult-adds ({unit}): {macs:0.2f}\n{divider}\n"
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
    def to_readable(num: int) -> tuple[str, float]:
        """Converts a number to millions, billions, or trillions."""
        if num >= 1e12:
            return "T", num / 1e12
        if num >= 1e9:
            return "G", num / 1e9
        return "M", num / 1e6
