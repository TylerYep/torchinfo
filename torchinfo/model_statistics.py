""" model_statistics.py """
from typing import Any, Iterable, List, Tuple, Union

import torch

from .formatting import FormattingOptions
from .layer_info import LayerInfo, prod

CORRECTED_INPUT_SIZE_TYPE = List[Union[Iterable[Any], torch.Size]]


class ModelStatistics:
    """Class for storing results of the summary."""

    def __init__(
        self,
        summary_list: List[LayerInfo],
        input_size: CORRECTED_INPUT_SIZE_TYPE,
        formatting: FormattingOptions,
    ) -> None:
        self.summary_list = summary_list
        self.input_size = input_size
        self.formatting = formatting
        self.total_params, self.trainable_params = 0, 0
        self.total_output, self.total_mult_adds = 0, 0
        self.total_input = sum(prod(sz) for sz in input_size) if input_size else 0

        for layer_info in summary_list:
            if layer_info.leaf_layer:
                self.total_mult_adds += layer_info.macs
                if layer_info.is_recursive:
                    continue
                self.total_params += layer_info.num_params
                if layer_info.trainable:
                    self.trainable_params += layer_info.num_params
                if layer_info.num_params > 0:
                    # x2 for gradients
                    self.total_output += 2 * prod(layer_info.output_size)

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
            summary_str += (
                "Total mult-adds ({}): {:0.2f}\n{}\n"
                "Input size (MB): {:0.2f}\n"
                "Forward/backward pass size (MB): {:0.2f}\n"
                "Params size (MB): {:0.2f}\n"
                "Estimated Total Size (MB): {:0.2f}\n".format(
                    *self.to_readable(self.total_mult_adds),
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
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num * 4 / 1e6

    @staticmethod
    def to_readable(num: int) -> Tuple[str, float]:
        """Converts a number to millions, billions, or trillions."""
        if num >= 1e12:
            return "T", num / 1e12
        if num >= 1e9:
            return "G", num / 1e9
        return "M", num / 1e6
