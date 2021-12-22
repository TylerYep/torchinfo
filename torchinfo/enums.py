""" constants.py """
from __future__ import annotations

from enum import Enum, unique


@unique
class RowSettings(Enum):
    """Enum containing all available row settings."""

    DEPTH = "depth"
    VAR_NAMES = "var_names"


@unique
class ColumnSettings(Enum):
    """Enum containing all available column settings."""

    KERNEL_SIZE = "kernel_size"
    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    NUM_PARAMS = "num_params"
    MULT_ADDS = "mult_adds"


@unique
class Verbosity(Enum):
    """Contains verbosity levels."""

    QUIET, DEFAULT, VERBOSE = 0, 1, 2
