from __future__ import annotations

from enum import Enum, IntEnum, unique


@unique
class Mode(str, Enum):
    """Enum containing all model modes."""

    TRAIN = "train"
    EVAL = "eval"


@unique
class RowSettings(str, Enum):
    """Enum containing all available row settings."""

    DEPTH = "depth"
    VAR_NAMES = "var_names"
    ASCII_ONLY = "ascii_only"


@unique
class ColumnSettings(str, Enum):
    """Enum containing all available column settings."""

    KERNEL_SIZE = "kernel_size"
    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    NUM_PARAMS = "num_params"
    MULT_ADDS = "mult_adds"
    TRAINABLE = "trainable"


@unique
class Verbosity(IntEnum):
    """Contains verbosity levels."""

    QUIET, DEFAULT, VERBOSE = 0, 1, 2
