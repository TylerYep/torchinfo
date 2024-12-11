from __future__ import annotations

from enum import Enum, IntEnum, unique


@unique
class Mode(str, Enum):
    """Enum containing all model modes."""

    __slots__ = ()

    TRAIN = "train"
    EVAL = "eval"
    SAME = "same"


@unique
class RowSettings(str, Enum):
    """Enum containing all available row settings."""

    __slots__ = ()

    DEPTH = "depth"
    VAR_NAMES = "var_names"
    ASCII_ONLY = "ascii_only"
    HIDE_RECURSIVE_LAYERS = "hide_recursive_layers"


@unique
class ColumnSettings(str, Enum):
    """Enum containing all available column settings."""

    __slots__ = ()

    KERNEL_SIZE = "kernel_size"
    GROUPS = "groups"
    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    NUM_PARAMS = "num_params"
    PARAMS_PERCENT = "params_percent"
    MULT_ADDS = "mult_adds"
    TRAINABLE = "trainable"


@unique
class Units(str, Enum):
    """Enum containing all available bytes units."""

    __slots__ = ()

    AUTO = "auto"
    MEGABYTES = "M"
    GIGABYTES = "G"
    TERABYTES = "T"
    NONE = ""


@unique
class Verbosity(IntEnum):
    """Contains verbosity levels."""

    QUIET, DEFAULT, VERBOSE = 0, 1, 2
