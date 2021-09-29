""" torchinfo """
from .formatting import ALL_COLUMN_SETTINGS, ALL_ROW_SETTINGS
from .model_statistics import ModelStatistics
from .torchinfo import summary

__all__ = ("ModelStatistics", "summary", "ALL_COLUMN_SETTINGS", "ALL_ROW_SETTINGS")

import pkg_resources as _pkg_resources

try:
    __version__ = _pkg_resources.get_distribution('torchinfo').version
except Exception:
    __version__ = 'unknown'
