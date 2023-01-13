import torch


def torchversion_at_least(version: str) -> bool:
    """
    Returns True if the installed version of torch is at least the given version.
    For example, if "1.13.1" is installed, `torchversion_at_least("1.8")` would
    yield `True`, but if "1.7.1" is installed, torchversion_at_least("1.8")` would
    yield `False`.
    """
    version_installed = torch.__version__.split(".")
    version_given = version.split(".")

    for num_installed, num_given in zip(version_installed, version_given):
        if int(num_given) < int(num_installed):
            return True
        if int(num_given) > int(num_installed):
            return False

    if len(version_given) > len(
        version_installed
    ):  # e.g. "1.7.1" installed, "1.7" given
        return False

    return True
