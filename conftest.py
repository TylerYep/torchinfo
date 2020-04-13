""" conftest.py """
import pytest


@pytest.fixture(autouse=True)
def reset_const():
    pass
