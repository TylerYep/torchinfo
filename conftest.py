""" conftest.py """
import pytest


def verify_output(capsys: pytest.CaptureFixture[str], filename: str) -> None:
    """ Utility function to ensure output matches file. """
    captured, _ = capsys.readouterr()
    with capsys.disabled(), open(filename, encoding="utf-8") as output_file:
        expected = output_file.read()
    assert captured == expected
