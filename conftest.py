""" conftest.py """
from pathlib import Path

import pytest


def verify_output(
    capsys: pytest.CaptureFixture[str], filename: str, overwrite_file: bool = False
) -> None:
    """
    Utility function to ensure output matches file.
    If you are writing new tests, set overwrite_file=True to generate the
    new test_output file.
    """
    captured, _ = capsys.readouterr()
    if overwrite_file:
        filepath = Path(filename)
        filepath.parent.mkdir(exist_ok=True)
        filepath.touch(exist_ok=False)
        filepath.write_text(captured)

    verify_output_str(captured, filename)


def verify_output_str(output: str, filename: str) -> None:
    with open(filename, encoding="utf-8") as output_file:
        expected = output_file.read()
    if output != expected:
        print(f"Got:\n{output}\nExpected:\n{expected}")
    assert output == expected
