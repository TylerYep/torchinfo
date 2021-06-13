""" conftest.py """
import sys
from pathlib import Path

import pytest
from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    """This allows us to check for this param in sys.argv."""
    parser.addoption("--overwrite", type=bool)


def verify_output(capsys: pytest.CaptureFixture[str], filename: str) -> None:
    """
    Utility function to ensure output matches file.
    If you are writing new tests, set overwrite_file=True to generate the
    new test_output file.
    """
    captured, _ = capsys.readouterr()
    if "--overwrite" in sys.argv:
        filepath = Path(filename)
        filepath.parent.mkdir(exist_ok=True)
        filepath.touch(exist_ok=True)
        filepath.write_text(captured)

    verify_output_str(captured, filename)


def verify_output_str(output: str, filename: str) -> None:
    with open(filename, encoding="utf-8") as output_file:
        expected = output_file.read()
    if output != expected:
        print(f"Expected:\n{expected}\nGot:\n{output}")
    assert output == expected
