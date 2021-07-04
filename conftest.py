""" conftest.py """
import sys
import warnings
from pathlib import Path
from typing import Iterator

import pytest
from _pytest.config.argparsing import Parser


@pytest.fixture(autouse=True)
def verify_capsys(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> Iterator[None]:
    yield
    test_name = request.node.name.replace("test_", "")
    if test_name == "lstm" and sys.version_info < (3, 7):
        try:
            verify_output(capsys, f"tests/test_output/{test_name}.out")
        except AssertionError:
            warnings.warn(
                "Verbose output is not determininstic because dictionaries "
                "are not necessarily ordered in versions before Python 3.7."
            )
    else:
        verify_output(capsys, f"tests/test_output/{test_name}.out")


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
    filepath = Path(filename)
    if not captured and not filepath.exists():
        return
    if "--overwrite" in sys.argv:
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
