import sys
import warnings
from pathlib import Path
from typing import Iterator

import pytest
from _pytest.config.argparsing import Parser

from torchinfo import ModelStatistics
from torchinfo.formatting import HEADER_TITLES, ColumnSettings
from torchinfo.torchinfo import clear_cached_forward_pass


def pytest_addoption(parser: Parser) -> None:
    """This allows us to check for these params in sys.argv."""
    parser.addoption("--overwrite", action="store_true", default=False)
    parser.addoption("--no-output", action="store_true", default=False)


@pytest.fixture(autouse=True)
def verify_capsys(
    capsys: pytest.CaptureFixture[str], request: pytest.FixtureRequest
) -> Iterator[None]:
    yield
    clear_cached_forward_pass()
    if "--no-output" in sys.argv:
        return

    test_name = request.node.name.replace("test_", "")
    if sys.version_info < (3, 8) and test_name == "tmva_net_column_totals":
        warnings.warn(
            "sys.getsizeof can return different results on earlier Python versions."
        )
        return

    verify_output(capsys, f"tests/test_output/{test_name}.out")


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
        filepath.write_text(captured, encoding="utf-8")

    verify_output_str(captured, filename)


def verify_output_str(output: str, filename: str) -> None:
    with open(filename, encoding="utf-8") as output_file:
        expected = output_file.read()
    assert output == expected
    for category in (ColumnSettings.NUM_PARAMS, ColumnSettings.MULT_ADDS):
        assert_sum_column_totals_match(output, category)


def get_column_value_for_row(line: str, offset: int) -> int:
    """Helper function for getting the column totals."""
    col_value = line[offset:]
    end = col_value.find(" ")
    if end != -1:
        col_value = col_value[:end]
    if (
        not col_value
        or col_value in ("--", "(recursive)")
        or col_value.startswith(("└─", "├─"))
    ):
        return 0
    return int(col_value.replace(",", "").replace("(", "").replace(")", ""))


def assert_sum_column_totals_match(output: str, category: ColumnSettings) -> None:
    """Asserts that column totals match the total from the table summary."""
    lines = output.replace("=", "").split("\n\n")
    header_row = lines[0].strip()
    offset = header_row.find(HEADER_TITLES[category])
    if offset == -1:
        return
    layers = lines[1].split("\n")
    calculated_total = sum(get_column_value_for_row(line, offset) for line in layers)
    results = lines[2].split("\n")

    if category == ColumnSettings.NUM_PARAMS:
        total_params = results[0].split(":")[1].replace(",", "")
        assert calculated_total == int(total_params)
    elif category == ColumnSettings.MULT_ADDS:
        total_mult_adds = results[-1].split(":")[1].replace(",", "")
        assert float(
            f"{ModelStatistics.to_readable(calculated_total)[1]:0.2f}"
        ) == float(total_mult_adds)
