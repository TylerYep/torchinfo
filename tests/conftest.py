import sys
import warnings
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from torchinfo import ModelStatistics
from torchinfo.enums import ColumnSettings
from torchinfo.formatting import HEADER_TITLES
from torchinfo.torchinfo import clear_cached_forward_pass


def pytest_addoption(parser: pytest.Parser) -> None:
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
            "sys.getsizeof can return different results on earlier Python versions.",
            stacklevel=2,
        )
        return

    if test_name == "input_size_half_precision":
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
    expected = Path(filename).read_text(encoding="utf-8")
    # Verify the input size has the same unit
    output_input_size, output_input_unit = get_input_size_and_unit(output)
    expected_input_size, expected_input_unit = get_input_size_and_unit(expected)
    assert output_input_unit == expected_input_unit

    # Sometime it does not have the same exact value, depending on torch version.
    # We assume the variation cannot be too large.
    if output_input_size != 0:
        assert abs(output_input_size - expected_input_size)/output_input_size < 1e-2

    if output_input_size != expected_input_size:
        # In case of a difference, replace the expected input size.
        expected = replace_input_size(expected, expected_input_unit, expected_input_size, output_input_size)
    assert output == expected
    for category in (ColumnSettings.NUM_PARAMS, ColumnSettings.MULT_ADDS):
        assert_sum_column_totals_match(output, category)

def replace_input_size(output: str, unit: str, old_value: float, new_value: float) -> str:
    return output.replace(f"Input size {unit}: {old_value:.2f}", f"Input size {unit}: {new_value:.2f}")

def get_input_size_and_unit(output_str: str) -> Tuple[float, str]:
    input_size = float(output_str.split('Input size')[1].split(':')[1].split('\n')[0].strip())
    input_unit = output_str.split('Input size')[1].split(':')[0].strip()
    return input_size, input_unit

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
    calculated_total = float(sum(get_column_value_for_row(line, offset) for line in layers))
    results = lines[2].split("\n")

    if category == ColumnSettings.NUM_PARAMS:
        total_params = results[0].split(":")[1].replace(",", "")
        splitted_results = results[0].split('(')
        if len(splitted_results) > 1:
            units = splitted_results[1][0]
            if units == 'T':
                calculated_total /= 1e12
            elif units == 'G':
                calculated_total /= 1e9
            elif units == 'M':
                calculated_total /= 1e6
            elif units == 'k':
                calculated_total /= 1e3
        assert calculated_total == float(total_params)
    elif category == ColumnSettings.MULT_ADDS:
        total_mult_adds = results[-1].split(":")[1].replace(",", "")
        assert float(
            f"{ModelStatistics.to_readable(calculated_total)[1]:0.2f}"
        ) == float(total_mult_adds)
