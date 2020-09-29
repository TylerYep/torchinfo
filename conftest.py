""" conftest.py """
from _pytest.capture import CaptureFixture


def verify_output(capsys: CaptureFixture, filename: str) -> None:
    """ Utility function to ensure output matches file. """
    captured, _ = capsys.readouterr()
    with capsys.disabled():
        with open(filename, encoding="utf-8") as output_file:
            expected = output_file.read()
    assert captured == expected
