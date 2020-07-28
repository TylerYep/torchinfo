""" conftest.py """


def verify_output(capsys, filename):
    """ Utility function to ensure output matches file. """
    captured, _ = capsys.readouterr()
    with capsys.disabled():
        with open(filename, encoding="utf-8") as output_file:
            expected = output_file.read()
    assert captured == expected
