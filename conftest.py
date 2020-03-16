''' conftest.py '''
import pytest


@pytest.fixture(autouse=True)
def reset_const():
    pass


# def debug_spacing_issues(captured: str, expected: str):
#     ''' Helper method for debugging print differences. '''
#     print(len(captured), len(expected))
#     for i, captured_char in enumerate(captured):
#         if captured_char != expected[i]:
#             print("INCORRECT: ", i, captured_char, "vs", expected[i])
#         else:
#             print(" " * 10, i, captured_char, "vs", expected[i])
