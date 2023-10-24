# pylint: skip-file

from __future__ import annotations

import pytest

from torchinfo.formatting import make_layer_tree


class MockLayer:
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def __repr__(self) -> str:
        return f"L({self.depth})"


L = MockLayer


@pytest.mark.no_verify_capsys
@pytest.mark.parametrize(
    "layers, expected",
    [
        ([], []),
        ([a := L(0)], [a]),
        ([a := L(0), b := L(0)], [a, b]),
        ([a := L(0), b := L(1)], [a, [b]]),
        ([a := L(0), b := L(0), c := L(0)], [a, b, c]),
        ([a := L(0), b := L(1), c := L(0)], [a, [b], c]),
        ([a := L(0), b := L(1), c := L(1)], [a, [b, c]]),
        ([a := L(0), b := L(1), c := L(2)], [a, [b, [c]]]),
        (
            # If this ever happens, there's probably a bug elsewhere, but
            # we still want to format things as best as possible.
            [a := L(1), b := L(0)],
            [[a], b],
        ),
    ],
)
def test_make_layer_tree(layers, expected):  # type: ignore [no-untyped-def]
    assert make_layer_tree(layers) == expected
