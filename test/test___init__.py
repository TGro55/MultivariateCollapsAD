"""Tests for jet/__init__.py."""

from typing import Callable, Dict

import pytest
from torch import sin, tensor

from jet import jet, rev_jet
from jet.operations import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients


def compare_jet_results(out1: ValueAndCoefficients, out2: ValueAndCoefficients):
    value1, series1 = out1
    value2, series2 = out2

    assert value1.allclose(value2)
    assert len(series1) == len(series2)
    for s1, s2 in zip(series1, series2):
        assert s1.allclose(s2)


def check_jet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients):
    rev_jet_f = rev_jet(f)
    rev_jet_out = rev_jet_f(arg)

    jet_f = jet(f, verbose=True)
    jet_out = jet_f(arg)

    compare_jet_results(jet_out, rev_jet_out)


CASES = [
    {
        "f": lambda x: sin(x),
        "primal": lambda: tensor([0.1]),
        "coefficients": lambda: (tensor([0.2]), tensor([0.3])),
        "id": "sin-2",
    },
    {
        "f": lambda x: sin(sin(x)),
        "primal": lambda: tensor([0.1, 0.15]),
        "coefficients": lambda: (tensor([0.2, 0.25]), tensor([0.3, 0.35])),
        "id": "sinsin-2",
    },
]


@pytest.mark.parametrize("config", CASES, ids=lambda c: c["id"])
def test_jet(config: Dict[str, Callable]):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary containing the function, input, and Taylor
            coefficients.
    """
    f = config["f"]
    x = config["primal"]()
    vs = config["coefficients"]()
    check_jet(f, (x, vs))
