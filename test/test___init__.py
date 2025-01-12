"""Tests for jet/__init__.py."""

from math import factorial
from typing import Callable, Dict, Tuple

import pytest
from torch import Tensor, sin, tensor, zeros_like
from torch.autograd import grad

from jet import jet
from jet.operations import Primal, PrimalCoefficients, Value, ValueCoefficients


def rev_jet(
    f: Callable[[Primal], Value]
) -> Callable[
    [Tuple[Primal], PrimalCoefficients], Tuple[Tuple[Value], ValueCoefficients]
]:
    """Implement Taylor-mode arithmetic via nested reverse-mode autodiff.

    Args:
        f: Function to overload. Maps a tensor to another tensor.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """

    def jet_f(
        primal: Tuple[Primal], series: PrimalCoefficients
    ) -> Tuple[Tuple[Value], ValueCoefficients]:
        """Compute the function and its Taylor coefficients.

        Args:
            primal: Tuple containing the input tensor.
            series: Tuple containing the Taylor coefficients.

        Returns:
            Tuple containing the function value and its Taylor coefficients.
        """
        (x,) = primal
        (vs,) = series
        order = len(vs)

        def path(t: Tensor):
            x_t = x + sum(
                t**n / factorial(n) * v_n for n, v_n in enumerate(vs, start=1)
            )
            return f(x_t)

        t = tensor(0.0, requires_grad=True)
        (f_x,) = path(t)

        vs_out = [zeros_like(f_x).flatten() for _ in vs]

        for i, dnf_dt in enumerate(f_x.flatten()):
            for n in range(order):
                (dnf_dt,) = grad(dnf_dt, t, create_graph=n != order - 1)
                vs_out[n][i] = dnf_dt.detach()

        f_x = f_x.detach()
        vs_out = tuple(v.detach().reshape_as(f_x) for v in vs_out)

        return (f_x,), (vs_out,)

    return jet_f


def compare_jet_results(out1, out2):
    (value1,), (series1,) = out1
    (value2,), (series2,) = out2

    assert value1.allclose(value2)
    assert len(series1) == len(series2)
    for s1, s2 in zip(series1, series2):
        assert s1.allclose(s2)


def check_jet(f, x, vs):
    args = (x,), (vs,)

    rev_jet_f = rev_jet(f)
    rev_jet_out = rev_jet_f(*args)

    jet_f = jet(f)
    jet_out = jet_f(*args)

    compare_jet_results(jet_out, rev_jet_out)


CASES = [
    {
        "f": lambda x: sin(x),
        "x": lambda: tensor([0.1]),
        "vs": lambda: (tensor([0.2]), tensor([0.3])),
        "id": "sin-2",
    }
]


@pytest.mark.parametrize("config", CASES, ids=lambda c: c["id"])
def test_jet(config: Dict[str, Callable]):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary containing the function, input, and Taylor
            coefficients.
    """
    f = config["f"]
    x = config["x"]()
    vs = config["vs"]()
    check_jet(f, x, vs)
