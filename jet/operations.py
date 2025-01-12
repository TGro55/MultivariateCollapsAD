"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Tuple

from torch import Tensor, cos, sin

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
PrimalCoefficients = Tuple[Tuple[Primal, ...]]
Value = Tensor
ValueCoefficients = Tuple[Tuple[Value, ...]]


def jet_sin(
    primal: Tuple[Primal], series: PrimalCoefficients
) -> Tuple[Tuple[Value], ValueCoefficients]:
    """Taylor-mode arithmetic for the sine function.

    Args:
        primal: Input tensor.
        series: Tuple of Taylor coefficients.

    Returns:
        Tuple containing the value of the sine function and its Taylor coefficients.
    """
    (x,), (vs,) = primal, series

    order = len(vs)
    assert order == 2

    value = sin(x)
    ((v1, v2),) = series

    jac = cos(x)
    hess = -value

    v1_out = v1 * jac
    v2_out = v2 * jac + v1**2 * hess

    return (value,), ((v1_out, v2_out),)


MAPPING = {sin: jet_sin}
