"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Tuple

from torch import Tensor, cos, sin

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
PrimalAndCoefficients = Tuple[Primal, Tuple[Primal, ...]]
Value = Tensor
ValueAndCoefficients = Tuple[Value, Tuple[Value, ...]]


def jet_sin(arg: PrimalAndCoefficients) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        arg: Input tensor and its Taylor coefficients.

    Returns:
        Tuple containing the value of the sine function and its Taylor coefficients.
    """
    (x, vs) = arg
    order = len(vs)
    assert order == 2

    value = sin(x)
    (v1, v2) = vs

    jac = cos(x)
    hess = -value

    v1_out = v1 * jac
    v2_out = v2 * jac + v1**2 * hess
    vs_out = (v1_out, v2_out)

    return value, vs_out


MAPPING = {sin: jet_sin}
