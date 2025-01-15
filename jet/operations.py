"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Tuple

from torch import Tensor, cos, sin, zeros_like

from jet.utils import integer_partitions, multiplicity, tensor_prod

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
    sin_x = sin(x)
    cos_x = cos(x)

    def dn_sin(*vs) -> Tensor:
        """Contract the derivative tensor along the vectors."""
        n = len(vs)
        sign = 1 if n % 4 in [0, 1] else -1
        func = sin_x if n % 2 == 0 else cos_x
        return sign * func * tensor_prod(*vs)

    vs_out = [zeros_like(sin_x) for _ in vs]
    order = len(vs)

    for k in range(order):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn_sin(*vs_contract), alpha=nu)

    return sin_x, tuple(vs_out)


MAPPING = {sin: jet_sin}
