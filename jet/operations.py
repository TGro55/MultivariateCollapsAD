"""Implementation of AD primitives in Taylor-mode arithmetic."""

from typing import Optional

from torch import Tensor, cos, sigmoid, sin, tanh, zeros_like
from torch.nn.functional import linear

from jet.utils import (
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
    integer_partitions,
    multiplicity,
    tensor_prod,
)


def jet_sin(s: PrimalAndCoefficients, K: int, vmap: bool) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = s[0], s[1:]

    # pre-compute derivatives
    sin_x = sin(x)
    dsin = {0: sin_x, 1: cos(x)}

    def dn(*vs: Primal) -> Value:
        """Contract the derivative tensor along the vectors.

        Args:
            vs: The vectors to contract the derivative tensor along.

        Returns:
            The contracted derivative tensor.
        """
        n = len(vs)
        sign = 1 if n % 4 in [0, 1] else -1
        func = dsin[0] if n % 2 == 0 else dsin[1]
        return tensor_prod(sign * func, *vs)

    vs_out = [zeros_like(sin_x) for _ in range(K)]

    for k in range(K):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn(*vs_contract), alpha=nu)

    return (sin_x, *vs_out)


def jet_tanh(s: PrimalAndCoefficients, K: int, vmap: bool) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the hyperbolic tangent function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If the order of the Taylor expansion is greater than 3.
    """
    x, vs = s[0], s[1:]

    # pre-compute derivatives
    tanh_x = x.tanh()
    sech_x = 1 / x.cosh()
    dtanh = {0: tanh_x}
    if K >= 1:
        dtanh[1] = sech_x**2
    if K >= 2:
        dtanh[2] = -2 * dtanh[0] * dtanh[1]
    if K >= 3:
        dtanh[3] = -2 * dtanh[1] ** 2 + 4 * dtanh[0] ** 2 * dtanh[1]
    if K > 3:
        raise NotImplementedError(
            f"Tanh only supports derivatives up to third order. Got {K}."
        )

    def dn(*vs: Primal) -> Value:
        """Contract the derivative tensor along the vectors.

        Args:
            vs: The vectors to contract the derivative tensor along.

        Returns:
            The contracted derivative tensor.
        """
        return tensor_prod(dtanh[len(vs)], *vs)

    vs_out = [zeros_like(tanh_x) for _ in range(K)]

    for k in range(K):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn(*vs_contract), alpha=nu)

    return (tanh_x, *vs_out)


def jet_sigmoid(s: PrimalAndCoefficients, K: int, vmap: bool) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sigmoid function.

    Args:
        s: The primal and its Taylor coefficients.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If the order of the Taylor expansion is greater than 2.
    """
    x, vs = s[0], s[1:]

    # pre-compute derivatives
    sigmoid_x = sigmoid(x)
    dsigmoid = {0: sigmoid_x}
    if K >= 1:
        dsigmoid[1] = sigmoid_x * (1 - sigmoid_x)
    if K >= 2:
        dsigmoid[2] = dsigmoid[1] * (1 - 2 * sigmoid_x)
    if K >= 3:
        raise NotImplementedError(
            f"Sigmoid only supports derivatives up to second order. Got {K}."
        )

    def dn(*vs: Primal) -> Value:
        """Contract the derivative tensor along the vectors.

        Args:
            vs: The vectors to contract the derivative tensor along.

        Returns:
            The contracted derivative tensor.
        """
        return tensor_prod(dsigmoid[len(vs)], *vs)

    vs_out = [zeros_like(sigmoid_x) for _ in range(K)]

    for k in range(K):
        for sigma in integer_partitions(k + 1):
            vs_contract = [vs[i - 1] for i in sigma]
            nu = multiplicity(sigma)
            vs_out[k].add_(dn(*vs_contract), alpha=nu)

    return (sigmoid_x, *vs_out)


def jet_linear(
    s: PrimalAndCoefficients,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    K=None,
    vmap: bool = False,
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the linear function.

    Args:
        s: The primal and its Taylor coefficients.
        weight: The weight matrix.
        bias: The (optional) bias vector.
        K: The order of the Taylor expansion.
        vmap: Whether to `vmap` the primal value and its Taylor coefficients.

    Returns:
        The value and its Taylor coefficients.
    """
    x, vs = s[0], s[1:]

    linear_x = linear(x, weight, bias=bias)
    vs_out = [linear(vs[k], weight) for k in range(K)]

    return (linear_x, *vs_out)


MAPPING = {
    sin: jet_sin,
    tanh: jet_tanh,
    sigmoid: jet_sigmoid,
    linear: jet_linear,
}
