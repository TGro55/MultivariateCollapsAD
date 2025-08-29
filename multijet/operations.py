"""Implementation of AD primitives in Multivariate Taylor-mode arithmetic."""

import operator

from scipy.special import comb, factorial, stirling2
from torch import Tensor, cos, mul, sigmoid, sin, tanh
from torch.nn.functional import linear

import multijet.utils
from multijet.utils import (
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
    multi_partitions,
    multi_partition_refined,
    complete_multi_partitions,
    create_multi_idx_list,
    set_to_idx,
    find_list_idx,
    multiplicity,
)


def _multivariate_faa_di_bruno(vs: tuple[Primal, ...], K: tuple[int, ...], dn: dict[int, Primal]) -> list[Value]:
    """Apply Faà di Bruno's formula for elementwise functions.

    Args:
        vs: The incoming Taylor coefficients.
        K: Multi-index representing the highest partial derivative.
        dn: A dictionary mapping the degree to the function's derivative.

    Returns:
        The outgoing Taylor coefficients.
    """
    vs_out = []
    for k in create_multi_idx_list(K)[1:]:                  #Trivial multi-index (0,...,0) is not necessary.
        for idx, sigma in complete_multi_partitions(k):
            if dn[len(sigma)] is None:
                continue

            vs_count = {set_to_idx(i): sigma.count(i) for i in sigma}
            vs_contract = [
                vs[find_list_idx(i,K)] ** count if count > 1 else vs[find_list_idx(i,K)]
                for i, count in vs_count
            ]
            term = vs_contract[0]
            for v in vs_contract[1:]:
                term = mul(term, v)
            term = mul(term, dn[len(sigma)])

            nu = multiplicity(sigma)
            #avoid multiplication by one
            term = nu * term if nu != 1.0 else term
            vs_out.append(term if idx == 0 else vs_out.pop(-1) + term)
    return vs_out 

def multijet_sin(
    s: PrimalAndCoefficients, K: tuple[int, ...], is_taylor: tuple[bool, ...]
) -> ValueAndCoefficients:
    """Taylor-mode arithmetic for the sine function.

    Args:
        s: The primal and its Taylor coefficients.
        K: Multi-index representing the highest partial derivative..
        is_taylor: A tuple indicating which arguments are Taylor coefficients (`True`)
            and which are constants (`False`).

    Returns:
        The value and its Taylor coefficients.
    """
    assert is_taylor == (True,)

    x, vs = s[0], s[1:]

    # pre-compute derivatives
    sin_x = sin(x)
    dsin = {0: sin_x}
    for k in range(1, sum(list(K))+1):
        if k == 1:
            dsin[k] = cos(x)
        elif k in {2, 3}:
            dsin[k] = -1 * dsin[k - 2]
        else:
            dsin[k] = dsin[k - 4]

    vs_out = _multivariate_faa_di_bruno(vs, K, dsin)

    return (sin_x, *vs_out)

MAPPING = {
    sin: multijet_sin,
}