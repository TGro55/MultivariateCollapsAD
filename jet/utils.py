"""Utility functions for computing jets."""

from torch import Tensor, einsum


def integer_partitions(n: int, I: int = 1):
    """Compute the integer partitions of a positive integer.

    Taken from: https://stackoverflow.com/a/44209393.

    Args:
        n: Positive integer.
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in integer_partitions(n - i, i):
            yield (i,) + p


def tensor_prod(*tensors: Tensor) -> Tensor:
    """Compute the element-wise product of tensors.

    Args:
        tensors: Tensors to be multiplied.

    Returns:
        Element-wise product of the tensors.
    """
    (ndim,) = {t.ndim for t in tensors}
    idxs = "".join(chr(ord("a") + i) for i in range(ndim))
    equation = ",".join(len(tensors) * [idxs]) + f"->{idxs}"
    return einsum(equation, *tensors)
