"""Utilities for testing multijet."""

from torch.autograd.functional import jacobian


def compute_deriv_tensor(func, inp, order):
    """Iteratively computes higher-order derivative tensors.

    Args:
        func: Function to differentiate.
        inp: Input tensor.
        order: Order of the derivative tensor

    Returns:
        Derivative tensor of the function of input degree.
    """
    current_func = func

    for _ in range(order):
        prev_func = current_func
        current_func = lambda x, f=prev_func: jacobian(
            f, x, create_graph=True
        )  # noqa: E731

    return current_func(inp)
