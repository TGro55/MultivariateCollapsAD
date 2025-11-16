"""Implements a module that computes the Laplacian via multijets and can be simplified.

Note: This module should be equivalent to the exact Laplacian form jet.laplacian. It was created to confirm just this.
"""

from typing import Callable

from torch import Tensor, eye, zeros
from torch.nn import Module

import jet
import multijet
from jet.vmap import traceable_vmap


class Laplacian(Module):
    """Module that computes the Laplacian of a function using jets."""

    def __init__(
        self, f: Callable[[Tensor], Tensor], dummy_x: Tensor, is_batched: bool
    ):
        """Initialize the Laplacian module.

        Args:
            f: The function whose Laplacian is computed.
            dummy_x: The input on which the Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
        """
        super().__init__()
        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.x_shape = dummy_x.shape
        self.x_kwargs = {"dtype": dummy_x.dtype, "device": dummy_x.device}

        self.unbatched_dim = (self.x_shape[1:] if is_batched else self.x_shape).numel()
        self.batched_dim = self.x_shape[0] if is_batched else 1
        self.is_batched = is_batched

        multijet_f = multijet.multijet(f, (2,))
        self.jet_f = traceable_vmap(multijet_f, self.unbatched_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 2-jet of f using
        canonical basis vectors for v1 and zero vectors for v2.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            Tuple containing the replicated function value, the Jacobian, and the
            Laplacian.
        """
        X0, X1, X2 = self.set_up_taylor_coefficients(x)
        F0, F1, F2 = self.jet_f(X0, X1, X2)
        return F0, F1, jet.utils.sum_vmapped(F2)

    def set_up_taylor_coefficients(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The three input tensors to the 2-jet that computes the Laplacian.
        """
        X0 = jet.utils.replicate(x, self.unbatched_dim)
        X2 = zeros(self.unbatched_dim, *self.x_shape, **self.x_kwargs)

        X1 = eye(self.unbatched_dim, **self.x_kwargs)
        if self.is_batched:
            X1 = X1.reshape(self.unbatched_dim, 1, *self.x_shape[1:])
            # copy without using more memory
            X1 = X1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X1 = X1.reshape(self.unbatched_dim, *self.x_shape)

        return X0, X1, X2
