"""Bilaplacian operator with Multivariate Taylor mode with a more general idea."""

from typing import Callable

from torch import Tensor, zeros, vstack
from torch.nn import Module

import jet
import multijet
from jet.vmap import traceable_vmap


class Bilaplacian(Module):
    """Module that computes the Bi-Laplacian of a function using multijets.

    The Bi-Laplacian of a function f(x) ∈ R with x ∈ Rⁿ
    is defined as the Laplacian of the Laplacian, or
    Δ²f(x) = ∑ᵢ ∑ⱼ ∂⁴f(x) / ∂xᵢ²∂xⱼ² ∈ R.
    For functions that produce vectors or tensors, the Bi-Laplacian
    is defined per output component and has the same shape as f(x).
    """

    def __init__(
        self, f: Callable[[Tensor], Tensor], dummy_x: Tensor, is_batched: bool
    ):
        """Initialize the Bi-Laplacian module.

        Args:
            f: The function whose Bi-Laplacian is computed.
            dummy_x: The input on which the Bi-Laplacian is computed. It is
            only used to infer meta-data of the function input that `torch.fx`
            is not capable of determining at the moment.
            is_batched: Whether the function and its input are batched. In
            this case, we can use that computations can be carried out inde-
            pendently along the leading dimension of tensors.
        """
        super().__init__()
        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.x_shape = dummy_x.shape
        self.x_kwargs = {"dtype": dummy_x.dtype, "device": dummy_x.device}

        self.unbatched_dim = (self.x_shape[1:] if is_batched else self.x_shape).numel()
        self.batched_dim = self.x_shape[0] if is_batched else 1
        self.is_batched = is_batched

        multijet_f_2_2 = multijet.multijet(f, (2, 2))
        self.multijet_f_2_2 = traceable_vmap(multijet_f_2_2, self.unbatched_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the Bi-Laplacian of the function at the input tensor.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The Bi-Laplacian. Has the same shape as f(x).
        """
        # Two lists of Taylor Coefficients.
        # First is a 4-multijet. Second is a 2-2-multijet.
        C1 = self.set_up_taylor_coefficients(x)

        # 2-2-multijet summand
        _, _, _, _, _, _, _, _, F_2_2 = self.multijet_f_2_2(*C1)

        return jet.utils.sum_vmapped(F_2_2)

    def set_up_taylor_coefficients(
        self, x: Tensor
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ]:
        """Create the Taylor coefficients for the Bi-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            A tuple containing the inputs to the three 4-jets.
        """
        D = self.unbatched_dim

        # Coefficients for 2-2-multijet
        X00 = jet.utils.replicate(x, D**2)
        X02 = zeros(D**2, *self.x_shape, **self.x_kwargs)
        X11 = zeros(D**2, *self.x_shape, **self.x_kwargs)
        X12 = zeros(D**2, *self.x_shape, **self.x_kwargs)
        X20 = zeros(D**2, *self.x_shape, **self.x_kwargs)
        X21 = zeros(D**2, *self.x_shape, **self.x_kwargs)
        X22 = zeros(D**2, *self.x_shape, **self.x_kwargs)

        X01 = zeros(D**2, D, **self.x_kwargs)
        X10 = zeros(D**2, D, **self.x_kwargs)
        counter = 0
        components = []
        for i in range(D):
            can_vector = zeros(D, **self.x_kwargs)
            can_vector[i] = 1
            component_i = jet.utils.replicate(can_vector, D)
            components.append(component_i)
            for j in range(D):
                X01[counter, i] = 1
                X10[counter, j] = 1
                counter += 1
        assert counter == D**2
        diff_X01 = vstack(components)

        if self.is_batched:
            X01 = X01.reshape(D**2, 1, *self.x_shape[1:])
            X10 = X10.reshape(D**2, 1, *self.x_shape[1:])
            # Potentially Optimized approach
            diff_X01 = diff_X01.reshape(D**2, 1, *self.x_shape[1:])
            # copy without using more memory
            X01 = X01.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
            X10 = X10.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
            # Potentially Optimized approach
            diff_X01 = diff_X01.expand(
                -1, self.batched_dim, *(-1 for _ in self.x_shape[1:])
            )
        else:
            X01 = X01.reshape(D**2, *self.x_shape)
            X10 = X10.reshape(D**2, *self.x_shape)
            # Potentially Optimized approach
            diff_X01 = diff_X01.reshape(D**2, *self.x_shape)

        C1 = (X00, diff_X01, X02, X10, X11, X12, X20, X21, X22)

        return C1
