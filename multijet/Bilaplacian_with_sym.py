"""Implements computing the Bi-Laplacian operator with Multivariate Taylor mode."""

from typing import Callable

from torch import Tensor, eye, zeros
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
            dummy_x: The input on which the Bi-Laplacian is computed. It is only used to
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

        multijet_f_4 = multijet.multijet(f, (4,))
        multijet_f_2_2 = multijet.multijet(f, (2, 2))
        self.multijet_f_4 = traceable_vmap(multijet_f_4, self.unbatched_dim)
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
        C1, C2 = self.set_up_taylor_coefficients(x)

        # 4-multijet summand
        _, _, _, _, F_4 = self.multijet_f_4(*C1)
        term1 = jet.utils.sum_vmapped(F_4)

        # there are no off-diagonal terms if the dimension is 1
        if self.unbatched_dim == 1:
            return term1

        # 2-2-multijet summand
        _, _, _, _, _, _, _, _, F_2_2 = self.multijet_f_2_2(*C2)
        factor2 = 2
        term2 = factor2 * jet.utils.sum_vmapped(F_2_2)

        return term1 + term2

    def set_up_taylor_coefficients(self, x: Tensor) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
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

        # Coefficients for 4-multijet
        X0 = jet.utils.replicate(x, D)
        X2 = zeros(D, *self.x_shape, **self.x_kwargs)
        X3 = zeros(D, *self.x_shape, **self.x_kwargs)
        X4 = zeros(D, *self.x_shape, **self.x_kwargs)

        X1 = eye(D, **self.x_kwargs)
        if self.is_batched:
            X1 = X1.reshape(D, 1, *self.x_shape[1:])
            # copy without using more memory
            X1 = X1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X1 = X1.reshape(D, *self.x_shape)

        C1 = (X0, X1, X2, X3, X4)

        # Coefficients for 2-2-multijet
        X00 = jet.utils.replicate(x, D * (D - 1) // 2)
        X02 = zeros(D * (D - 1) // 2, *self.x_shape, **self.x_kwargs)
        X11 = zeros(D * (D - 1) // 2, *self.x_shape, **self.x_kwargs)
        X12 = zeros(D * (D - 1) // 2, *self.x_shape, **self.x_kwargs)
        X20 = zeros(D * (D - 1) // 2, *self.x_shape, **self.x_kwargs)
        X21 = zeros(D * (D - 1) // 2, *self.x_shape, **self.x_kwargs)
        X22 = zeros(D * (D - 1) // 2, *self.x_shape, **self.x_kwargs)

        X01 = zeros(D * (D - 1) // 2, D, **self.x_kwargs)
        X10 = zeros(D * (D - 1) // 2, D, **self.x_kwargs)
        counter = 0
        for i in range(D - 1):
            for j in range(i + 1, D):
                X01[counter, i] = 1
                X10[counter, j] = 1
                counter += 1
        assert counter == D * (D - 1) // 2

        if self.is_batched:
            X01 = X01.reshape(D * (D - 1) // 2, 1, *self.x_shape[1:])
            X10 = X10.reshape(D * (D - 1) // 2, 1, *self.x_shape[1:])
            # copy without using more memory
            X01 = X01.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
            X10 = X10.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X01 = X01.reshape(D * (D - 1) // 2, *self.x_shape)
            X10 = X10.reshape(D * (D - 1) // 2, *self.x_shape)

        C2 = (X00, X01, X02, X10, X11, X12, X20, X21, X22)

        return C1, C2
