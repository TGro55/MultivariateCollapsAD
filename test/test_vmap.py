from torch import eye, manual_seed, rand, vmap, zeros_like
from torch.autograd.functional import hessian
from torch.nn import Linear, Sequential, Tanh

from jet import jet


def laplacian_jet_loop(f, x):
    # compute the Laplacian using jets
    jet_f = jet(f, 2)
    lap = 0

    v2 = zeros_like(x)

    for i in range(x.numel()):
        v1 = zeros_like(x).flatten()
        v1[i] = 1.0
        v1 = v1.reshape(x.shape)
        _, _, d2i = jet_f(x, v1, v2)
        lap += d2i

    return lap


def laplacian_jet_vmap(f, x):
    jet_f = jet(f, 2)
    v2 = zeros_like(x)

    def d2_f(v1):
        return jet_f(x, v1, v2)[2]

    dim_x = x.numel()
    v1s = eye(dim_x, dtype=x.dtype, device=x.device).reshape(dim_x, *x.shape)

    vmap_d2_f = vmap(d2_f)
    lap = vmap_d2_f(v1s).sum()

    return lap


def laplacian(f, x):
    """Compute the Laplacian of a scalar function.

    Args:
        f: The scalar function to compute the Hessian of.
        x: The point at which to compute the Hessian.

    Returns:
        The Hessian of the function f at the point x.
    """
    hess = hessian(f, x)
    return hess.trace()


def test_laplacian_vmap():
    """Compute Laplacians in 3 different ways, ensure jet is vmap-compatible."""
    manual_seed(0)
    mlp = Sequential(
        Linear(5, 1, bias=False),
        Tanh(),
        # Linear(3, 1, bias=True),
        # Tanh(),
    )
    x = rand(5)

    lap_rev = laplacian(mlp, x)

    lap_jet_loop = laplacian_jet_loop(mlp, x)
    assert lap_rev.allclose(lap_jet_loop)
    print("Functorch and jet (loop) Laplacians match.")

    lap_jet_vmap = laplacian_jet_vmap(mlp, x)
    assert lap_rev.allclose(lap_jet_vmap)
    print("Functorch and jet (vmap) Laplacians match.")
