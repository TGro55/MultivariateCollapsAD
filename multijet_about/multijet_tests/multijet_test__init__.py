"""Tests for multijet/__init__.py."""

import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(project_root)
sys.path.insert(0, project_root)

import jet
import multijet

from pytest import mark
from torch import Tensor, cos, manual_seed, rand, sigmoid, sin, tanh, tensor, zeros_like
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

from jet.utils import (
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
    integer_partitions,
)
from jet.tracing import capture_graph
from multijet.utils import find_list_idx, create_multi_idx_list


from typing import Any, Callable
from itertools import combinations
from test.test___init__ import f_multiply, setup_case, compare_jet_results
from utils import compute_deriv_tensor


# We check against the already established jet-module
def check_multijet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients):
    x, vs = arg
    k = len(vs)

    multijet_f = multijet.multijet(f, (k,))
    multijet_out = multijet_f(x, *vs)

    jet_f = jet.jet(f, k, verbose=True)
    jet_out = jet_f(x, *vs)

    compare_jet_results(jet_out, multijet_out)


def create_multijet_input(
    K: tuple[int, ...], directions: dict[int, int], input: Primal
):
    """Creates multijet inputs given input directions.
    E.g. for the multijet (2,2) with directions {3: 2, 5: 2} and an input vector of size 5, this will return a list
    of input vectors the size of 5, of which the (0,1) multijet-entry will be tensor([0., 0., 1.0, 0., 0.,]) and the
    (1,0) entry will be tensor([0., 0., 0., 0., 1.0,]), while the other entries are just zero tensors.

    Args:
        K : Tuple of non-negative integers representing the endpoint of the multijet.
        directions : Dictionary, with keys as directions and derivative degree of the direction as items.
        input: Primal, to understand the shape of the input.

    Returns:
        List of Tensors, which are supposed to serve as multijet_input.
    """
    vs = []
    for root in create_multi_idx_list(K):
        vector = zeros_like(input)
        if sum(root) == 1:
            order = K[root.index(1)]
            for dir in list(directions.keys()):
                if directions[dir] == order:
                    vector[dir - 1] = 1.0
                    vs.append(vector)
                    del directions[dir]
                    break
        else:
            vs.append(vector)

    return vs


INF = float("inf")

# make generation of test cases deterministic
manual_seed(1)

JET_CASES = [
    # 1d sine function
    {"f": sin, "shape": (1,), "id": "sin"},
    # 2d sine function
    {"f": sin, "shape": (2,), "id": "sin"},
    # 3d cosine function
    {"f": cos, "shape": (3,), "id": "cos"},
    # 3d tanh function
    {"f": tanh, "shape": (5,), "id": "tanh"},
    # 4d sigmoid function
    {"f": sigmoid, "shape": (4,), "id": "sigmoid"},
    # linear layer
    {"f": Linear(4, 2), "shape": (4,), "id": "linear"},
    # 5d power function, two non-vanishing derivatives
    {"f": lambda x: x**2, "shape": (5,), "id": "pow-2"},
    # 5d power function, ten non-vanishing derivatives
    {"f": lambda x: x**10, "shape": (5,), "id": "pow-10"},
    # 5d power function, non-vanishing derivatives
    {"f": lambda x: x**1.5, "shape": (5,), "id": "pow-1.5"},
    # addition of a tensor and a float
    {"f": lambda x: x + 2.0, "shape": (5,), "id": "add-2.0"},
    # subtraction of a tensor and a float
    {"f": lambda x: x - 2.0, "shape": (5,), "id": "sub-2.0"},
    # multiplication of a tensor and a float
    {"f": lambda x: x * 3.0, "shape": (5,), "id": "mul-3.0"},
    # {"f": lambda x: jet.utils.replicate(x, 6), "shape": (5,), "id": "replicate-6"}, ##Not implemented
    # 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    # 2d tanh(tanh) function
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    # 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "id": "tanh-linear",
    },
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    # 5d tanh-activated two-layer MLP with batched input
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (10, 5),
        "is_batched": True,
        "id": "batched-two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {"f": lambda x: sigmoid(sigmoid(x)), "shape": (3,), "id": "sigmoid-sigmoid"},
    # 3d sin function with residual connection
    {"f": lambda x: sin(x) + x, "shape": (3,), "id": "sin-residual"},
    # 3d sin function with negative residual connection
    {"f": lambda x: sin(x) - x, "shape": (3,), "id": "sin-neg-residual"},
    # multiplication two variables
    {"f": f_multiply, "shape": (5,), "id": "multiply-variables"},
    # sum_vmapped
    # {"f": lambda x: jet.utils.sum_vmapped(x), "shape": (3, 5), "id": "sum_vmapped-3"}, ##Not implemented
]

# set the `is_batched` flag for all cases
for config in JET_CASES:
    config["is_batched"] = config.get("is_batched", False)

JET_CASES_IDS = [config["id"] for config in JET_CASES]

K_MAX = 5
K = list(range(K_MAX + 1))
K_IDS = [f"{k=}" for k in K]


@mark.parametrize("k", K, ids=K_IDS)
@mark.parametrize("config", JET_CASES, ids=JET_CASES_IDS)
def test_jet(config: dict[str, Any], k: int):
    """Compare forward jet with reverse-mode reference implementation.

    Args:
        config: Configuration dictionary of the test case.
        k: The order of the jet to compute.
    """
    f, x, vs, _ = setup_case(config, k=k)
    check_multijet(f, (x, vs))


@mark.parametrize("k", K, ids=K_IDS)
@mark.parametrize("config", JET_CASES, ids=JET_CASES_IDS)
def test_symbolic_trace_jet(config: dict[str, Any], k: int):
    """Test whether the function produced by jet can be traced.

    Args:
        config: Configuration dictionary of the test case.
        k: The order of the jet to compute.
    """
    f, _, _, _ = setup_case(config, k=k)
    # generate the jet's compute graph
    multijet_f = multijet.multijet(f, (k,))

    # try tracing it
    capture_graph(multijet_f)


def f_multiply_summed(x: Tensor) -> Tensor:
    """Test function for multiplication of two variables.

    Args:
        x: Input tensor.

    Returns:
        Tensor resulting from the multiplication of sin(x) and cos(sin(x)).
    """
    y = sin(x)
    return sin(y) * sum(cos(y))


# Altered list from above
JET_CASES = [
    # 1d sine function
    {"f": sin, "shape": (1,), "id": "sin"},
    # 2d sine function
    {"f": sin, "shape": (2,), "id": "sin"},
    # 3d cosine function
    {"f": cos, "shape": (3,), "id": "cos"},
    # 3d tanh function
    {"f": tanh, "shape": (5,), "id": "tanh"},
    # 4d sigmoid function
    {"f": sigmoid, "shape": (4,), "id": "sigmoid"},
    # linear layer
    {"f": Linear(4, 2), "shape": (4,), "id": "linear"},
    # 5d power function, two non-vanishing derivatives
    {"f": lambda x: x**2, "shape": (5,), "id": "pow-2"},
    # 5d power function, ten non-vanishing derivatives
    {"f": lambda x: x**10, "shape": (5,), "id": "pow-10"},
    # 5d power function, non-vanishing derivatives
    {"f": lambda x: x**1.5, "shape": (5,), "id": "pow-1.5"},
    # addition of a tensor and a float
    {"f": lambda x: x + 2.0, "shape": (5,), "id": "add-2.0"},
    # subtraction of a tensor and a float
    {"f": lambda x: x - 2.0, "shape": (5,), "id": "sub-2.0"},
    # multiplication of a tensor and a float
    {"f": lambda x: x * 3.0, "shape": (5,), "id": "mul-3.0"},
    # {"f": lambda x: jet.utils.replicate(x, 6), "shape": (5,), "id": "replicate-6"}, ##Not implemented
    # 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    # 2d tanh(tanh) function
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    # 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "id": "tanh-linear",
    },
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    # 5d tanh-activated two-layer MLP with batched input
    # {
    #     "f": Sequential(
    #         Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
    #     ),
    #     "shape": (10, 5),
    #     "is_batched": True,
    #     "id": "batched-two-layer-tanh-mlp",
    # },
    # 3d sigmoid(sigmoid) function
    {"f": lambda x: sigmoid(sigmoid(x)), "shape": (3,), "id": "sigmoid-sigmoid"},
    # 3d sin function with residual connection
    {"f": lambda x: sin(x) + x, "shape": (3,), "id": "sin-residual"},
    # 3d sin function with negative residual connection
    {"f": lambda x: sin(x) - x, "shape": (3,), "id": "sin-neg-residual"},
    # multiplication two variables
    {"f": f_multiply, "shape": (5,), "id": "multiply-variables"},
    # multiplication of variables with summation
    # {"f": f_multiply_summed, "shape": (5,), "id": "multiply-variables-summed"},
    # sum_vmapped
    # {"f": lambda x: jet.utils.sum_vmapped(x), "shape": (3, 5), "id": "sum_vmapped-3"}, ##Not implemented
]

# set the `is_batched` flag for all cases
for config in JET_CASES:
    config["is_batched"] = config.get("is_batched", False)

JET_CASES_IDS = [config["id"] for config in JET_CASES]

K_MAX = 3
K = list(range(1, K_MAX + 1))
K_IDS = [f"{k=}" for k in K]


@mark.parametrize("k", K, ids=K_IDS)
@mark.parametrize("config", JET_CASES, ids=JET_CASES_IDS)
def test_mixed_partials(config: dict[str, Any], k: int):
    """Test wether the multijet computes the correct end-points"""

    f = config["f"]
    shape = config["shape"]

    if isinstance(f, Module):
        f = f.double()

    # Input
    x = rand(*shape).double()

    # Compute full k-th order derivative tensor
    derivative_tensor = compute_deriv_tensor(f, x, k)

    # Setup different multijet-types for derivatives of degree k
    multijets = {}
    for partial in integer_partitions(k):
        multijets[partial] = multijet.multijet(f, partial)

    # Create dictionaries for which to check the endpoint derivatives
    poss_dir = []
    for i in range(1, shape[0] + 1):
        for _ in range(k):
            poss_dir.append(i)
    derivative_dictionaries = []
    for choice in set(combinations(poss_dir, k)):
        dictionary = {}
        for unique in set(choice):
            dictionary[unique] = choice.count(unique)
        derivative_dictionaries.append(dictionary)

    # Go through dictionary and assert correctness of the calculation
    for dict in derivative_dictionaries:

        # Find partial to dictionary and entry in the derivative tensor
        partial = []
        index = []
        for key in dict.keys():
            partial.append(dict[key])
            for _ in range(dict[key]):
                index.append(key - 1)
        partial = tuple(sorted(partial))

        # Calculate partial derivative using multijets
        vs = create_multijet_input(partial, dict, x)
        multijet_sol = multijets[partial](x, *vs)[-1]

        # Check against derivative tensor entry
        derivative_tensor_sol = derivative_tensor.transpose(0, -1)[
            tuple(index)
        ]  # The way the jacobian is constructed this transposition is needed, if input and output dimensions are not the same and the output dimension is not 1.
        assert multijet_sol.allclose(derivative_tensor_sol, rtol=1e-5, atol=1e-8)
