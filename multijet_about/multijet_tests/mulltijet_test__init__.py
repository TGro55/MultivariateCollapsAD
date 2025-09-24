"""Tests for multijet/__init__.py."""

# For now, this program only tests derivatives with respect to one variable

# Make imports possible
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytest import mark
from torch import Tensor, cos, manual_seed, rand, sigmoid, sin, tanh, tensor
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

from typing import Any, Callable
from test.test___init__ import f_multiply, setup_case, compare_jet_results

import jet
import multijet
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients
from jet.tracing import capture_graph

# We check against the already established jet-module
def check_multijet(f: Callable[[Primal], Value], arg: PrimalAndCoefficients):
    x, vs = arg
    k = len(vs)

    multijet_f = multijet.multijet(f, (k,))
    multijet_out = multijet_f(x, *vs)

    jet_f = jet.jet(f, k, verbose=True)
    jet_out = jet_f(x, *vs)

    compare_jet_results(jet_out, multijet_out)

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