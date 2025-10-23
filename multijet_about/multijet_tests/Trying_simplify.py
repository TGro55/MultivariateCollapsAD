"""Module to test simplifying using the existing simplify function"""

# Pathing to make imports possible.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports
from torch import Tensor, cos, manual_seed, rand, sin
from torch.nn import Linear, Sequential, Tanh
import torch.nn as nn

# General jet imports
import jet
from jet.simplify import simplify
from jet.tracing import capture_graph

# General multijet imports
import multijet

# Bilaplacians
from jet.bilaplacian import Bilaplacian as Bilaplacian_with_jets
from multijet.Bilaplacian import Bilaplacian as Bilaplacian_with_multijets

_ = manual_seed(0)  # make deterministic


## Functions for testing
# Sine
class Sine(nn.Module):
    def forward(self, x):
        return sin(x)


# Cos
class Cos(nn.Module):
    def forward(self, x):
        return cos(x)


class Cube(nn.Module):
    def forward(self, x):
        return x**3


print("First, comparing general results.")

# Test function
test_func = Sequential(Linear(3, 1), Tanh(), Cube())

# Input
x_dummy = rand(3)

# Bilaplacians
jet_bilaplacian = Bilaplacian_with_jets(test_func, x_dummy, is_batched=False)
multijet_bilaplace = Bilaplacian_with_multijets(test_func, x_dummy, is_batched=False)

# Calculate jet-result for comparison
jet_bilaplace = jet_bilaplacian(x_dummy)

# Comparison
if jet_bilaplace.allclose(multijet_bilaplace(x_dummy)):
    print("Bilaplacian from jet and multijet coincide!\n")
else:
    print("Differing Results.")
    print(
        f"Result from using jets is {jet_bilaplace}.\nResult from using multijets is {multijet_bilaplace}."
    )

print("--" * 50)

# Graph 1: Simply capture the module that computes the Bilaplacian
mod_traced = capture_graph(multijet_bilaplace)
mod_traced_eval = mod_traced(x_dummy)
if jet_bilaplace.allclose(mod_traced_eval):
    print("Bilaplacian from jet and multijet coincide after capturing the graph!\n")
else:
    print("Differing Results.")
    print(
        f"Result from using jets is {jet_bilaplace}.\nResult from using multijets is {mod_traced_eval}."
    )

print("---" * 50)

# Graph 2: Simplify the module by removing replicate computations
mod_standard = simplify(mod_traced, pull_sum_vmapped=False)
mod_standard_eval = mod_standard(x_dummy)
if jet_bilaplace.allclose(mod_standard_eval):
    print("Bilaplacian from jet and multijet coincide, when ONLY pushing replicates.\n")
else:
    print("Differing Results.")
    print(
        f"Result from using jets is {jet_bilaplace}.\nResult from using multijets is {mod_standard_eval}."
    )

print("---" * 50)

# Graph 3: Simplify the module by removing replicate computations and pulling up the
# summations to directly propagate sums of Taylor coefficients
mod_collapsed = simplify(mod_traced, pull_sum_vmapped=True)
mod_collapsed_eval = mod_collapsed(x_dummy)
if jet_bilaplace.allclose(mod_collapsed_eval):
    print("Bilaplacian from jet and multijet coincide, when fully collapsed.\n")
else:
    print("Differing Results.")
    print(
        f"Result from using jets is {jet_bilaplace}.\nResult from using multijets is {mod_collapsed_eval}."
    )

# Node comparison
print("Nodes for multi-jet Bilaplacian.")
print(f"1) Captured: {len(mod_traced.graph.nodes)} nodes")
print(f"2) Standard simplifications: {len(mod_standard.graph.nodes)} nodes")
print(f"3) Collapsing simplifications: {len(mod_collapsed.graph.nodes)} nodes")

# For node comparison with multi-jet
mod_traced = capture_graph(jet_bilaplacian)
mod_standard = simplify(mod_traced, pull_sum_vmapped=False)
mod_collapsed = simplify(mod_traced, pull_sum_vmapped=True)
print("Nodes for jet Bilaplacian.")
print(f"1) Captured: {len(mod_traced.graph.nodes)} nodes")
print(f"2) Standard simplifications: {len(mod_standard.graph.nodes)} nodes")
print(f"3) Collapsing simplifications: {len(mod_collapsed.graph.nodes)} nodes")
