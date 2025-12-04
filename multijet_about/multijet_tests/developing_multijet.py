"""Testing code for development of multijet

Note: By now, this file has become overcrowded. I think it can serve for now as an
introduction to how the multijet was developed. Also, there are still some issues
with it that show up in the latter half of this file, that somewhat need addressing.
"""

# Pathing to make imports possible.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports
from torch import cos, manual_seed, ones_like, rand, sin, zeros_like
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh
import torch.nn as nn

# Importing multijet
from multijet import multijet
from multijet.Bilaplacian_with_sym import Bilaplacian as Bilaplacian_multijets_sym
from multijet.Bilaplacian import Bilaplacian as Bilaplacian_with_multijets
from multijet.utils import create_multi_idx_list, find_list_idx

# Importing jet
from jet.bilaplacian import Bilaplacian as Bilaplacian_with_jets

# For nodes creation
import copy  # Probalby unnecessary..

_ = manual_seed(0)  # make deterministic

# First some general testing
# Scalar-to-Scalar function
f = sin
k = (2,)  # Different to jet, since we deal with tuples now.
f_multijet = multijet(f, k)

# Set up the Taylor coefficients to compute the second derivative
x = rand(1)

x0 = x
x1 = ones_like(x)
x2 = zeros_like(x)

# Evaluate the second derivative
f0, f1, f2 = f_multijet(x0, x1, x2)

# Compare to the second derivative computed with first-order autodiff
d2f = hessian(f)(x)

if f2.allclose(d2f):
    print("Multivariate Taylor mode Hessian matches functorch Hessian!")
else:
    raise ValueError(f"{f2} does not match {d2f}!")

# We know the sine function's second derivative, so let's also compare with that
d2f_manual = -sin(x)
if f2.allclose(d2f_manual):
    print("Multivariate Taylor mode Hessian matches manual Hessian!")
else:
    raise ValueError(f"{f2} does not match {d2f_manual}!")


# Vector-to-Scalar function
D = 3

f = Sequential(Linear(D, 1), Tanh())
f_multijets = []
partial = [0 for i in range(D)]
for i in range(
    D
):  # Multiple multijets are actually not necessary, since they take tuples as input
    partial[i - 1] = 0
    partial[i] = 2
    f_multijets.append(
        multijet(f, tuple(partial))
    )  # Can also just use the same multijet: multijet(f,(2,))!

x = rand(D)

# constant Taylor coefficients
x0 = x
x2 = zeros_like(x)

d2_diag = zeros_like(x)

# Compute the d-th diagonal element of the Hessian
for d in range(D):
    x1 = zeros_like(x)
    x1[d] = 1.0  # d-th canonical basis vector
    f0, f1, f2 = f_multijets[d](x0, x1, x2)
    d2_diag[d] = f2

# %%
#
# Let's compare this to computing the Hessian with `functorch` and then taking its
# diagonal:

d2f = hessian(f)(x)  # has shape [1, D, D]
hessian_diag = d2f.squeeze(0).diag()

if d2_diag.allclose(hessian_diag):
    print(
        "Multivariate Taylor mode Hessian diagonal matches functorch"
        + " Hessian diagonal!\n"
    )
else:
    raise ValueError(f"{d2_diag} does not match {hessian_diag}!")

print("-" * 50)
# There is no difference between (2,2) and (2,0,2) representing the derivative.
# Only the shape of the multijet is important.

# Setting up Taylor coefficients
y = rand(D)
y0 = y
y1 = zeros_like(y)
nec_derivs = []
for i in create_multi_idx_list((2, 0, 2)):
    nec_derivs.append(i)
derivs = [y1 for i in range(len(nec_derivs) + 1)]

y1[2] = 1.0
y_0_0_1 = copy.deepcopy(y1)
y1[2] = 0

y1[0] = 1.0
y_1_0_0 = copy.deepcopy(y1)
y1[0] = 0

derivs[0] = y0
derivs[1] = y_0_0_1
derivs[3] = y_1_0_0

# First (2,0,2)
f_multijet_2_0_2 = multijet(f, (2, 0, 2))
solution_2_0_2 = f_multijet_2_0_2(*derivs)[-1]

# Second (2,2)
f_mulitjet_2_2 = multijet(f, (2, 2))
solution_2_2 = f_mulitjet_2_2(*derivs)[-1]

if solution_2_0_2.allclose(solution_2_2):
    print(
        "Indeed! multijet started with (2,2) and (2,0,2)"
        + " give same output given same input nodes.\n"
    )

print("-" * 50)
# Some multi-jets are more general than others and may be used to
# compute instinctively different mixed partials.
# Here some comparisons:


# Some modules for testing
# Sine
class Sine(nn.Module):
    def forward(self, x):
        return sin(x)


# Cos
class Cos(nn.Module):
    def forward(self, x):
        return cos(x)


# Cube
class Cube(nn.Module):
    def forward(self, x):
        return x**3


# Function
f = Sequential(Linear(3, 1), Sine())

# Derivative Direction
direction = 1

# Point of interest
y = rand(D)
y0 = y
y1 = zeros_like(y)

# Checking (1,1,1,1)-multijet against (4,) multijet with "same" inputs
# 4-multijet first
nodes = [y0]
for _ in range(4):
    nodes.append(copy.deepcopy(y1))
nodes[1][direction - 1] = 1.0

f_multijet_4_result = multijet(f, (4,))(*nodes)[-1]

# (1,1,1,1)-multijet second
nodes = [copy.deepcopy(y1) for _ in range(2**4)]
nodes[0] = y0
for k in create_multi_idx_list((1, 1, 1, 1)):
    if sum(k) == 1:
        nodes[find_list_idx(k, (1, 1, 1, 1))][direction - 1] = 1.0
f_multijet_1_1_1_1_result = multijet(f, (1, 1, 1, 1))(*nodes)[-1]

# Comparing
if f_multijet_4_result.allclose(f_multijet_1_1_1_1_result):
    print("(1,1,1,1) and (4,) give the same result.")
else:
    print(
        f"Differing results.\n4-multijet result is {f_multijet_4_result},"
        + f" while (1,1,1,1)-multijet result is {f_multijet_1_1_1_1_result}."
    )

# Checking (2,1,1)-multijet against (2,2)-multijet
# Two directions now
direction1 = 1
direction2 = 2

# (2,2)-multijet
nodes = [y0]
for _ in range(8):
    nodes.append(copy.deepcopy(y1))
nodes[1][direction1 - 1] = 1.0
nodes[3][direction2 - 1] = 1.0

f_multijet_2_2_result = multijet(f, (2, 2))(*nodes)[-1]

# (2,1,1)-multijet
nodes = [y0]
for _ in range(11):
    nodes.append(copy.deepcopy(y1))
nodes[1][direction1 - 1] = 1.0
nodes[2][direction1 - 1] = 1.0
nodes[4][direction2 - 1] = 1.0

f_multijet_2_1_1_result = multijet(f, (2, 1, 1))(*nodes)[-1]

# Comparing
if f_multijet_2_2_result.allclose(f_multijet_2_1_1_result):
    print("(2,2) and (2,1,1) give the same result.")
else:
    print(
        f"Differing results.\n(2,2)-multijet result is {f_multijet_2_2_result},"
        + f" while (2,1,1)-multijet result is {f_multijet_2_1_1_result}."
    )

# Checking (4,)-multijet against (2,2)-multijet
# (2,2)-multijet
nodes = [y0]
for _ in range(8):
    nodes.append(copy.deepcopy(y1))
nodes[1][direction - 1] = 1.0
nodes[3][direction - 1] = 1.0

f_multijet_2_2_same = multijet(f, (2, 2))(*nodes)[-1]

# Comparing
if f_multijet_2_2_same.allclose(f_multijet_4_result):
    print("(2,2) and (4,) give the same result.")
else:
    print(
        f"Differing results.\n(2,2)-multijet result is {f_multijet_2_2_same},"
        + f" while (2,1,1)-multijet result is {f_multijet_4_result}."
    )

print("-" * 50)
# Now to see, if we can succesfully compute the bilaplacian using multijets.
# We remain with a simple function to test
D = 3
f = Sequential(Linear(D, 1), Tanh())

# Bilaplacian using multijets
# Taylor coefficients
y0 = rand(D)
y = zeros_like(y0)

# The following is a manual intuitive computation of the Bilaplacian using multijets
to_be_summed = []

for d in range(D):
    y_list = [copy.deepcopy(y) for _ in range(5)]
    y_list[0] = y0
    y[d] = 1.0
    y_list[1] = copy.deepcopy(y)
    y[d] = 0
    to_be_summed.append(multijet(f, (4,))(*y_list)[-1])

for d1 in range(D):
    for d2 in range(D):
        if d1 == d2:
            continue
        y_list = [copy.deepcopy(y) for _ in range(9)]
        y_list[0] = y0

        # First non-zero node
        y[d1] = 1.0
        y_list[1] = copy.deepcopy(y)
        y[d1] = 0

        # Second non-zero node
        y[d2] = 1.0
        y_list[1] = copy.deepcopy(y)
        y[d2] = 0

        # Calculate the multijet
        to_be_summed.append(multijet(f, (2, 2))(*y_list)[-1])

manual_result = sum(to_be_summed)

print("-" * 50)

print("Now comparing Bilaplacian that uses symmetry with manual result.")
other_comparison = Bilaplacian_multijets_sym(f, y0, False)(y0)
if manual_result.allclose(other_comparison):
    print("Manual and multijet-Module results are the same.")
else:
    print("Differing results.")
    print(f"Manual result is {manual_result}. Module result is {other_comparison}.")

print("-" * 50)

print("Now comparing manual result to jet.")
manuel_jet_comparison = Bilaplacian_with_jets(f, y0, False)(y0)
if manual_result.allclose(manuel_jet_comparison):
    print("Manual and -jet-Module results are the same.")
else:
    print("Differing results.")
    print(
        f"Manual result is {manual_result}. Module result is {manuel_jet_comparison}."
    )

print("-" * 50)


print("Now comparing Bilaplacian modules.")
# Different test function, as there seems to be different erros ocurring
test_func = Sequential(Linear(D, 1), Tanh(), Cube())
x_dummy = rand(D)

jet_bilaplace = Bilaplacian_with_jets(test_func, x_dummy, is_batched=False)(x_dummy)
multijet_bilaplace = Bilaplacian_multijets_sym(test_func, x_dummy, is_batched=False)(
    x_dummy
)

if jet_bilaplace.allclose(multijet_bilaplace):
    print("Bilaplacian from jet and multijet coincide!\n")
else:
    print("Differing Results.")
    print(
        f"Result from using jets is {jet_bilaplace}."
        + f"\nResult from using multijets is {multijet_bilaplace}."
    )  # Different Functions lead to different errors.
    # Often multijet-result is 2^8 that of jet-result..

print("-" * 50)
print("Now comparing jet-bilaplacian with Ver2 of Bilaplacian-multijets.")

multijet_bilaplace_ver_2 = Bilaplacian_with_multijets(
    test_func, x_dummy, is_batched=False
)(x_dummy)

if jet_bilaplace.allclose(multijet_bilaplace_ver_2):
    print("Bilaplacian from jet and multijet coincide!\n")
else:
    print("Differing Results.")
    print(
        f"Result from using jets is {jet_bilaplace}."
        + f"\nResult from using multijets_ver_2 is {multijet_bilaplace_ver_2}."
    )

print("-" * 50)
