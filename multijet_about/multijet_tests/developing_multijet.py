"""Testing code for development of multijet"""

#Pathing to make imports possible. DOES NOT WORK ;(( 
import os
import sys
parent_directory = os.path.abspath('..')
parent_2_directory= os.path.dirname(parent_directory)

print(parent_2_directory)
sys.path.append(parent_2_directory)

#Imports
from torch import Tensor, cos, manual_seed, ones_like, rand, sin, zeros_like
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh

#Importing multijet
# from parent_2_directory import multijet
from multijet import multijet

_ = manual_seed(0)  # make deterministic

#Scalar-to-Scalar function
f = sin
k = (2,)    #Different to jet, since we usually deal with tuples.
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


#Vector-to-Scalar function
D = 3

f = Sequential(Linear(D, 1), Tanh())
f_multijets = []
partial = [0 for i in range(D)]
for i in range(D):
    partial[i-1] = 0
    partial[i] = 2
    f_multijets.append(multijet(f,tuple(partial)))

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
    print("Multivariate Taylor mode Hessian diagonal matches functorch Hessian diagonal!")
else:
    raise ValueError(f"{d2_diag} does not match {hessian_diag}!")