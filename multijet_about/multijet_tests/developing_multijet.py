"""Testing code for development of multijet"""

#Pathing to make imports possible. 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

#Imports
from torch import Tensor, cos, manual_seed, ones_like, rand, sin, zeros_like
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh

#Importing multijet
from multijet import multijet 

_ = manual_seed(0)  # make deterministic

#Scalar-to-Scalar function
f = sin
k = (2,)    #Different to jet, since we deal with tuples now.
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
for i in range(D):      #Multiple multijets are now necessary, since they take tuples as input
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
    f0, f1, f2 = f_multijets[d](x0, x1, x2) # 'd' can be replaced with 0 and it still works. To be investigated!
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

# Is there a difference between (2,2) and (2,0,2) representing the derivative \partial^2x \partial^2y (f)? There seems to be!
from multijet.utils import create_multi_idx_list
y = rand(D)
y0 = y
y1 = zeros_like(y)

# First (2,0,2)

f_multijet_2_0_2 = multijet(f,(2,0,2))
nec_derivs = []
for i in create_multi_idx_list((2,0,2)):
    nec_derivs.append(i)
derivs = [y1 for i in range(len(nec_derivs)+1)]
for idx, partial in enumerate(nec_derivs):
    if sum(partial) == 0:
        derivs[idx+1] = y0
    elif sum(partial) == 1:
        y1[partial.index(1)] =  1.0
        derivs[idx+1] = y1
        y1[partial.index(1)] = 0

solution_2_0_2 = f_multijet_2_0_2(*derivs)[-1]

# Second (2,2)
f_mulitjet_2_2 = multijet(f,(2,2))
nec_derivs = []
for i in create_multi_idx_list((2,2)):
    nec_derivs.append(i)
print(nec_derivs)
derivs = [y1 for i in range(len(nec_derivs)+1)]
print(len(derivs))
for idx, partial in enumerate(nec_derivs):
    if sum(partial) == 0:
        derivs[idx+1] = y0
    elif sum(partial) == 1:
        y1[partial.index(1)] =  1.0
        derivs[idx+1] = y1
        y1[partial.index(1)] = 0

solution_2_2 = f_mulitjet_2_2(*derivs)[-1]

# print(solution_2_2 - solution_2_0_2)


