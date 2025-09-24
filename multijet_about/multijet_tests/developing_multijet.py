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
import copy

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
for i in range(D):      #Multiple multijets are actually not necessary, since they take tuples as input
    partial[i-1] = 0
    partial[i] = 2
    f_multijets.append(multijet(f,tuple(partial))) # Can also just use the same multijet: multijet(f,(2,))!

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
    print("Multivariate Taylor mode Hessian diagonal matches functorch Hessian diagonal!\n")
else:
    raise ValueError(f"{d2_diag} does not match {hessian_diag}!")

# There is no difference between (2,2) and (2,0,2) representing the derivative. Only the shape of the multijet is important.
from multijet.utils import create_multi_idx_list

# Setting up Taylor coefficients
y = rand(D)
y0 = y
y1 = zeros_like(y)
nec_derivs = []
for i in create_multi_idx_list((2,0,2)):
    nec_derivs.append(i)
derivs = [y1 for i in range(len(nec_derivs)+1)]

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
f_multijet_2_0_2 = multijet(f,(2,0,2))
solution_2_0_2 = f_multijet_2_0_2(*derivs)[-1]

# Second (2,2)
f_mulitjet_2_2 = multijet(f,(2,2))
solution_2_2 = f_mulitjet_2_2(*derivs)[-1]

if solution_2_0_2.allclose(solution_2_2):
    print("Indeed! multijet started with (2,2) and (2,0,2) give same output given same input nodes.\n")

# Now to see, if we can succesfully compute the bilaplacian using multijets.

# Imports
from jet.bilaplacian import Bilaplacian

# We remain with the same simple function to test
D = 3
f = Sequential(Linear(D, 1), Tanh())

# Bilaplacian using multijets
# Taylor coefficients
y0 = rand(D)
y = zeros_like(y0)

to_be_summed = []

for d in range(D):
    y_list = [copy.deepcopy(y) for i in range(5)]
    y_list[0] = y0
    y[d] = 1.0
    y_list[1] = copy.deepcopy(y)
    y[d] = 0
    to_be_summed.append(multijet(f,(4,))(*y_list)[-1])

for d1 in range(D):
    for d2 in range(D):
        if d1 == d2:
            continue
        y_list = [copy.deepcopy(y) for i in range(9)]
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
        to_be_summed.append(multijet(f,(2,2))(*y_list)[-1])

# Imports for checking correctness

result = sum(to_be_summed)

jet_f = Bilaplacian(f,y0,False)

result_jet_f = jet_f.forward(y0)

if result_jet_f.allclose(result):
    print("Bilplacian from jet and multijet coincide!\n")
else:
    print("Differing Results.")
    print(f"Result from using jets is {result_jet_f}.\nResult from using multijets is {result}.") # Currently seems to be off by a factor of 2.. 

# Comparing jet with multijet results
from jet import jet

jet_4_f = jet(f,4)

to_be_compared1 = []
to_be_compared2 = []
for d in range(D):
    y_list = [copy.deepcopy(y) for i in range(5)]
    y_list[0] = y0
    y[d] = 1.0
    y_list[1] = copy.deepcopy(y)
    y[d] = 0
    to_be_compared1.append(multijet(f,(4,))(*y_list)[-1])
    to_be_compared2.append(jet_4_f(*y_list)[-1])

print("-"*50)
for idx, cont in enumerate(to_be_compared1):
    if cont.allclose(to_be_compared2[idx]):
        print("YES! "*(idx+1))