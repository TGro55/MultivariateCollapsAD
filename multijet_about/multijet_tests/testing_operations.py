  """Testing operations written in multijet.operations"""

# Make imports possible
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports
from multijet import multijet
from torch import Tensor, cos, manual_seed, ones_like, rand, sin, zeros_like, stack, ones, transpose, sigmoid
from torch import sum as sumtorch
from torch.func import hessian
from torch.nn import Linear, Sequential, Tanh
import torch.nn as nn
from torch.autograd.functional import jacobian
from random import random

# Testing Dimension
D=3

# Test for the Operation
def test_hessian(f, operation, double=False):
    """Function to test multijet result for given function by caclulating the hessian.
    
    Args:
        f: function to test.
        operation: String indicating the test operation.
    
    Returns:
        ---
    
    Raises:
        ValueError, if multijet doesn't compute the same result as the built in tool.
    """

    local_dim = D
    if double:
        local_dim = 2* D

    # Multijet
    k = (2,)
    f_multijet = multijet(f, k)

    # Input
    x = rand(local_dim)
    x0 = x
    x2 = zeros_like(x)

    # Output of multi-jet
    d2_diag = zeros_like(x)
    if operation == "Multiplication of two variables":
        print(d2_diag)
        d2_diag = []

    # Compute the d-th diagonal element of the Hessian
    for d in range(local_dim):
        x1 = zeros_like(x)
        x1[d] = 1.0  # d-th canonical basis vector
        f0, f1, f2 = f_multijet(x0, x1, x2) # 'd' can be replaced with 0 and it still works. To be investigated!
        if operation != "Multiplication of two variables":
            d2_diag[d] = f2
        else:
            d2_diag.append(f2)

    if operation == "Multiplication of two variables":
        d2_diag_stack = stack(d2_diag)
        print(d2_diag)

    d2f = hessian(f)(x)  # has shape [1, D, D]
    if operation == "Multiplication of two variables":
        print(d2f)

    hessian_diag = d2f.squeeze(0).diag()

    if not(d2_diag.allclose(hessian_diag)):
        raise ValueError(operation + f" is wrong. {d2_diag} does not match {hessian_diag}!")

def test_gradient(f, operation, double=False):
    """Function to test multijet result for given function by caclulating the hessian.
    
    Args:
        f: function to test.
        operation: String indicating the test operation.
    
    Returns:
        ---
    
    Raises:
        ValueError, if multijet doesn't compute the same result as the built in tool.
    """
    local_dim = D
    if double:
        local_dim = 2*D
    # Multijet
    k = (1,)
    f_multijet = multijet(f, k)

    # Input
    x = rand(local_dim)
    x0 = x

    # Output of multi-jet
    gradient_list = []

    # Compute the d-th diagonal element of the Hessian
    for d in range(local_dim):
        x1 = zeros_like(x)
        x1[d] = 1.0  # d-th canonical basis vector
        f0, f1 = f_multijet(x0, x1)
        gradient_list.append(f1)

    f_gradient = transpose(stack(gradient_list), 0, 1)

    control_gradient = jacobian(f,x)

    if not(f_gradient.allclose(control_gradient)):
        raise ValueError(operation + f" is wrong. {f_gradient} does not match {control_gradient}!")
    
# Operations followed by test.

# First the basis
f = Sequential(Linear(D,1))
test_hessian(f, "Linear")

# Sine
class Sine(nn.Module):
    def forward(self, x):
        return sin(x)
f = Sequential(Linear(D,1), Sine())
test_hessian(f, "Sin")

# Cosine
class Cosine(nn.Module):
    def forward(self, x):
        return cos(x)
f = Sequential(Linear(D,1), Cosine())
test_hessian(f, "Cos")

# Tanh
f = Sequential(Linear(D,1), Tanh())
test_hessian(f,"Tanh")

# Sigmoid
class Sigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)
f = Sequential(Linear(D,1), Sigmoid())
test_hessian(f, "Sigmoid", False)
test_gradient(f, "Sigmoid", False)

# Integer Exponents
class Pow(nn.Module):
    def forward(self, x):
        return x**exponent

for exponent in range(2,5):
    f = Sequential(Linear(D,1), Pow())
    test_hessian(f,f"Exponantiation with {exponent}")

# Addition

# Adding a constant
const = ones(D)
class AddConstant(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.register_buffer("const", const)

    def forward(self, x):
        return x + self.const

f = Sequential(AddConstant(const), Linear(D,1))
test_hessian(f, "Adding a constant")
test_gradient(f, "Adding a constant")

# Add to a constant
# class AddtoConstant(nn.Module):
#     def __init__(self, const):
#         super().__init__()
#         self.register_buffer("const", const)

#     def forward(self, x):
#         return self.const + x

# f = Sequential(AddtoConstant(const), Linear(D,1))
# test_hessian(f, "Adding to a constant")
# test_gradient(f, "Adding to a constant")

# Adding of two Variables


# Subtraction

# Subtracting a constant
class SubConstant(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.register_buffer("const", const)

    def forward(self, x):
        return x - self.const

f = Sequential(SubConstant(const), Linear(D,1))
test_hessian(f, "Subtracting a constant")
test_gradient(f, "Subtracting a constant")

# Subtracting from a constant
# class SubfromConstant(nn.Module):
#     def __init__(self, const):
#         super().__init__()
#         self.register_buffer("const", const)

#     def forward(self, x):
#         return self.const - x

# f = Sequential(SubfromConstant(const), Linear(D,1))
# test_hessian(f, "Subtracting from a constant")
# test_gradient(f, "Subtracting from a constant")

# Subtracting of two Variables

# Multiplication

# Multiplication with a constant
class MultiplybyConstant(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.register_buffer("const", const)

    def forward(self, x):
        return x * self.const

f = Sequential(MultiplybyConstant(const), Linear(D,1))
test_hessian(f, "Multiplying by a constant") 
test_gradient(f, "Multiplying by a constant")

# Mulitplication of a constant with a variable
# class MultiplytoConstant(nn.Module):
#     def __init__(self, const):
#         super().__init__()
#         self.register_buffer("const", const)

#     def forward(self, x):
#         return self.const * x

# f = Sequential(MultiplytoConstant(const), Linear(D,1))
# test_hessian(f, "Mulitplication of a constant with a variable")
# test_gradient(f, "Mulitplication of a constant with a variable")

# Multiplication of two variables
def f_multiply(x: Tensor) -> Tensor:
    """Test function for multiplication of two variables.

    Args:
        x: Input tensor.

    Returns:
        Tensor resulting from the multiplication of sin(x) and cos(sin(x)).
    """
    y = sin(x)
    return sin(y) * cos(y)

test_hessian(f_multiply, "Multiplication of two variables", False)
test_gradient(f_multiply, "Multiplication of two variables", False)