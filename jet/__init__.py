"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from math import factorial
from typing import Callable

from torch import Tensor, tanh, tensor, zeros_like
from torch.autograd import grad
from torch.fx import GraphModule, symbolic_trace
from torch.nn import Linear, Tanh
from torch.nn.functional import linear

from jet.operations import (
    MAPPING,
    Primal,
    PrimalAndCoefficients,
    Value,
    ValueAndCoefficients,
)


def jet(
    f: Callable[[Primal], Value], verbose: bool = False
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        verbose: Whether to print the traced graphs before and after overloading.
            Default: `False`.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """
    graph = symbolic_trace(f)

    if verbose:
        print("Traced graph before jet overloading:")
        print(graph.graph)

    jet_f = _replace_operations_with_taylor(graph)

    if verbose:
        print("Traced graph after jet overloading:")
        print(jet_f.graph)

    return jet_f


def _replace_operations_with_taylor(graph: GraphModule) -> GraphModule:
    """Replace operations in the graph with Taylor-mode equivalents.

    Args:
        graph: Traced PyTorch computation graph.

    Returns:
        The overloaded computation graph with Taylor arithmetic.

    Raises:
        NotImplementedError: If an unsupported operation or node is encountered while
            carrying out the overloading.
    """
    for node in tuple(graph.graph.nodes):
        if node.op == "call_function":
            f = node.target
            if f not in MAPPING.keys():
                raise NotImplementedError(f"Unsupported node target: {node.target}")
            with graph.graph.inserting_after(node):
                new_node = graph.graph.call_function(
                    MAPPING[f], args=node.args, kwargs=node.kwargs
                )
            node.replace_all_uses_with(new_node)
            graph.graph.erase_node(node)
        elif node.op == "call_module":
            submodule = graph.get_submodule(node.target)

            with graph.graph.inserting_before(node):
                if isinstance(submodule, Linear):
                    # Create placeholder nodes for weight and bias
                    weight_node = graph.graph.create_node(
                        "get_attr", f"{node.target}.weight"
                    )
                    bias_node = (
                        graph.graph.create_node("get_attr", f"{node.target}.bias")
                        if submodule.bias is not None
                        else None
                    )
                    # Use these nodes in the function call
                    new_node = graph.graph.call_function(
                        MAPPING[linear],
                        args=node.args,
                        kwargs={
                            "weight": weight_node,
                            "bias": bias_node,
                            **node.kwargs,
                        },
                    )
                elif isinstance(submodule, Tanh):
                    new_node = graph.graph.call_function(
                        MAPPING[tanh], args=node.args, kwargs=node.kwargs
                    )
                else:
                    raise NotImplementedError(f"Unsupported module: {submodule}")
            node.replace_all_uses_with(new_node)
            graph.graph.erase_node(node)

        elif node.op not in ["output", "placeholder", "get_attr"]:
            raise NotImplementedError(f"Unsupported node operation: {node.op}")

    graph.graph.lint()
    graph.recompile()
    return graph


def rev_jet(
    f: Callable[[Primal], Value]
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Implement Taylor-mode arithmetic via nested reverse-mode autodiff.

    Args:
        f: Function to overload. Maps a tensor to another tensor.

    Returns:
        The overloaded function that computes the function and its Taylor coefficients
        from the input tensor and its Taylor coefficients.
    """

    def jet_f(arg: PrimalAndCoefficients) -> ValueAndCoefficients:
        """Compute the function and its Taylor coefficients.

        Args:
            arg: Tuple containing the input tensor and its Taylor coefficients.

        Returns:
            Tuple containing the function value and its Taylor coefficients.
        """
        x, vs = arg
        order = len(vs)

        def path(t: Tensor):
            x_t = x + sum(
                t**n / factorial(n) * v_n for n, v_n in enumerate(vs, start=1)
            )
            return f(x_t)

        t = tensor(0.0, requires_grad=True, dtype=x.dtype, device=x.device)
        f_x = path(t)

        vs_out = [zeros_like(f_x).flatten() for _ in vs]

        for i, dnf_dt in enumerate(f_x.flatten()):
            for n in range(order):
                (dnf_dt,) = grad(dnf_dt, t, create_graph=True)
                vs_out[n][i] = dnf_dt.detach()

        f_x = f_x.detach()
        vs_out = tuple(v.detach().reshape_as(f_x) for v in vs_out)

        return f_x, vs_out

    return jet_f
