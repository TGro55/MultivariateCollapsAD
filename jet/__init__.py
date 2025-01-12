"""Taylor-mode automatic differentiation (jets) in PyTorch."""

from typing import Callable, Tuple

from torch.fx import GraphModule, symbolic_trace

from jet.operations import MAPPING, Primal, PrimalCoefficients, Value, ValueCoefficients


def jet(
    f: Callable[[Primal], Value], verbose: bool = False
) -> Callable[
    [Tuple[Primal], PrimalCoefficients], Tuple[Tuple[Value], ValueCoefficients]
]:
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
    # Create new input nodes for the Taylor coefficients at the start of the graph.
    # Insert them right at the beginning of the graph.
    first_node = next(iter(graph.graph.nodes))
    with graph.graph.inserting_after(first_node):
        vs_node = graph.graph.placeholder("vs")

    for idx, node in enumerate(tuple(graph.graph.nodes)):
        if node.op == "call_function":
            f = node.target
            if f in MAPPING.keys():
                # Pass the original input and the Taylor coefficient nodes to the
                # first input node
                args = (node.args[0], vs_node) if idx == 2 else (node.args[0],)
                with graph.graph.inserting_after(node):
                    new_node = graph.graph.call_function(MAPPING[f], args=args)
                node.replace_all_uses_with(new_node)
                graph.graph.erase_node(node)
            else:
                raise NotImplementedError(f"Unsupported node target: {node.target}")
        elif node.op not in ["output", "placeholder"]:
            raise NotImplementedError(f"Unsupported node operation: {node.op}")

    graph.graph.lint()
    graph.recompile()
    return graph
