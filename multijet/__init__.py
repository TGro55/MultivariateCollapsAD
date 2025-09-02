"""Multivariate Taylor-mode automatic differentiation (multijets) in PyTorch."""

from typing import Callable
from warnings import warn

from torch import add, prod, tensor, zeros_like
from torch.fx import Graph, GraphModule, Node

from multijet.operations import MAPPING
from multijet.tracing import capture_graph
from multijet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients

def analyze_dependencies(graph: Graph) -> tuple[set[Node], set[Node]]:
    """Determine nodes that depend on placeholders or only on constants.

    Args:
        graph: The graph to analyze.

    Returns:
        A tuple containing two sets:
        - The first set contains nodes that depend on placeholder nodes.
        - The second set contains nodes that depend only on constants.

    Raises:
        RuntimeError: If the dependencies cannot be determined for a node.
    """
    placeholder_nodes = {node for node in graph.nodes if node.op == "placeholder"}
    constant_nodes = {node for node in graph.nodes if node.op == "get_attr"}

    for node in graph.nodes:
        if node.op in ["placeholder", "get_attr"]:
            continue

        if any(n in placeholder_nodes for n in node.all_input_nodes):
            placeholder_nodes.add(node)
        elif all(n in constant_nodes for n in node.all_input_nodes):
            constant_nodes.add(node)
        else:
            raise RuntimeError(f"Could not detect dependencies for {node=}.\n{graph}")

    return placeholder_nodes, constant_nodes

def multijet(
    f: Callable[[Primal], Value], k: tuple[int,...], verbose: bool = False
) -> Callable[[PrimalAndCoefficients], ValueAndCoefficients]:
    """Overload a function with its Taylor-mode equivalent.

    Args:
        f: Function to overload. Maps a tensor to another tensor.
        k: Multi-index representing the highest partial derivative.
        verbose: Whether to print the traced graphs before and after overloading.
            Default: `False`.

    Returns:
        The overloaded function that computes the function and its sub-box of partial derivatives
        from the input tensor and its sub-box of partial derivatives.
    """
    mod = capture_graph(f)

    if verbose:
        print(f"Traced graph before jet overloading:\n{mod.graph}")

    multijet_mod = _replace_operations_with_multivariate_taylor(mod, k)

    if verbose:
        print(f"Traced graph after jet overloading:\n{multijet_mod.graph}")

    return multijet_mod

def _replace_operations_with_multivariate_taylor(  # noqa: C901
    mod: GraphModule, k: tuple[int,...]
) -> GraphModule:
    """Replace operations in the graph with multivariate Taylor-mode equivalents.

    Args:
        mod: Traced PyTorch computation graph module.
        k: Multi-index representing the highest partial derivative.

    Returns:
        The overloaded computation graph module with Taylor arithmetic.

    Raises:
        NotImplementedError: If an unsupported operation or node is encountered while
            carrying out the overloading.
        RuntimeError: If the multiplication type cannot be detected for a node.
    """
    graph = mod.graph

    # find the nodes that depend on the placeholder nodes and those that depend
    # only on constants
    dependent_on_placeholders, dependent_on_constants = analyze_dependencies(mod.graph)

    # If the output only depends on constants, the Taylor coefficients will be zero
    (output_node,) = [node for node in graph.nodes if node.op == "output"]
    if output_node not in dependent_on_placeholders:
        assert output_node in dependent_on_constants
        warn(
            f"The {output_node=} does not depend on the placeholder nodes. "
            f"The resulting jet will be trivially zero. {graph}"
        )
        # insert a node that generates the trivial Taylor components based on the
        # function value
        (out_tensor,) = output_node.args
        assert isinstance(out_tensor, Node)
        with graph.inserting_before(output_node):
            trivial_node = graph.call_function(
                lambda *args: tuple(
                    args[0] if i == 0 else zeros_like(args[0]) for i in range(prod(add(tensor(list(k)), 1))+1) #Different from jet
                ),
                args=(out_tensor,),
            )
            output_node.replace_input_with(out_tensor, trivial_node)
        dependent_on_placeholders.add(trivial_node)

    # find the input node and insert nodes for the Taylor coefficients
    (x,) = [node for node in graph.nodes if node.op == "placeholder"]
    with graph.inserting_after(x):
        vs = [graph.placeholder(name=f"v{i}") for i in reversed(range(1,prod(add(tensor(list(k)), 1))))][::-1] #Different from jet ##using tensor is probably unnecessary..

    # find the nodes that consume the original input, replace each with a new node whose
    # argument is the tuple of original input and Taylor coefficients
    children_x = [node for node in graph.nodes if x in node.args]
    for child_x in children_x:
        with graph.inserting_after(child_x):
            where = child_x.args.index(x)
            new_args = list(child_x.args)
            new_args[where] = (x, *vs)
            new_node = graph.call_function(
                child_x.target, args=tuple(new_args), kwargs=child_x.kwargs
            )
        child_x.replace_all_uses_with(new_node)
        graph.erase_node(child_x)
        dependent_on_placeholders.add(new_node)

    # replace all ops (including that of new_node) with their Taylor mode equivalents
    for node in tuple(graph.nodes):
        if node.op == "call_function":

            # figure out which arguments are constants, and which depend on placeholders
            is_taylor = []
            for arg in node.args:
                if isinstance(arg, Node):
                    in_placeholders = arg in dependent_on_placeholders
                    in_constants = arg in dependent_on_constants
                    assert int(in_placeholders) + int(in_constants) == 1
                    is_taylor.append(in_placeholders)

                elif isinstance(arg, tuple) and all(isinstance(a, Node) for a in arg):
                    is_taylor.append(True)

                elif isinstance(arg, (int, float)) or arg is None:
                    is_taylor.append(False)

                else:
                    raise RuntimeError(
                        f"Could not detect dependency of {arg} for {node.target=}."
                    )
            is_taylor = tuple(is_taylor)

            f = node.target

            # if all arguments are constants, we don't have to replace
            if not any(is_taylor):
                # add the node to constant dependencies
                dependent_on_constants.add(node)
                continue

            elif f not in MAPPING.keys():
                raise NotImplementedError(f"Unsupported {node.target=}.")

            with graph.inserting_after(node):
                new_node = graph.call_function(
                    MAPPING[f],
                    args=node.args,
                    kwargs={**node.kwargs, "K": k, "is_taylor": is_taylor},
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            dependent_on_placeholders.add(new_node)

        elif node.op == "call_module":
            module = graph.get_submodule(node.target)
            raise NotImplementedError(
                f"Unsupported module: {module}. Consider adding it to the"
                " `JetTracer.is_leaf_module` function."
            )

        elif node.op not in ["output", "placeholder", "get_attr"]:
            raise NotImplementedError(f"Unsupported node operation: {node.op}")

    mod.graph.lint()
    mod.recompile()

    return mod
    
