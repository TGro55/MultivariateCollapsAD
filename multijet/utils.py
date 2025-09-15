"""Utility functions for computing multijets."""

# Note: The functions in this file aid in computing partial derivatives according to the Mutlivariate Faà Di Bruno formula
# from M. Hardy's 'Combinatorics of Partial Derivatives', found here: "https://arxiv.org/pdf/math/0601149".
# To explain some terms:
# A multi-index refers to a specific partial derivative, e.g. (2,) refers to the second derivative in an unknown direction
# or (2,2) refers to the derivative ∂⁴f(x) / ∂xᵢ²∂xⱼ² for two unkown directions i,j.
# To a given multi-index the associated multi-set is just a list containing the entry positions of the multi-index exactly
# as many times, as the entry value says, e.g. for the multi-index (2,0,2) the associated multi-set is [1,1,3,3].
# A partition of a multi-set is called a multi-partition, it is a list of lists, such that if you were to take the union
# of the lists interpreted as sets and remember repeats, you again obtain the multi-set, e.g. for (2,0,2) the multi-set
# is [1,1,3,3] and a possible multi-partition is [[1],[1,3,3]].

from torch import Tensor

from math import factorial, prod
from itertools import combinations
import copy

# Imports from jet
from jet.utils import Primal, PrimalAndCoefficients, Value, ValueAndCoefficients

def yield_multi_partitions(K: tuple[int, ...], I: int = 1):  # noqa: E741 #TODO
    """Compute the multi-partitions of a multi-index (with possible repeats).
    E.g. For the multi-index (1,0,2), at mimimum, this function yields the following:
    [[1],[2],[2]], [[1],[2,2]], [[2], [1,2]], [[1,2,2]]

    Args:
        k: Tuple of non-negative integers representing a multi-index.
        I: Minimal size of multi-set to be yielded.

    Yields:
        List of lists of integers representing the possible multi-index partitions.
    """
    
    #Find original multi-set
    multi_set = []
    for idx, count in enumerate(K):
        for i in range(0,count):
            multi_set.append(idx+1)

    if len(multi_set) == I:
        yield [multi_set]
    
    for i in range(I, len(multi_set)// 2 + 1):
        summands = []
        for elem in sorted(set(combinations(multi_set,i))):
            summands.append(list(elem))
        for smd in summands:
            multi_set_copy = copy.deepcopy(multi_set)
            for k in smd:
                multi_set_copy.remove(k)
            if len(multi_set_copy) != 0:
                for k in range(i,len(multi_set_copy)+1):
                    for j in yield_multi_partitions(set_to_idx(multi_set_copy,len(K)),k):
                        yield [smd] + j

def refine_multi_partitions(unrefined_multi_partitions):
    """Eliminates repeats in list of multi-partitions from 'yield_multi_partitions()' and
    if necessary, adds the trivial partition, not yielded by 'yield multi_partitions()'.
    E.g. Given you have the two multi-partitions [[1,1],[2,2]] and [[2,2],[1,1]] in a
    list of a possible multi-partitions of the mult-set [1,1,2,2], they are repeats of 
    each other and only one of them will be in the final list, which is returned.

    Args:
        unrefined_mulit_partitions : List of Lists of Lists of non-negative integers repre-
        senting possible multi-partitions of a multi-set but with repeats
    
    Returns:
        hope: List of Lists of Lists of non-negative integers representing 
        possible multi-partitions of a multi-set WTIHOUT repeats.
    """
    #Kill doubles
    results = []
    for multpart in unrefined_multi_partitions:
        part = []
        for elem in multpart:
            part.append(tuple(sorted(elem)))
        #print(part)
        results.append(tuple(sorted(tuple(part))))
        #print(results)
    refining = tuple(sorted(results))
    # print(refining)
    refined = set(refining)

    #Return to set structure
    results = []
    for multpart in refined:
        part = []
        for elem in multpart:
            part.append(list(elem))
        results.append(list(part))
    
    #Add trivial multi-partition
    maximal_part_len = max([len(part) for part in results])

    part = results[0]
    idx = 0
    while len(part) != maximal_part_len:
        idx += 1
        part = results[idx]
    
    trivial = []
    for elem in part:
        trivial.append(elem[0])
    
    # Check if trivial partition is already in multi_partitions (mistake happens for multi-partitions for (1,)-type
    trivial_check = (tuple(trivial),)
    if not(trivial_check in refined):
        results.append([trivial])

    return results

def multi_partitions(K: tuple[int,...]):
    """Combines the two functions of above. 
    
    Currently, to compute multi-partitions to a given multi-set, 
    'yield_multi_partitions()' first generates multi-partitions to a given multi-set, but with 
    repreats, i.e. for the multi-index (2,2) it will give out [[1,1], [2,2]] and [[2,2], [1,1]]
    as two different multi-partitions, when they are really the same, and
    'refine_mulit_partitions()' afterwards, takes care of the repeats and also yields the multi-
    partition possibly not taken care of by 'yield_mulit_partitions()', which is at most the 
    trivial partition, to a multi-index, i.e. for (2,2) the multi-partition [[1,1,2,2]].

    Args:
        k: Tuple of non-negative integers representing a multi-index.

    Returns:
        List of Lists of Lists of non-negative integers, representing all possible multi-partitions to a given multi-index.
    """
    return refine_multi_partitions(yield_multi_partitions(K))

def set_to_idx(multi_set: list[int, ...], maxlen):
    """Turn multi-set into multi_index.
    E.g. Given the multi-set [1,3,3], this function turns that into a multi-index,
    like
    (1,0,2) or (1,0,2,0) or (1,0,2,0,...,0)
    depending on the given maximal length of the endpoint of the multi-jet, in-
    dicated by 'maxlen'.

    Args:
        multi_set : List of non-negative integers.
        maxlen : Positive integer representing the length the output should have.
    
    Returns:
        multi_idx : Tuple of integers representing the multi-set as a multi_idx.
    """
    multi_idx = []
    for i in range(maxlen):
        multi_idx.append(0)
    
    for i in set(multi_set):
        multi_idx[i-1] = multi_set.count(i)
    
    return tuple(multi_idx)

def find_list_idx(k: tuple[int,...],K: tuple[int,...]):
    """Finds the index/entry of a multi-index in the multi-jet list.

    Args:
        k : Tuple of non-negative integers, to be found the list index/entry of.
        K : Tuple of non-negative integers, maximal multi-index and the end of the multi-jet.
    
    Returns:
        Index/entry of k in the K-multi-jet list.
    """
    k_fixed = [0 for idx in K]
    for idx, count in enumerate(k):
        k_fixed[idx] = count
    
    for idx, count in enumerate(k_fixed):
        if idx == 0:
            N = count
            continue
        else:
            N = N*(K[idx]+1) + count
    
    return N

def create_multi_idx_list(K: tuple[int,...]):
    """Make a list of multi-indices smaller or equal to given multi-index list, except
    for the trivial multi-index.
    E.g. To the multi-index (1,0,2), this yields:
    (0,0,1), (0,0,2), (1,0,0), (1,0,1), (1,0,2),
    but does not yield (0,0,0).

    Args:
        K : Tuple of non-negative integers, representing a multi-index.

    Yields:
        List of tuples of non-negative integers, which are smaller or equal to input multi-index.
    """
    #Starting mult-index
    previous = [0 for idx in K]

    #Constructer of next multi-index
    def increase_idx(curr, position=0):
        """Recursive function to increase multi-index correctly, i.e. it makes sure that 'k <= K' component-wise.

        Args:
            curr: List of integers representing the multi-index as it currently is.
            position : Integer representing the position that needs increasing.
        
        Returns:
            List of integers representing the correctly increased multi-index.
        """
        rev_curr = curr[::-1]
        rev_curr[position] = rev_curr[position] + 1
        if rev_curr[position] <= K[-position-1]:
            return rev_curr[::-1]
        else:
            rev_curr[position] = 0
            return increase_idx(rev_curr[::-1],position + 1)

    # Loop to yield the multi-indices
    for i in range(prod([(idx+1) for idx in K])-1):

        # Trivial multi-index (0,...,0) is not yielded
        if i == 0:
            previous = increase_idx(previous)
            continue
        
        to_be_yielded = tuple(previous)                 
        yield to_be_yielded
        previous = increase_idx(previous)
    
    # Final yield
    yield K

def multiplicity(sigma: list[list[[int, ...], ...], ...]) -> float:
    """Compute the scaling of a summand in Faa di Bruno's formula taken from M. Hardy's 'Combinatorics
    of Partial Derivatives' found here: "https://arxiv.org/pdf/math/0601149".

    Args:
        sigma: List of lists of non-negative integers representing a multi-partition of a multi-index.

    Returns:
        Multiplicity of sigma.

    Raises:
        ValueError: If the multiplicity is not an integer.
    """
    #Find original multi-partition
    k = []
    for part in sigma:
        k += part

    #Prepare the product of counting the integers
    n_part = {i: prod(factorial(part.count(i)) for part in sigma) for i in set(k)}

    #Find multiplicities in summands of sigma
    m_part = [[tuple(eta) for eta in sigma].count(unique) for unique in set(tuple(smd) for smd in sigma)]

    #Calculate multiplicity
    multiplicity = (
        prod(factorial(k.count(i)) for i in set(k))
        / prod(factorial(eta) for eta in m_part)
        / prod(val for val in n_part.values())
    )
    if not multiplicity.is_integer():
        raise ValueError(f"Multiplicity should be an integer, but got {multiplicity}.")
    return multiplicity