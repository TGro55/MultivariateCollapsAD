"""Utility functions for computing multijets."""

from torch import Tensor

from math import factorial, prod
from itertools import combinations
import copy

# type annotation for arguments and Taylor coefficients in input and output space
Primal = Tensor
Value = Tensor
# primals and values form a tuple
PrimalAndCoefficients = tuple[Primal, ...]
ValueAndCoefficients = tuple[Value, ...]

def multi_partitions(k: tuple[int, ...], I: int = 1):  # noqa: E741 #TODO
    """Compute the integer partitions of a positive integer.

    Args:
        k: Tuple of non-negative integers representing a multi-index.
        I: Minimal size of multi-set to be yielded.

    Yields:
        List of lists of integers representing the possible multi-index partitions.
    """
    
    #Find original multi-set
    multi_set = []
    for idx, count in enumerate(k):
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
                for k in range(1,len(multi_set_copy)+1):
                    for j in multi_partitions(set_to_idx(multi_set_copy),k):
                        yield [smd] + j
            else:
                yield [smd] + j

def multi_partition_refined(unrefined_multi_partitions):
    """Makes the multi-partitions from above unique and adds the final (trivial) multi-partition

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
    results.append([trivial])

    return results

def complete_multi_partitions(K: tuple[int,...]):
    """Combines the two functions of above. 
    They currently do the job AND give out the trivial partition [[K]] 
    as the final item, which is probably ok. Probably needs be improved
    in the future though.
    """
    return multi_partition_refined(multi_partitions(K))

def set_to_idx(multi_set: list[int, ...]):
    """Turn multi-set into multi_index

    Args:
        multi_set : List of non-negative integers.
    
    Returns:
        multi_idx : Tuple of integers representing the multi-set as a multi_idx.
    """
    multi_idx = []
    for i in range(max(multi_set)):
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
    """Make a list of multi-indices smaller or equal to given multi-index list

    Args:
        K : Tuple of non-negative integers, representing a multi-index.

    Yields:
        List of tuples of non-negative integers, which are smaller or equal to input multi-index.
    """
    #Starting mult-index
    previous = [0 for idx in K]

    #Constructer of next multi-index
    def increase_idx(curr, position=0):
        """Recursive function to increase multi-index correctly, i.e. it makes sure that 'k <= K'.

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

    #Loop to yield the multi-indices
    for i in range(prod([(idx+1) for idx in K])-1):

        #Trivial multi-index (0,...,0) must not be yielded
        if i == 0:
            previous = increase_idx(previous)
            continue
        
        to_be_yielded = tuple(previous)                 
        yield to_be_yielded
        previous = increase_idx(previous)
    
    #Final yield
    yield K

def multiplicity(sigma: list[list[[int, ...], ...], ...]) -> float:
    """Compute the scaling of a summand in Faa di Bruno's formula.

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