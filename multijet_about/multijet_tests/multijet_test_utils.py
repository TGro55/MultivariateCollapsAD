"""Tests `multijet.utils`."""

# Make imports possible
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multijet.utils import (
    multi_partitions,  # noqa: E402
    create_multi_idx_list,  # noqa: E402
    set_to_idx,  # noqa: E402
    find_list_idx,  # noqa: E402
    multiplicity,  # noqa: E402
)  # noqa: E402


def test_multi_partitions():
    """Tests 'multi_partitions' function."""

    def assert_loop(to_test, multi_idx):
        partitions = multi_partitions(multi_idx)
        to_be_compared_1 = []
        for part in to_test:
            ordered_partition = []
            for elem in part:
                ordered_partition.append(tuple(elem))
            ordered_partition = sorted(ordered_partition)
            to_be_compared_1.append(tuple(ordered_partition))
        to_be_compared_1 = sorted(to_be_compared_1)

        to_be_compared_2 = []
        for part in partitions:
            ordered_partition = []
            for elem in part:
                ordered_partition.append(tuple(elem))
            ordered_partition = sorted(ordered_partition)
            to_be_compared_2.append(tuple(ordered_partition))
        to_be_compared_2 = sorted(to_be_compared_1)

        assert to_be_compared_1 == to_be_compared_2

    test1 = [[[1]]]
    assert_loop(test1, (1,))

    test2 = [
        [
            [1],
            [1],
        ],
        [[1, 1]],
    ]
    assert_loop(test2, (2,))

    test0_2_0_2 = [
        [[2], [2], [4], [4]],
        [[2], [2], [4, 4]],
        [[2, 2], [4], [4]],
        [[2], [2, 4], [4]],
        [[2, 4], [2, 4]],
        [[2, 2], [4, 4]],
        [[2], [2, 4, 4]],
        [[2, 2, 4], [4]],
        [[2, 2, 4, 4]],
    ]
    assert_loop(test0_2_0_2, (0, 2, 0, 2))

    test1_3 = [
        [[1], [2], [2], [2]],
        [[1], [2], [2, 2]],
        [[1, 2], [2], [2]],
        [[1], [2, 2, 2]],
        [[1, 2, 2], [2]],
        [[1, 2], [2, 2]],
        [[1, 2, 2, 2]],
    ]
    assert_loop(test1_3, (1, 3))

    test1_2_0_1 = [
        [[1], [2], [2], [4]],
        [[1], [2], [2, 4]],
        [[1], [2, 2], [4]],
        [[1, 2], [2], [4]],
        [[1, 4], [2], [2]],
        [[1, 2, 4], [2]],
        [[1, 2, 2], [4]],
        [[1], [2, 2, 4]],
        [[1, 2], [2, 4]],
        [[1, 4], [2, 2]],
        [[1, 2, 2, 4]],
    ]
    assert_loop(test1_2_0_1, (1, 2, 0, 1))


def test_set_to_idx():
    """Tests 'set_to_idx' function."""
    assert set_to_idx([1], len((1,))) == (1,)
    assert set_to_idx([1], len((2,))) == (1,)
    assert set_to_idx([1, 2, 2], len((2, 2))) == (1, 2)
    assert set_to_idx([3, 3], len((1, 0, 3))) == (0, 0, 2)
    assert set_to_idx([1, 2], len((1, 1, 2))) == (1, 1, 0)


def test_find_list_idx():
    """Tests 'find_list_idx function."""
    assert find_list_idx((0,), (1,)) == 0
    assert find_list_idx((1,), (2,)) == 1
    assert find_list_idx((0, 0, 3), (1, 0, 3)) == 3
    assert find_list_idx((1, 2), (2, 2)) == 5
    assert find_list_idx((1, 0, 0), (1, 1, 2)) == 6


def test_create_multi_idx_list():
    """Tests 'create_multi_idx_list' function."""
    assert list(create_multi_idx_list((1,))) == [(1,)]
    assert list(create_multi_idx_list((2,))) == [(1,), (2,)]
    assert list(create_multi_idx_list((2, 2))) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    assert list(create_multi_idx_list((1, 0, 3))) == [
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
    ]
    assert list(create_multi_idx_list((1, 1, 2))) == [
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
    ]


def test_multiplicity():
    """Tests 'multiplicity' function."""
    assert multiplicity([[1]]) == 1
    assert multiplicity([[1], [1]]) == 1
    assert multiplicity([[1], [1, 2]]) == 2
    assert multiplicity([[1], [3], [3, 3]]) == 3
    assert multiplicity([[1, 1, 2, 2]]) == 1


# Do the tests
test_set_to_idx()
test_find_list_idx()
test_create_multi_idx_list()
test_multiplicity()
test_multi_partitions()
