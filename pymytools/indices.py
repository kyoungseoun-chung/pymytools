#!/usr/bin/env python3
"""Returning list of indices for a various purposes."""
from itertools import combinations_with_replacement


def tensor_idx(dim: int) -> list[tuple[int, int]]:
    """Return the list of indices for the Cartesian tensor.

    Example:

        >>> c_tensor(3)
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        >>> c_tensor(2)
        [(0, 0), (0, 1), (1, 1)]
        >>> c_tensor(1)
        [(0, 0)]


    Args:
        dim (int): Dimension of the tensor.

    Returns:
        list[tuple[int, int]]: List of indices.
    """

    assert dim in [1, 2, 3], "Only 1, 2, and 3 dimensions are supported."

    return list(combinations_with_replacement(range(dim), 2))
