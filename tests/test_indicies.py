#!/usr/bin/env python3


def test_tensor_idx() -> None:
    """Test for c_tensor."""

    from pymytools.indices import tensor_idx

    assert tensor_idx(3) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    assert tensor_idx(2) == [(0, 0), (0, 1), (1, 1)]

    assert tensor_idx(1) == [(0, 0)]
