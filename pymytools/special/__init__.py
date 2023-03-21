#!/usr/bin/env python3
"""Torch wrapper for the scipy.special functions.

* This functions are much slower than scipy.special functions. However, we need to use it for `torch.vmap` utilization.

Performance:
    - n=100000
        ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃     Name     ┃   Elapsed time (s)   ┃
        ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ scipy_ellipe │ 0.001932209124788642 │
        │ scipy_ellipk │ 0.025676249992102385 │
        │ vamp_ellipe  │ 0.05203570798039436  │
        │ vmap_ellipk  │ 0.08591929194517434  │
        └──────────────┴──────────────────────┘

Note on `t_ellipk` and `t_ellipe` (`torch.vmap` is not working for this function.):

- Since torch doesn't provide `ellipk` and `ellipe` functions, we need to wrap around `scipy.special` functions. This is necessary to utilize GPU backend.
- Here, we convert input tensor to cpu and return as `torch.device` Tensor.
- Performance is just comparable to direct `scipy` function call with `cpu`. But using `gpu` backend is slightly worse.
    - I guess this is due to the copy of memory while it converts `gpu` tensor to `cpu` tensor.
    - Also, the actual computation is in `cpu` so it is expected.
    - Elliptic integral Performance, n=1000000
        ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
        ┃     Name     ┃ Elapsed time (s) ┃
        ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
        │ scipy_ellipe │     0.00806      │
        │ scipy_ellipk │     0.00845      │
        │ torch_ellipe │     0.00824      │
        │ torch_ellipk │     0.00841      │
        │ torch (gpu)  │     0.01007      │
        └──────────────┴──────────────────┘
"""
from pymytools.special._ellipe import ellipe
from pymytools.special._ellipe import t_ellipe
from pymytools.special._ellipk import ellipk
from pymytools.special._ellipk import t_ellipk


__all__ = ["ellipk", "ellipe", "t_ellipk", "t_ellipe"]
