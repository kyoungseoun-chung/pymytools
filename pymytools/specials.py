#!/usr/bin/env python3
"""Torch wrapper for the scipy.special functions.

Since torch doesn't provide `ellipk` and `ellipe` functions, we need to wrap around `scipy.special` functions.

This is necessary to utilize GPU backend.

Note:
    - Here, we convert input tensor to cpu and return as `torch.device` Tensor.
    - Performance is just comparable to direct `scipy` function call with `cpu`. But using `gpu` backend is slightly worse.
        - I guess this is due to the copy of memory while it converts `gpu` tensor to `cpu` tensor.
        - Also, the actual computation is in `cpu` so it is expected.
        - Elliptic integral Performance, n=1000000
            ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
            ┃     Name     ┃ Elapsed time [s] ┃
            ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
            │ scipy_ellipe │     0.00806      │
            │ scipy_ellipk │     0.00845      │
            │ torch_ellipe │     0.00824      │
            │ torch_ellipk │     0.00841      │
            │ torch (gpu)  │     0.01007      │
            └──────────────┴──────────────────┘
"""
from scipy.special import ellipe
from scipy.special import ellipk
from torch import Tensor


def t_ellipk(x: Tensor):
    """Wrapper for scipy.special.ellipk."""
    return ellipk(x.cpu()).to(x.device, x.dtype)


def t_ellipe(x: Tensor):
    """Wrapper for scipy.special.ellipe."""
    return ellipe(x.cpu()).to(x.device, x.dtype)
