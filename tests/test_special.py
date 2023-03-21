#!/usr/bin/env python3
"""Test special functions."""
import time

import torch
from scipy.special import ellipe
from scipy.special import ellipk
from torch import vmap
from torch.testing import assert_close  # type: ignore

from pymytools.logger import console
from pymytools.logger import Report
from pymytools.logger import Table
from pymytools.special import ellipe as t_ellipe_new
from pymytools.special import ellipk as t_ellipk_new
from pymytools.special import t_ellipe
from pymytools.special import t_ellipk


def test_elliptic_vmap() -> None:
    n_test = 1000

    device = torch.device("mps")

    vmap_ellipk = vmap(t_ellipk_new)
    vmap_ellipe = vmap(t_ellipe_new)

    test_k = torch.arange(-2, 0.9, 2.9 / n_test, dtype=torch.float32, device=device)

    tic = time.perf_counter()
    ek_scipy = t_ellipk(test_k)
    s_epk = time.perf_counter() - tic

    tic = time.perf_counter()
    ek_n = vmap_ellipk(test_k)
    t_epk = time.perf_counter() - tic

    assert_close(ek_scipy, ek_n)

    test_e = torch.arange(-1, 1, 2.0 / n_test, dtype=torch.float32, device=device)

    tic = time.perf_counter()
    ee_scipy = t_ellipe(test_e)
    s_epe = time.perf_counter() - tic

    tic = time.perf_counter()
    ee_n = vmap_ellipe(test_e)
    t_epe = time.perf_counter() - tic

    assert_close(ee_scipy, ee_n)

    data = {
        "Name": ["scipy_ellipe", "scipy_ellipk", "vamp_ellipe", "vmap_ellipk"],
        "Elapsed time (s)": [s_epe, s_epk, t_epe, t_epk],
    }
    table = Report(f"Time comparison for elliptic integral\nn={n_test}", data)
    table.display()


def test_new_elliptic() -> None:
    n_test = 1000000

    device = torch.device("cpu")

    test_k = torch.arange(-2, 0.95, 2.95 / n_test, device=torch.device("cpu")).to(
        device=device
    )

    tic = time.perf_counter()
    s_test = t_ellipk(test_k)
    s_epk = time.perf_counter() - tic

    tic = time.perf_counter()
    t_test = t_ellipk_new(test_k)
    n_epk = time.perf_counter() - tic

    assert_close(s_test, t_test)

    tic = time.perf_counter()
    s_test = t_ellipe(test_k)
    s_epe = time.perf_counter() - tic

    tic = time.perf_counter()
    t_test = t_ellipe_new(test_k)
    n_epe = time.perf_counter() - tic

    assert_close(s_test, t_test)


def test_elliptic_integral() -> None:
    n_test = 1000

    device = torch.device("cpu")
    dtype = torch.float64

    test_k = torch.arange(-2, 0.9, 2.9 / n_test, dtype=dtype, device=device)

    tic = time.perf_counter()
    s_test = ellipk(test_k)
    s_epk = time.perf_counter() - tic

    tic = time.perf_counter()
    t_test = t_ellipk(test_k)
    t_epk = time.perf_counter() - tic

    assert_close(s_test, t_test)

    test_e = torch.arange(-1, 1, 2.0 / n_test, dtype=dtype, device=device)

    tic = time.perf_counter()
    s_test = ellipe(test_e)
    s_epe = time.perf_counter() - tic

    tic = time.perf_counter()
    t_test = t_ellipe(test_e)
    t_epe = time.perf_counter() - tic

    assert_close(s_test, t_test)

    if torch.backends.mps.is_available():  # type: ignore
        device = torch.device("mps")
        dtype = torch.float32

        test_e = test_e.to(device=device, dtype=dtype)
        tic = time.perf_counter()
        g_test = t_ellipe(test_e)
        g_epe = time.perf_counter() - tic

        assert_close(
            s_test.to(dtype=dtype), g_test.to(device=s_test.device, dtype=dtype)
        )

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float64

        test_e = test_e.to(device=device, dtype=dtype)
        tic = time.perf_counter()
        g_test = t_ellipe(test_e)
        g_epe = time.perf_counter() - tic

        assert_close(
            s_test.to(dtype=dtype), g_test.to(device=s_test.device, dtype=dtype)
        )
    else:
        g_epe = "Not defined"

    table = Table(title=f"Elliptic integral Performance, n={n_test}")
    table.add_column("Name", justify="center", style="cyan")
    table.add_column(r"Elapsed time \[s]", justify="center", style="green")

    names = [
        "scipy_ellipe",
        "scipy_ellipk",
        "torch_ellipe",
        "torch_ellipk",
        "torch (gpu)",
    ]
    times = [s_epe, s_epk, t_epe, t_epk, g_epe]

    for k, v in zip(names, times):
        table.add_row(k, f"{v:.5f}")

    console.print(table)
