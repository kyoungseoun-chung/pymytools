#!/usr/bin/env python3
"""Test diagnostics"""
from enum import Enum, auto
import random
from pathlib import Path

import torch
from pymyplot import myplt as plt
from torch.testing import assert_close  # type: ignore

from pymytools.diagnostics import DataLoader
from pymytools.diagnostics import DataSaver
from pymytools.diagnostics import DataTracker
from pymytools.diagnostics import file_list_by_pattern

SAVE_DIR = "./tests/test_data/"


def test_file_list_by_pattern() -> None:
    f_list = file_list_by_pattern("./tests/test_data/", "test_*.vtu")

    target_list = [
        "tests/test_data/test_0.vtu",
        "tests/test_data/test_1.vtu",
        "tests/test_data/test_2.vtu",
    ]

    assert len(f_list) == len(target_list)

    for f in f_list:
        assert f in target_list


def test_vti() -> None:
    x = torch.linspace(0, 1, 2)
    y = torch.linspace(0, 1, 2)
    z = torch.linspace(0, 1, 2)

    grid = torch.meshgrid([x, y, z], indexing="ij")

    field_1 = grid[0] * 2
    field_2 = grid[0] * 0.2

    # Save single data
    ds = DataSaver(SAVE_DIR)
    ds.save_vti(grid, {"field_1": field_1, "field_2": field_2}, "test")

    dl = DataLoader(dtype=torch.float32)
    # Single key
    res = dl.read_vti(SAVE_DIR + "test.vti", "field_1")

    assert_close(res["grid"][0], grid[0])
    assert_close(res["data"]["field_1"], field_1)

    # Multiple keys
    res = dl.read_vti(SAVE_DIR + "test.vti", ["field_1", "field_2"])
    assert_close(res["data"]["field_1"], field_1)
    assert_close(res["data"]["field_2"], field_2)


def test_vtu() -> None:
    x = torch.linspace(0, 1, 2)
    y = torch.linspace(0, 1, 2)
    z = torch.linspace(0, 1, 2)

    grid = torch.meshgrid([x, y, z], indexing="ij")

    pos_1 = torch.stack(grid, dim=-1).reshape(-1, 3)
    vel_1 = torch.zeros_like(pos_1)
    for i in range(vel_1.shape[1]):
        vel_1[:, i] = torch.linspace(0, 1, vel_1.shape[0])

    pos_2 = pos_1.clone()
    vel_2 = vel_1.clone()

    # Save single data
    ds = DataSaver(SAVE_DIR)
    ds.save_vtu({"pos": pos_1, "vel": vel_1}, "test_0")

    ds.save_vtu(
        [{"pos": pos_1, "vel": vel_1}, {"pos": pos_2, "vel": vel_2}],
        ["test_1", "test_2"],
    )

    dl = DataLoader(dtype=torch.float32)

    data = dl.read_vtu(SAVE_DIR + "test_0.vtu")

    assert_close(data["pos"], pos_1)
    assert_close(data["vel"], vel_1)


def test_csv() -> None:
    x = torch.linspace(0, 1, 8)
    y = torch.linspace(0, 1, 8) * 2
    z = torch.linspace(0, 1, 8) * 3

    ds = DataSaver(SAVE_DIR)
    ds.save_csv({"x": x, "y": y, "z": z}, "test_1")

    dl = DataLoader(dtype=torch.float32)
    res = dl.read_csv(SAVE_DIR + "test_1.csv", "x")

    assert_close(res["x"], x)

    res = dl.read_csv(SAVE_DIR + "test_1.csv", ["x", "y"])

    assert_close(res["y"], y)

    res = dl.read_csv(SAVE_DIR + "test_1.csv")

    assert_close(res["z"], z)

    for i in range(8):
        ds.save_csv({"x": i * 2, "itr": i}, "test_2.csv", i)

    res = dl.read_csv(SAVE_DIR + "test_2.csv")
    target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
    assert_close(res["x"], target * 2)
    assert_close(res["itr"], target)


def test_hdf5() -> None:
    x = torch.linspace(0, 1, 8)
    y = torch.linspace(0, 1, 8) * 2
    z = torch.linspace(0, 1, 8) * 3

    ds = DataSaver(SAVE_DIR)
    ds.save_hdf5({"x": x, "y": y, "z": z}, "test_1")
    ds.save_hdf5({"grp1": {"x": x, "y": y, "z": z}, "grp2": {"a": x, "b": y}}, "test_2")

    dl = DataLoader(dtype=torch.float32)
    res = dl.read_hdf5(SAVE_DIR + "test_1.h5", "x")

    assert_close(res["x"], x)

    res = dl.read_hdf5(SAVE_DIR + "test_2.h5", ["x", "y", "a"])
    assert_close(res["y"], y)
    assert_close(res["a"], x)

    res = dl.read_hdf5(SAVE_DIR + "test_2.h5")
    assert_close(res["b"], y)


def test_tensorboard() -> None:
    tracker = DataTracker(Path(SAVE_DIR, "tmp"))

    x = torch.linspace(0, 1, 8)

    class Stage(Enum):
        TEST = auto()

    tracker.set_stage(Stage.TEST)
    for i in range(10):
        tracker.add_scalar("single", i * random.random(), i)
        tracker.add_scalars(
            "multi", {"a": i * random.random(), "b": i * random.random()}, i
        )
        tracker.add_epoch_metric("rand", i * random.random(), i)
        tracker.add_batch_metric("rand", i * random.random(), i)

        y = x**2 * torch.rand(x.shape)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        tracker.add_figure("fig", fig, i)

    tracker.flush()
    tracker.clear_data()
