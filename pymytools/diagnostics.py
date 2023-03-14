#!/usr/bin/env python3
"""Diagnotstics for pystops project.

It contains
    - Save gridded data
    - Save discrete (particle) data
    - Tensorboard

Note that we assume all data are in the form of `torch.Tensor`.
"""
import glob
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import pandas as pd
import pyevtk
import torch
from pymyplot import Figure
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
from vtkmodules.vtkIOXML import vtkXMLImageDataReader

from pymytools.logger import logging
from pymytools.logger import markup

PATHLIKE = str | Path | os.PathLike
"""Path types. Either `str`, `pathlib.Path` or `os.PathLike`"""

DTYPE_T2N = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
}
"""Datatype converter from torch.dtype to numpy.dtype"""


class VTILoaded(TypedDict):
    """Loaded data from VTK file."""

    grid: tuple[Tensor, ...]
    """Grid information"""
    data: dict[str, Tensor]
    """Retrived data"""


class VTULoaded(TypedDict):
    """Loaded data from VTK file."""

    pos: Tensor
    "Particle position"
    vel: Tensor
    "particle velocity"


@dataclass
class DataLoader:
    """Load data from path.

    It supports following file types:
        - VTK (.vti, .vtu)
        - CSV (.csv)
        - HDF5 (.h5)

    Note:
        - Reading `vtk` format is not straightforward. Also, `pyevtk` does not support. Therefore, use `vtk` module directly.

    Args:
        dtype (torch.dtype): Data type of loaded data. Default is `torch.float64`.
        device (torch.device): Device of loaded data. Default is `torch.device("cpu")`.
    """

    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")

    def read_state_dict(
        self, architecture: torch.nn.Module, model_path: PATHLIKE
    ) -> torch.nn.Module:
        """Load state dict of the nn architecture."""

        state_dict_loc = Path(model_path).as_posix()

        architecture.load_state_dict(
            torch.load(state_dict_loc, map_location=self.device)
        )

        return architecture

    def read_vti(self, file_path: str, key: str | list[str] | None = None) -> VTILoaded:
        """Read VTK image data.

        Example:
            >>> loader = DataLoader()
            >>> res = loader.read_vtk("test.vti", key="test")
            # If a list of keys is given, it returns all data corresponding to the keys.
            # If None is given as the key, it returns all data.
            >>> res["grid"]
            tuple(Tensor, ...) # len(data["grid"]) is based on mesh dimension
            >>> res["data"]
            {key: Tensor}
            # res["grid"][0].shape == res["data"][key].shape

        Args:
            file_path (str): Path to load data. Note that `vtk` only accept string.
            key (str | list[str] | None): Key to retrive data. If `None` is given, it returns all data. Default is `None`.

        Returns:
            VTILoaded: loaded data, which is a dictionary with two keys: `grid` and `data`.
        """

        path = Path(file_path)
        if is_file(path) is False:
            raise FileNotFoundError(f"File {path} does not exist.")

        reader = vtkXMLImageDataReader()
        reader.SetFileName(path.as_posix())
        reader.Update()
        cell = reader.GetOutput()
        n_components = cell.GetCellData().GetNumberOfComponents()

        if isinstance(key, str):
            key = [key]
        elif key is None:
            key = [cell.GetCellData().GetArrayName(i) for i in range(n_components)]
        elif isinstance(key, list):
            assert all(
                isinstance(i, str) for i in key
            ), "Dataloader: some or all of keys are not str."
        else:
            raise TypeError("Dataloader: wrong type for key.")

        dx = cell.GetSpacing()
        origin = cell.GetOrigin()
        extent = cell.GetExtent()
        nx = [extent[2 * i] + extent[2 * i + 1] for i in range(3)]

        # Determine the dimension of the data
        # We put 1 grid point with the size of dx for dummy dimension
        data_dim = 3
        dim_check = 0
        for n in nx:
            if n == 1:
                dim_check += 1
                break
        data_dim -= dim_check

        cell_data: dict[str, Tensor] = {}

        # Retrive VTK cell data
        for k in key:
            # Convert VTK data to numpy array using key
            cell_val = vtk_to_numpy(cell.GetCellData().GetScalars(k))
            cell_val = cell_val.reshape(*nx[:data_dim]).T
            cell_dtype = cell_val.dtype

            # Convert numpy array to torch.Tensor
            cell_val = torch.tensor(cell_val, dtype=self.dtype, device=self.device)
            if cell_dtype == np.int32:
                cell_val = cell_val.gt(0.0)

            cell_data[k] = cell_val

        grid = [
            torch.linspace(
                origin[i] + dx[i] / 2,
                origin[i] + dx[i] / 2 + dx[i] * (nx[i] - 1),
                nx[i],
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(data_dim)
        ]
        mesh = torch.meshgrid(grid, indexing="ij")

        return {"grid": mesh, "data": cell_data}

    def read_vtu(self, file_path: str) -> VTULoaded:
        """Read VTK point data (unstructured grid).
        In my use case, `.vtu` file contains particle data usually either particle position or velocity. Especially the velocity information follows a convention of its identifier `U`, `V`, `W` for each dimension, therefore, we will return concatenated `Tensor` which is the data corresponding to `UVW` convention (lower case accepted) and has a dimension of `num_particle x 3`. Other than this keyword will be seprately loaded as single column (vector) data.

        Note:
            - Since this method only works for particle position and velocity, no key is required.

        Example:
            >>> dl = DataLoader()
            >>> data = dl.read_vtu("test.vtu")
            >>> data["pos"]
            Tensor # shape: num_particle x 3
            >>> data["vel"]
            Tensor # shape: num_particle x 3

        Args:
            file_path (str): Path to load data. Note that `vtk` only accept string.

        Returns:
            VTULoaded: Loaded data contains, which is a dictionary with two keys: `pos` and `vel`.
        """
        path = Path(file_path)

        if is_file(path) is False:
            raise FileNotFoundError(f"File {path} does not exist.")

        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(path.as_posix())
        reader.Update()
        point = reader.GetOutput()
        n_components = point.GetPointData().GetNumberOfComponents()

        # Key for particle data
        key = [point.GetPointData().GetArrayName(i) for i in range(n_components)]

        concat: list[Tensor] = []

        # Retrive VTK point data
        for k in key:
            # Convert VTK data to numpy array using key
            point_val = vtk_to_numpy(point.GetPointData().GetScalars(k))

            # Convert numpy array to torch.Tensor
            concat.append(torch.tensor(point_val, dtype=self.dtype, device=self.device))

        # Save VTK velocity data
        data_vel = torch.vstack(concat).T

        # Retrive VTK position data
        pos_val = vtk_to_numpy(point.GetPoints().GetData())
        data_pos = torch.tensor(pos_val, dtype=self.dtype, device=self.device)

        return {"pos": data_pos, "vel": data_vel}

    def read_csv(
        self, path: PATHLIKE, key: str | list[str] | None = None
    ) -> dict[str, Tensor]:
        """Read csv data using `pandas.read_csv`.

        Example:
            >>> ds = DataLoader()
            >>> res = ds.read_csv("test.csv", "x")
            >>> res
            {"x": Tensor}
            >>> res = ds.read_csv("test.csv", ["x", "y"])
            >>> res
            {"x": Tensor, "y": Tensor}

        Args:
            path (PATHLIKE): Path to load data.
            key (str or list[str], optional): Key for the loaded data. If None, all data will be loaded. Defaults to None

        Returns:
            dict[str, Tensor]: Loaded data.
        """

        if is_file(path) is False:
            raise FileNotFoundError(f"File {path} does not exist.")

        df = pd.read_csv(path)
        if isinstance(key, str):
            key = [key]
        elif key is None:
            key = list(df.columns)
        elif isinstance(key, list):
            assert all(
                [isinstance(k, str) for k in key]
            ), "key must be str or list[str]."
        else:
            raise TypeError("Dataloader: wrong type for key.")

        data_csv: dict[str, Tensor] = {}

        for k in key:
            data_csv[k] = torch.tensor(
                df[k].values, dtype=self.dtype, device=self.device
            )

        return data_csv

    def read_hdf5(
        self, path: PATHLIKE, key: str | list[str] | None = None
    ) -> dict[str, Tensor]:
        """Read hdf5 file. Loaded data will be

        Args:
            path (PATHLIKE): Path to load data.
            addr (str or list[str], optional): address for the loaded data (group name).

        Returns:
            Tensor: Loaded data.
        """

        if is_file(path) is False:
            raise FileNotFoundError(f"File {path} does not exist.")

        with h5py.File(path, "r") as f_h5:
            # If there is no subgroup, the keys are the name in datasets
            # If there are subgroups, the keys are the name of the subgroups
            all_keys = get_hdf5_dataset_keys(f_h5)

            if isinstance(key, str):
                key = [key]
            elif isinstance(key, list):
                assert all(
                    isinstance(k, str) for k in key
                ), "Dataloader: all entry of the given list of key must be str."
            elif key is None:
                key = [ak.rsplit("/", 1)[-1] for ak in all_keys]

            # Key to group + key
            key_to_get: list[str] = []
            for k in key:
                for ak in all_keys:
                    if k == ak.rsplit("/", 1)[-1]:
                        key_to_get.append(ak)

            assert len(key_to_get) == len(
                key
            ), f"Dataloader: some of given keys don't exist in hdf data set. (given: {key}, exist: {key_to_get})"

            data_hdf5: dict[str, Tensor] = {}

            for k in key_to_get:
                data = f_h5[k]

                assert isinstance(data, h5py.Dataset)

                data_hdf5[k.rsplit("/", 1)[-1]] = torch.tensor(
                    data[:], dtype=self.dtype, device=self.device
                )

            return data_hdf5


def get_hdf5_dataset_keys(f: h5py.File) -> list[str]:
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    if None in keys:
        raise KeyError("Dataloader: no keys exists in the given hdf5 dataset.")
    return keys


@dataclass
class DataSaver:
    """DataSaver class

    Args:
        save_dir (str): Directory to save data.
    """

    save_dir: str
    """Save directory"""

    def __post_init__(self) -> None:
        # Validate save directory. If not exist, create it.
        is_dir(self.save_dir, create=True)

    def save_model(
        self, model: torch.nn.Module, model_name: str | None = None, verbose: int = 0
    ):
        """Save the trained model state dict.

        Note:
            - The model is always saved as `model_name.pth` in the save directory.

        Args:
            model (torch.nn.Module): trained model
            model_name (str, optional): Model name. Defaults to "".
            verbose (int, optional): Verbosity level. Defaults to 0.
        """

        if model_name is None:
            model_name = "model.pth"
        else:
            model_name += ".pth"

        if verbose > 0:
            logging.info(markup(f"Saving model to {self.save_dir}", "blue", "bold"))

        torch.save(model.state_dict(), Path(self.save_dir, model_name))

    def save_hdf5(
        self, data: dict[str, Tensor] | dict[str, dict[str, Tensor]], file_name: str
    ) -> None:
        """Save data as hdf5 file.

        If data is given as `dict[str, dict[str, Tensor]]`, the data will be saved as a group in the hdf5 file.

        Args:
            data (dict[str, Tensor] | dict[str, dict[str, Tensor]]): data to save
            file_name (str): File name. It can be either with or without .h5 extension.
        """
        log_dir = Path(self.save_dir, file_name)

        if log_dir.suffix == "":
            log_dir = log_dir.with_suffix(".h5")

        with h5py.File(log_dir, "w") as f:
            for k, v in data.items():
                if isinstance(v, dict):
                    grp = f.create_group(k)
                    for k1, v1 in v.items():
                        grp.create_dataset(
                            k1, shape=v1.shape, dtype=DTYPE_T2N[v1.dtype], data=v1
                        )
                else:
                    f.create_dataset(k, shape=v.shape, dtype=DTYPE_T2N[v.dtype], data=v)

    def save_csv(
        self,
        data: dict[str, Tensor | float | int],
        file_name: str,
        itr: int | None = None,
    ):
        """Save data as csv file. If `itr` is not `None`, the data will be appended to the file every `itr`. Designed for saving time series data.

        data (dict[str, Tensor]): Data to save.
        file_name (str): File name. It can be either with or without .csv extension.
        itr (int, optional): Iteration key. Defaults to None.
        """
        log_dir = Path(self.save_dir, file_name)

        if log_dir.suffix == "":
            log_dir = log_dir.with_suffix(".csv")

        # If file already exists, delete it
        if itr is None:
            pd_mode = "w"
            pd_header = True

            df = pd.DataFrame(data)
            df.to_csv(log_dir.as_posix(), header=pd_header, mode=pd_mode)

        else:
            if itr == 0:
                # If file already exists, delete it
                if is_file(log_dir):
                    log_dir.unlink()

                pd_mode = "w"
                pd_header = True
            else:
                # If itr is given as integer, append data to the file without header
                pd_mode = "a"
                pd_header = False

            data_to_save = {}
            for k, v in data.items():
                data_to_save[k] = [v]
            df = pd.DataFrame(data_to_save)
            df.to_csv(log_dir.as_posix(), header=pd_header, mode=pd_mode)

    def save_vtu(
        self,
        data: dict[str, Tensor] | list[dict[str, Tensor]],
        file_name: str | list[str],
        chunk: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Save VTK point (unstructured grid) data (.vtu format).

        The purpose of this class is to save particle data. Therefore, it only accepts particle position and velocity data.

        Note:
            - The data must be in the form of `dict` with keys `pos` and `vel`.

        Example:
            >>> pos_e = torch.tensor(...) # shape: num_particle x 3
            >>> vel_e = torch.tensor(...) # shape: num_particle x 3
            >>> ds = DataSaver(save_dir="./data")
            >>> ds.save_vtu(
                {"pos": pos_e, "vel": vel_e}, "p_e.vtu"
                ) # Save to ./data/p_e.vtu
            # Or you can save multiple data at once by passing list of dict and list of file name
            >>> pos_i = torch.tensor(...) # shape: num_particle x 3
            >>> vel_i = torch.tensor(...) # shape: num_particle x 3
            >>> ds.save_vtu(
                [{"pos": pos_e, "vel": vel_e}, {"pos": pos_i, "vel": vel_i}],
                ["p_e", "p_i"]
                ) # Save to ./data/p_e.vtu and ./data/p_i.vtu

        Args:
            data (dict[str, Tensor] | list[dict[str, Tensor]]): data to save. Either dictionary or list of dictionary.
            file_name (str | list[str]): file name to save. Either string or list of string.
            chunk (int | None, optional): chunk data. Save only `filter` number of particle to the file. Defaults to None (save all).
            shuffle (bool, optional): Shuffle data to save. Defaults to False.
        """

        if isinstance(data, dict):
            data = [data]

        if isinstance(file_name, str):
            file_name = [file_name]

        assert len(data) == len(
            file_name
        ), "Dataloader: dict_data and file_name must have the same length."

        for idx, d in enumerate(data):
            slicer = slice(0, chunk)
            pos = d["pos"]
            vel = d["vel"]

            # Shuffle data. Mostly used for saving chunk of data.
            # If not shuffled, only a few first particles in the tensor will be saved.
            # This is not an ideal way to save data in case there is inflow or outflow in the simulation. Therefore, we shuffle the data.
            if shuffle:
                n_tot = pos.shape[0]
                pos = pos[torch.randperm(n_tot), :]
                vel = vel[torch.randperm(n_tot), :]

            log_dir = Path(self.save_dir, file_name[idx])

            # Do I need detach and cpu?
            pyevtk.hl.pointsToVTK(
                path=log_dir.as_posix(),
                x=np.ascontiguousarray(pos[slicer, 0]),
                y=np.ascontiguousarray(pos[slicer, 1]),
                z=np.ascontiguousarray(pos[slicer, 2]),
                data={
                    "U": np.ascontiguousarray(vel[slicer, 0]),
                    "V": np.ascontiguousarray(vel[slicer, 1]),
                    "W": np.ascontiguousarray(vel[slicer, 2]),
                },
                fieldData=None,
            )

    def save_vti(
        self,
        grid: tuple[Tensor, ...],
        data: dict[str, Tensor],
        file_name: str,
        empty_dim_scale: list[float] = [1.0, 1.0],
    ) -> None:
        """Save gridded data.

        Note:
            - Data always saved in 3D. Therefore, if the data is 2D, we add one dimension to the data.
            - To save in 3D, we introduce empty_dim_scale for non 3D data. It places one grid point to the empty dimension with scale factor.

        Example:
            >>> grid = torch.meshgrid(...)
            >>> field1 = torch.tensor(...)
            >>> field2 = torch.tensor(...)
            >>> ds = DataSaver(save_dir="./data")
            >>> ds.save_vti(
                {"field1": field1, "field2": field2}, "field_data"
                ) # Save to ./data/field_data.vtk

        Args:
            grid (tuple[Tensor, ...]): Grid information. Should be the result of torch.meshgrid.
            data (dict[str, Tensor]): Data to save. Data in the dictionary should have same shape of grid entities.
            file_name (str): File name to save.
            empty_dim_scale (list[float], optional): Scale factor for empty dimension. Defaults to [1.0, 1.0].
        """

        # Get grid information
        dim = len(grid)
        nx = grid[0].shape
        dx = [
            ((grid[i].max() - grid[i].min()) / (nx[i] - 1)).item() for i in range(dim)
        ]
        lower = [grid[i].min().item() - 0.5 * dx[i] for i in range(dim)]

        to_save = {}

        # Assign dummy delta x for 1 and 2D cases
        if dim == 1:
            view_size = (*nx, 1, 1)

            g_min = (
                *lower,
                -dx[0] * empty_dim_scale[0],
                -dx[0] * empty_dim_scale[1],
            )
            g_dx = (
                *dx,
                dx[0] * empty_dim_scale[0],
                dx[0] * empty_dim_scale[1],
            )
        elif dim == 2:
            view_size = (*nx, 1)

            g_min = (
                *lower,
                -dx[0] * empty_dim_scale[0],
            )
            g_dx = (
                *dx,
                dx[0] * empty_dim_scale[0],
            )
        else:
            view_size = (*nx,)
            g_min = (*lower,)
            g_dx = (*dx,)

        for k, v in data.items():
            # Convert torch tensor to numpy array since pyevtk only accepts numpy array
            data_to_save = (v.view(*view_size)).detach().cpu().numpy()

            # If data dtype is booleon, convert to int
            if data_to_save.dtype == np.bool_:
                data_to_save = data_to_save.astype(np.int32)

            to_save.setdefault(k, data_to_save)

        log_dir = Path(self.save_dir, file_name)

        pyevtk.hl.imageToVTK(
            path=log_dir.as_posix(),
            origin=g_min,
            spacing=g_dx,
            cellData=to_save,
        )


def is_dir(log_dir: PATHLIKE, create: bool = True, verbose: int = 0) -> bool:
    """Check directory and if doesn't exist, create directory.

    Example:
        # No `./data` directory exists.
        >>> is_dir("./data", create=False, verbose=1) # Check if directory exists.
        >>> is_dir("./data", create=True, verbose=1) # Create directory if it doesn't exist.
        True
        >>> is_dir("./data", create=False, verbose=1) # Check if directory exists.
        True

    Args:
        log_dir (PATHLIKE): Directory to check.
        create (bool, optional): Create directory if it doesn't exist. Defaults to True.
        verbose (int, optional): Verbosity level. Defaults to 0.
    """
    from pymytools.logger import logging, markup

    log_dir = Path(log_dir).resolve()

    if log_dir.exists():
        if verbose > 0:
            logging.info(markup(f"Directory exists at {log_dir}", "green", "bold"))
        return True
    else:
        if create:
            if verbose > 0:
                logging.info(markup(f"Creating directory: {log_dir}", "yellow", "bold"))
            log_dir.mkdir(parents=True)
            return True
        else:
            logging.info(markup(f"Directory doesn't exist!", "red", "bold"))
            return False


def is_file(file_path: PATHLIKE) -> bool:
    """Check file and if doesn't exist, create file."""

    file_path = Path(file_path)

    return file_path.is_file()


class DataTracker:
    """Track simulation data using tensorboard."""

    stage: Enum | None

    def __init__(self, log_dir: PATHLIKE, overwrite: bool = True):
        self.log_dir = log_dir

        is_dir(self.log_dir, create=True)

        if overwrite:
            pass
        else:
            self.clear_data()

        self._writer = SummaryWriter(log_dir)

    def set_stage(self, stage: Enum):
        """Set stage (Enum class)."""

        self.stage = stage

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        """Add batch metric to tensorboard."""

        if self.stage is None:
            msg = f"batch/{name}"
        else:
            msg = f"{self.stage.name}/batch/{name}"

        self._writer.add_scalar(msg, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        """Add epoch metric to tensorboard."""

        if self.stage is None:
            msg = f"epoch/{name}"
        else:
            msg = f"{self.stage.name}/epoch/{name}"

        self._writer.add_scalar(msg, value, step)

    def add_scalar(self, msg: str, value: float, step: int) -> None:
        """Add scalar metric to tensorboard."""

        if self.stage is None:
            msg = msg
        else:
            msg = f"{self.stage.name}/{msg}"

        self._writer.add_scalar(msg, value, step)

    def add_scalars(self, msg: str, value_dict: dict, step: int) -> None:
        """Add scalar metric to tensorboard."""

        self._writer.add_scalars(msg, value_dict, step)

    def add_figure(self, msg: str, fig: Figure, step: int) -> None:
        """Add matplotlib figure to tensorboard."""

        self._writer.add_figure(msg, fig, step)

    def clear_data(self) -> None:
        """Clear generated data for the tensorboard."""

        # Remove files in the log directory
        pth = Path(self.log_dir)
        rm_tree(pth)

    def flush(self):
        """Flush the data."""

        self._writer.flush()


def rm_tree(pth: Path):
    """Recursively remove all files in the directory."""

    # Remove all files
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    # Remove directory
    pth.rmdir()


def file_list_by_pattern(data_dir: PATHLIKE, pattern: str) -> list[str]:
    """Check directory and return list of all files matching the pattern.

    Example:
        # In `./data/` directory, there are `data_0000.h5`, `data_0001.h5`, `data_0002.h5`.
        >>> file_list_by_pattern("./data/", "data_*.h5")
        ['./data/data_0000.h5', './data/data_0001.h5', './data/data_0002.h5']
    """

    path = Path(data_dir, pattern).as_posix()

    # else if the directory is given, read all h5 files.
    file_list = glob.glob(path)

    return file_list
