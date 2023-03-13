# Collection of Useful Tools for `pystops` Project

During my PhD, I noticed that some features that I implemented throughout my projects appeared repeatedly.
As a result, I decided to separate these features and manage them appropriately.

## Goal

Manage/unify fragmented features and deploy to `pypi.org`.

## Features

- Diagnostics
  - file I/O
    - Supports: `csv`, `hdf5` (`.h5`), and `vtk` (both `.vti` and `.vtu`) formats
    - All data loaded are `torch.Tensor`
  - Tensorboard tracker
- Logging, timer, and progress bar
  - Prettier representation using `rich` package

## Dependencies

Main dependencies:

- `python = "^3.10"`
- `h5py = "^3.8.0"`
- `pyevtk = "^1.5.0"`
- `rich = "^13.3.2"`
- `tensorboard = "^2.12.0"`
- `tqdm = "^4.65.0"`
- `pandas = "^1.5.3"`
- `torch = "^1.13.1"`
- `vtk = "^9.2.6"`

My other personal project:

- `pymyplot = "^0.2.7"` (for plotting)
