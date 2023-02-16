from __future__ import annotations
import h5py
import qimpy as qp
import numpy as np
import torch
from typing import Any, Optional, NamedTuple


class Checkpoint(h5py.File):
    """Helper for checkpoint load/save from HDF5 files."""

    writable: bool  #: Whether file has been opened for writing

    def __init__(self, filename: str, *, mode: str = "r") -> None:
        super().__init__(filename, mode, driver="mpio", comm=qp.rc.comm)
        self.writable = not mode.startswith("r")
        mode_name = "writing:" if self.writable else "reading"
        qp.log.info(f"Opened checkpoint file '{filename}' for {mode_name}")

    def write_slice(
        self, dset: Any, offset: tuple[int, ...], data: torch.Tensor
    ) -> None:
        """Write a slice of data to dataset `dset` at offset `offset`
        from `data` (taking care of transfer to CPU if needed).
        Note that all of `data` is written, so pass in the slice to be
        written from current process.
        This may be called from any subset of MPI processes independently,
        as no metadata modification such as dataset creation is done here.
        """
        assert self.writable
        assert len(offset) == len(data.shape)
        assert len(offset) == len(dset.shape)
        index = tuple(
            slice(offset[i], offset[i] + s_i) for i, s_i in enumerate(data.shape)
        )
        dset[index] = data.to(qp.rc.cpu).numpy()

    def read_slice(
        self, dset: Any, offset: tuple[int, ...], size: tuple[int, ...]
    ) -> torch.Tensor:
        """Read a slice of data from data set `dset` in file,
        starting at `offset` and of length `size` in each dimension.
        Returns data on CPU or GPU as specified by `qimpy.rc.device`.
        """
        assert len(offset) == len(dset.shape)
        assert len(offset) == len(size)
        index = tuple(
            slice(offset[i], offset[i] + size[i]) for i, s_i in enumerate(dset.shape)
        )
        return torch.from_numpy(dset[index]).to(qp.rc.device)

    def create_dataset_real(
        self, path: str, shape: tuple[int, ...], dtype: torch.dtype = torch.float64
    ) -> Any:
        """Create a dataset at `path` suitable for a real array of size `shape`.
        Unlike `h5py.File.create_dataset`, this does not fail if the dataset exists,
        and instead removes the existing dataset. Additionally, `dtype` is translated
        from torch to numpy for convenience."""
        if path in self:
            del self[path]
        return self.create_dataset(path, shape=shape, dtype=qp.rc.np_type[dtype])

    def create_dataset_complex(
        self, path: str, shape: tuple[int, ...], dtype: torch.dtype = torch.complex128
    ) -> Any:
        """Create a dataset at `path` suitable for a complex array of size
        `shape`. This creates a real array with a final dimension of length 2.
        This format is used by :meth:`write_slice_complex` and
        :meth:`read_slice_complex`."""
        if path in self:
            del self[path]
        dtype_real = np.float64 if (dtype == torch.complex128) else np.float32
        return self.create_dataset(path, shape=(shape + (2,)), dtype=dtype_real)

    def write_slice_complex(
        self, dset: Any, offset: tuple[int, ...], data: torch.Tensor
    ) -> None:
        """Same as :meth:`write_slice`, but for complex `data`. Converts data
        to real storage compatible with :meth:`create_dataset_complex`"""
        assert torch.is_complex(data)
        self.write_slice(dset, offset + (0,), torch.view_as_real(data))

    def read_slice_complex(
        self, dset: Any, offset: tuple[int, ...], size: tuple[int, ...]
    ) -> torch.Tensor:
        """Same as :meth:`read_slice`, but for complex `data`. Converts data
        from real storage as created by :meth:`create_dataset_complex`
        to a complex tensor on output."""
        return torch.view_as_complex(self.read_slice(dset, offset + (0,), size + (2,)))


class CpPath(NamedTuple):
    """Combination of optional checkpoint and path within it.
    Useful as construction parameter for objects, to load data from
    checkpoint when available."""

    checkpoint: Optional[Checkpoint] = None  #: Checkpoint, if available.
    path: str = ""  #: Path within checkpoint

    def relative(self, relative_path: str) -> CpPath:
        """Create `CpPath` with path relative to current one.
        Specifically, `relative_path` is the path of the result relative
        to `self.path`.
        """
        return CpPath(
            checkpoint=self.checkpoint, path="/".join((self.path, relative_path))
        )

    def member(self, name: str) -> CpPath:
        """Member `name` at `path` within `checkpoint`, if present.
        Otherwise, return an empty `CpPath`.
        """
        path = "/".join((self.path, name))
        return (
            CpPath(checkpoint=self.checkpoint, path=path)
            if ((self.checkpoint is not None) and (path in self.checkpoint))
            else CpPath()
        )

    def __bool__(self):
        return self.checkpoint is not None

    @property
    def attrs(self):
        """Access attributes at `path` within `checkpoint`."""
        checkpoint, path = self
        assert checkpoint is not None
        if (path not in checkpoint) and checkpoint.writable:
            checkpoint.create_group(path)
        return checkpoint[path].attrs

    def write(self, name: str, data: torch.Tensor) -> str:
        """Write `data` available on all processes to `name` within current path.
        This is convenient for small tensors that are not split over MPI.
        For complex data, pass a real view that has a final dimension of length 2.
        Returns `name`, which is convenient for accumulating the names of
        written datasets during reporting."""
        checkpoint, path = self.relative(name)
        assert checkpoint is not None
        dset = checkpoint.create_dataset_real(path, data.shape, data.dtype)
        if qp.rc.is_head:
            dset[...] = data.to(qp.rc.cpu).numpy()
        return name

    def read(self, name: str, report: bool = True) -> torch.Tensor:
        """Read entire dataset from `name`, reporting to log if `report`."""
        checkpoint, path = self.relative(name)
        assert checkpoint is not None
        assert path in checkpoint
        if report:
            qp.log.info(f"Loading {name}")
        dset = checkpoint[path]
        return torch.from_numpy(dset[...]).to(qp.rc.device)

    def read_optional(self, name: str, report: bool = True) -> Optional[torch.Tensor]:
        """Handle optional dataset with `read`, returning None if not found."""
        try:
            return self.read(name, report)
        except AssertionError:
            return None


class CpContext(NamedTuple):
    """Identify from where/when a checkpoint is being written."""

    stage: str  #: Stage of calculation being checkpointed e.g. "geometry", "end"
    i_iter: int = 0  #: Iteration number of that stage
