from __future__ import annotations

import torch
import numpy as np

from qimpy import TreeNode, MPI, rc
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, TaskDivision, BufferView
from qimpy.profiler import stopwatch
from qimpy.transport.material import Material
from . import (
    Advect,
    BicubicPatch,
    parse_svg,
    QuadSet,
    SubQuadSet,
    subdivide,
    select_division,
)


class Geometry(TreeNode):
    """Geometry specification."""

    comm: MPI.Comm  #: Communicator for real-space split over patches
    quad_set: QuadSet  #: Original geometry specification from SVG
    sub_quad_set: SubQuadSet  #: Division into smaller quads for tuning parallelization
    patches: list[Advect]  #: Advection for each quad patch local to this process
    patch_division: TaskDivision  #: Division of patches over `comm`

    # v_F and N_theta should eventually be material paramteters
    def __init__(
        self,
        *,
        material: Material,
        svg_file: str,
        grid_spacing: float,
        grid_size_max: int = 0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        svg_file
            :yaml:`Path to an SVG file containing the input geometry.
        grid_spacing
            :yaml:`Maximum spacing between grid points anywhere in the geometry.`
            This is used to select the number of grid points in each domain.
        grid_size_max
            :yaml:`Maximum grid points per dimension after quad subdvision.`
            If 0, will be determined automatically from number of processes.
            Note that this only affects parallelization and performance by
            changing how data is divided into patches, and does not affect
            the accuracy of format of the output.
        """
        super().__init__()
        self.comm = process_grid.get_comm("r")
        self.quad_set = parse_svg(svg_file, grid_spacing)

        # Subdivide:
        if not grid_size_max:
            grid_size_max = select_division(self.quad_set, self.comm.size)
        self.sub_quad_set = subdivide(self.quad_set, grid_size_max)
        self.patch_division = TaskDivision(
            n_tot=len(self.sub_quad_set.quad_index),
            n_procs=self.comm.size,
            i_proc=self.comm.rank,
        )

        # Build an advect object for each sub-quad local to this process:
        self.patches = []
        mine = slice(self.patch_division.i_start, self.patch_division.i_stop)
        for i_quad, grid_start, grid_stop, adjacency in zip(
            self.sub_quad_set.quad_index[mine],
            self.sub_quad_set.grid_start[mine],
            self.sub_quad_set.grid_stop[mine],
            self.sub_quad_set.adjacency[mine],
        ):
            boundary = torch.from_numpy(self.quad_set.get_boundary(i_quad))
            transformation = BicubicPatch(boundary=boundary.to(rc.device))
            self.patches.append(
                Advect(
                    transformation=transformation,
                    grid_size_tot=tuple(self.quad_set.grid_size[i_quad]),
                    grid_start=grid_start,
                    grid_stop=grid_stop,
                    material=material,
                    need_reflector=(adjacency[:, 0] == -1),
                )
            )
        self.dt = self.comm.allreduce(
            min(patch.dt_max for patch in self.patches), op=MPI.MIN
        )

    @property
    def rho_list(self) -> list[torch.Tensor]:
        return [patch.rho for patch in self.patches]

    @rho_list.setter
    def rho_list(self, rho_list_new) -> None:
        for patch, rho_new in zip(self.patches, rho_list_new):
            patch.rho = rho_new

    @stopwatch
    def apply_boundaries(self, rho_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply all boundary conditions to `rho` and produce ghost-padded version.
        The list contains the data for each patch."""
        # Create padded version for all patches:
        out_list = []
        for patch, rho in zip(self.patches, rho_list):
            out = torch.zeros(patch.rho_padded_shape, device=rc.device)
            out[Advect.NON_GHOST, Advect.NON_GHOST] = rho
            out_list.append(out)

        # Populate ghost zones across patches where needed:
        requests = []
        pending_reads = []  # keep reference to data so that it doesn't deallocate
        pending_writes = []  # keep plans for writes till transfers complete
        for i_patch, adjacency in enumerate(self.sub_quad_set.adjacency):
            for i_edge, (other_patch, other_edge) in enumerate(adjacency):
                if other_patch < 0:
                    # Reflection (always local)
                    if self.patch_division.is_mine(i_patch):
                        i_patch_mine = i_patch - self.patch_division.i_start
                        reflector = self.patches[i_patch_mine].reflectors[i_edge]
                        assert reflector is not None
                        # Fetch the data in appropriate orientation:
                        ghost_data = rho_list[i_patch_mine][IN_SLICES[i_edge]]
                        if i_edge % 2:
                            ghost_data = ghost_data.swapaxes(0, 1)  # long axis first
                        # Reflect:
                        ghost_data = reflector(ghost_data)  # reciprocal space changes
                        ghost_data = ghost_data.flip(dims=(1,))  # flip short axis
                        # Store back:
                        if i_edge % 2:
                            ghost_data = ghost_data.swapaxes(0, 1)  # restore axis order
                        out_list[i_patch_mine][OUT_SLICES[i_edge]] = ghost_data
                else:
                    # Pass-through boundary:
                    read_mine = self.patch_division.is_mine(other_patch)
                    write_mine = self.patch_division.is_mine(i_patch)
                    tag = 4 * i_patch + i_edge  # unique for each message
                    if read_mine:
                        rho = rho_list[other_patch - self.patch_division.i_start]
                        ghost_data = rho[IN_SLICES[other_edge]]
                        delta_edge = other_edge - i_edge
                        if delta_edge % 2:
                            ghost_data = ghost_data.swapaxes(0, 1)
                        if flip_dims := FLIP_DIMS[delta_edge]:
                            ghost_data = ghost_data.flip(dims=flip_dims)
                        if not write_mine:
                            write_whose = self.patch_division.whose(i_patch)
                            ghost_data = ghost_data.contiguous()
                            pending_reads.append(ghost_data)  # hold till transfers done
                            requests.append(
                                self.comm.Isend(
                                    BufferView(ghost_data), write_whose, tag
                                )
                            )
                    if write_mine:
                        i_patch_mine = i_patch - self.patch_division.i_start
                        if read_mine:
                            out_list[i_patch_mine][OUT_SLICES[i_edge]] = ghost_data
                        else:
                            read_whose = self.patch_division.whose(other_patch)
                            ghost_data = torch.empty_like(
                                out_list[i_patch_mine][OUT_SLICES[i_edge]]
                            )
                            requests.append(
                                self.comm.Irecv(BufferView(ghost_data), read_whose, tag)
                            )
                            pending_writes.append(
                                [i_patch_mine, OUT_SLICES[i_edge], ghost_data]
                            )

        # Finish pending data transfers and writes:
        if requests:
            MPI.Request.Waitall(requests)
            for i_patch_mine, out_slice, ghost_data in pending_writes:
                out_list[i_patch_mine][out_slice] = ghost_data
        return out_list

    def next_rho_list(
        self,
        dt: float,
        rho_list_initial: list[torch.Tensor],
        rho_list_eval: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Ingredient of time step: compute rho_initial + dt * f(rho_eval)."""
        return [
            (rho_initial + patch.drho(dt, rho_eval))
            for rho_initial, rho_eval, patch in zip(
                rho_list_initial, self.apply_boundaries(rho_list_eval), self.patches
            )
        ]

    def time_step(self) -> None:
        """Second-order correct time step."""
        rho_list_init = self.rho_list
        rho_list_half = self.next_rho_list(0.5 * self.dt, rho_list_init, rho_list_init)
        self.rho_list = self.next_rho_list(self.dt, rho_list_init, rho_list_half)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        saved_list = [
            cp_path.write("vertices", torch.from_numpy(self.quad_set.vertices)),
            cp_path.write("quads", torch.from_numpy(self.quad_set.quads)),
            cp_path.write(
                "displacements", torch.from_numpy(self.quad_set.displacements)
            ),
            cp_path.write("adjacency", torch.from_numpy(self.quad_set.adjacency)),
            cp_path.write("grid_size", torch.from_numpy(self.quad_set.grid_size)),
            "q",
            "rho",
            "v",
        ]
        # MPI-split data:
        checkpoint, path = cp_path
        for i_quad, grid_size_np in enumerate(self.quad_set.grid_size):
            prefix = f"{path}/quad{i_quad}"
            grid_size = tuple(grid_size_np)
            dset_q = checkpoint.create_dataset_real(f"{prefix}/q", grid_size + (2,))
            dset_rho = checkpoint.create_dataset_real(f"{prefix}/rho", grid_size)
            dset_v = checkpoint.create_dataset_real(f"{prefix}/v", grid_size + (2,))
            for i_patch in np.where(self.sub_quad_set.quad_index == i_quad)[0]:
                if self.patch_division.is_mine(i_patch):
                    patch = self.patches[i_patch - self.patch_division.i_start]
                    offset = tuple(self.sub_quad_set.grid_start[i_patch])
                    checkpoint.write_slice(dset_q, offset + (0,), patch.q)
                    checkpoint.write_slice(dset_rho, offset, patch.density)
                    checkpoint.write_slice(dset_v, offset + (0,), patch.velocity)
        return saved_list


# Constants for edge data transfer:
IN_SLICES = [
    (slice(None), Advect.GHOST_L),
    (Advect.GHOST_R, slice(None)),
    (slice(None), Advect.GHOST_R),
    (Advect.GHOST_L, slice(None)),
]  #: input slice for each edge orientation during edge communication

OUT_SLICES = [
    (Advect.NON_GHOST, Advect.GHOST_L),
    (Advect.GHOST_R, Advect.NON_GHOST),
    (Advect.NON_GHOST, Advect.GHOST_R),
    (Advect.GHOST_L, Advect.NON_GHOST),
]  #: output slice for each edge orientation during edge communication

FLIP_DIMS = [(0, 1), (0,), None, (1,)]  #: which dims to flip during edge transfer
