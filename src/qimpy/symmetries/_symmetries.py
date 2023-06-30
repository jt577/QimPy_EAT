import qimpy as qp
import numpy as np
import torch
from typing import Union, Optional
from ._lattice import _get_lattice_point_group, _symmetrize_lattice, _symmetrize_matrix
from ._positions import (
    LabeledPositions,
    _get_space_group,
    _symmetrize_positions,
    _symmetrize_forces,
)
from ._grid import _check_grid_shape, _get_grid_shape


class Symmetries(qp.TreeNode):
    """Space group symmetries.
    Detects space group from lattice and ions, and provides methods to
    symmetrize properties such as positions, forces and densities."""

    lattice: qp.lattice.Lattice  #: Corresponding lattice vectors
    labeled_positions: Optional[LabeledPositions]  #: Positions determining space group
    tolerance: float  #: Relative error threshold in detecting symmetries
    n_sym: int  #: Number of space group operations
    rot: torch.Tensor  #: Rotations in fractional coordinates (n_sym x 3 x 3)
    trans: torch.Tensor  #: Translations in fractional coordinates (n_sym x 3)
    position_map: torch.Tensor  #: Index each position maps to (n_sym x n_positions)
    i_id: int  #: Index of identity operation within space group
    i_inv: list[int]  #: Indices of any inversion operations in space group

    symmetrize_lattice = _symmetrize_lattice
    symmetrize_matrix = _symmetrize_matrix
    symmetrize_positions = _symmetrize_positions
    symmetrize_forces = _symmetrize_forces
    check_grid_shape = _check_grid_shape
    get_grid_shape = _get_grid_shape

    def __init__(
        self,
        *,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        lattice: qp.lattice.Lattice,
        labeled_positions: Optional[LabeledPositions] = None,
        axes: dict[str, np.ndarray] = {},
        tolerance: float = 1e-6,
        override: Union[None, str, list, np.ndarray] = None,
    ) -> None:
        """Determine space group from `lattice` and `ions`.

        Parameters
        ----------
        lattice
            Lattice vectors for determining point group.
        positions
            List of covariant coordinates of eg. atoms to deduce space group.
        scalar_labels

        tolerance
            :yaml:`Threshold for detecting symmetries.`
        override
            :yaml:`Override with identity-only or manual list of operations.`
            By default (`override` = None), use automatically-detected symmetries.
            If `override` = 'identity', disable symmetries by only keeping identity.
            Otherwise, specify a `N x 4 x 3` array or nested list of `N` operations,
            each as a `4 x 3` matrix, where the first three rows are the rotation
            `rot` and the final row is the translation `trans` of the space group
            operation. The operations are specified in lattice coordinates, which
            means that `rot` must be composed only of integers.
        """
        super().__init__()
        self.lattice = lattice
        self.labeled_positions = labeled_positions
        self.tolerance = tolerance
        qp.log.info("\n--- Initializing Symmetries ---")
        rot, trans, position_map = Symmetries.detect(
            lattice, labeled_positions, axes, tolerance
        )

        # Down-select to manual symmetries (if any):
        if override is not None:
            if isinstance(override, str):
                assert override == "identity"
                sel = [Symmetries.find_identity(rot, trans, tolerance)]
            else:
                ops = torch.tensor(override, device=qp.rc.device)
                assert len(ops.shape) == 3
                assert ops.shape[-2:] == (4, 3)
                rot_in = ops[:, :3, :]
                trans_in = ops[:, 3, :]
                sel = Symmetries.find(rot, trans, rot_in, trans_in, tolerance)
            qp.log.info(f"Override: {len(sel)} space-group symmetries")
            rot = rot[sel]
            trans = trans[sel]
            position_map = position_map[sel]
            Symmetries.check_group(rot, trans, tolerance)

        # Set and enforce symmetries:
        self.rot = rot
        self.trans = trans
        self.position_map = position_map
        self.n_sym = self.rot.shape[0]
        self.report()
        self.enforce(
            lattice,
            None if (labeled_positions is None) else labeled_positions.positions,
        )
        self.i_id = Symmetries.find_identity(rot, trans, tolerance)
        self.i_inv = Symmetries.find_inversion(rot, tolerance)

    @staticmethod
    def detect(
        lattice: qp.lattice.Lattice,
        labeled_positions: Optional[LabeledPositions],
        axes: dict[str, np.ndarray],
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect space group: rotations, translations and mapping of ions."""
        # Point group:
        lattice_sym = _get_lattice_point_group(lattice.Rbasis, tolerance)
        n_point = lattice_sym.shape[0]
        qp.log.info(f"Found {n_point} point-group symmetries of the Bravais lattice")
        lattice_sym = Symmetries.reduce_axes(lattice_sym, lattice, axes, tolerance)
        # Space group:
        if labeled_positions is None:
            # Special case of no ions: space group = point group:
            rot = lattice_sym.clone().detach()
            trans = torch.zeros((lattice_sym.shape[0], 3), device=rot.device)
            position_map = torch.zeros(
                (lattice_sym.shape[0], 0), dtype=torch.long, device=rot.device
            )
            return rot, trans, position_map
        else:
            rot, trans, position_map = _get_space_group(
                lattice_sym, lattice, labeled_positions, tolerance
            )
            qp.log.info(f"Found {rot.shape[0]} space-group symmetries with basis")
            return rot, trans, position_map

    @staticmethod
    def reduce_axes(
        lattice_sym: torch.Tensor,
        lattice: qp.lattice.Lattice,
        axes: dict[str, np.ndarray],
        tolerance: float,
    ) -> torch.Tensor:
        """Reduce lattice (point group) symmetries by any global `axes`."""
        sym_axis = (lattice.Rbasis @ lattice_sym) @ lattice.invRbasis
        for axis_name, axis_np in axes.items():
            axis = torch.from_numpy(axis_np).to(qp.rc.device)
            sel = torch.where((sym_axis @ axis - axis).norm(dim=-1) < tolerance)[0]
            lattice_sym = lattice_sym[sel]
            sym_axis = sym_axis[sel]
            n_point = len(sel)
            qp.log.info(f"Reduced to {n_point} point-group symmetries with {axis_name}")
        return lattice_sym

    def report(self) -> None:
        """Print symmetry matrices."""
        for rot, trans in zip(
            self.rot.to(qp.rc.cpu, dtype=torch.int), self.trans.to(qp.rc.cpu)
        ):
            rot_str = ", ".join(qp.utils.fmt(row) for row in rot)
            qp.log.info(f"- [{rot_str}, {qp.utils.fmt(trans)}]")
        qp.log.debug("Ion map:\n" + qp.utils.fmt(self.position_map))

    def enforce(
        self, lattice: qp.lattice.Lattice, positions: Optional[torch.Tensor]
    ) -> None:
        """Enforce symmetries exactly on lattice and ions."""
        qp.log.info("Enforcing symmetries:")
        lattice.update(self.symmetrize_lattice(lattice.Rbasis))
        if positions is not None:
            positions_sym = self.symmetrize_positions(positions)
            rms = (positions_sym - positions).square().mean().sqrt()
            positions[:] = positions_sym
            qp.log.info(f"RMS change in fractional positions of ions: {rms:e}")

    @staticmethod
    def find_identity(rot: torch.Tensor, trans: torch.Tensor, tolerance: float) -> int:
        """Find index of identity matrix in space group."""
        id = torch.eye(3, device=qp.rc.device)
        id_diff = ((rot - id) ** 2).sum(dim=(1, 2)) + (trans**2).sum(dim=1)
        i_id = int(id_diff.argmin().item())
        if id_diff[i_id] > tolerance**2:
            raise ValueError("Identity operation not found in space group.")
        return i_id

    @staticmethod
    def find_inversion(rot: torch.Tensor, tolerance: float) -> list[int]:
        """Find list of indices of space group operations with inversion."""
        id = torch.eye(3, device=qp.rc.device)
        inv_diff = ((rot + id) ** 2).sum(dim=(1, 2))
        return torch.where(inv_diff < tolerance**2)[0].tolist()

    @staticmethod
    def find(
        rot: torch.Tensor,
        trans: torch.Tensor,
        rot_in: torch.Tensor,
        trans_in: torch.Tensor,
        tolerance: float,
    ) -> list[int]:
        """Find indices of operations `(rot_in, trans_in)` within `(rot, trans)`.
        Raise KeyError if any operations are not found within specified `tolerance`."""
        diff_mat = ((rot_in[:, None] - rot[None]) ** 2).sum(dim=(2, 3))
        diff_mat += ((trans_in[:, None] - trans[None]) ** 2).sum(dim=2)
        diff_min, indices = diff_mat.min(dim=1)
        indices_not_found = torch.where(diff_min > tolerance**2)[0].tolist()
        if indices_not_found:
            raise KeyError(f"Overrides at indices {indices_not_found}")
        assert len(torch.unique(indices)) == len(indices)  # Ensure no duplicates
        return indices

    @staticmethod
    def check_group(rot: torch.Tensor, trans: torch.Tensor, tolerance: float) -> None:
        """Check that operations `(rot, trans)` form a group.
        Raises exceptions if any group condition not satisfied."""
        i_id = Symmetries.find_identity(rot, trans, tolerance)

        # Construct multiplication of all pairs of operations:
        rot_pair = torch.einsum("iab, jbc -> ijac", rot, rot)
        trans_pair = torch.einsum("iab, jb -> ija", rot, trans) + trans[:, None, :]

        # Map multiplication results back onto original:
        diff_mat = ((rot_pair[:, :, None] - rot[None, None]) ** 2).sum(dim=(3, 4))
        diff_mat += ((trans_pair[:, :, None] - trans[None, None]) ** 2).sum(dim=3)
        diff_min, indices_pair = diff_mat.min(dim=2)
        indices_not_found = torch.nonzero(diff_min > tolerance**2)
        if len(indices_not_found) > 0:
            raise KeyError(f"Group not closed for pairs {indices_not_found.tolist()}")

        # Multiplication now closed (and associative by definition); check inverse:
        inv_not_found = torch.nonzero(abs(indices_pair - i_id).min(dim=1)[0])
        if len(inv_not_found) > 0:
            raise KeyError(f"Group inverse not found for {inv_not_found.tolist()}")

    @property
    def rot_cart(self) -> torch.Tensor:
        """Symmetry rotation matrices in Cartesian coordinates."""
        return self.lattice.Rbasis @ self.rot @ self.lattice.invRbasis
