from __future__ import annotations

import os
from typing import Optional, Union
import pathlib
import re
import json

import numpy as np
import torch

from qimpy import TreeNode, log, rc, dft
from qimpy.io import fmt, CheckpointPath, CheckpointContext
from qimpy.math import cis
from qimpy.lattice import Lattice
from qimpy.symmetries import LabeledPositions
from qimpy.grid import FieldH
from . import Pseudopotential, _ions_projectors, _ions_atomic, _ions_update


class Ions(TreeNode):
    """Ionic system: ionic geometry and pseudopotentials."""

    lattice: Lattice  #: Lattice vectors of corresponding unit cell
    fractional: bool  #: use fractional coordinates in input/output
    n_ions: int  #: number of ions
    n_types: int  #: number of distinct ion types
    n_ions_type: list[int]  #: number of ions of each type
    symbols: list[str]  #: symbol for each ion type
    slices: list[slice]  #: slice to get each ion type
    pseudopotentials: list[Pseudopotential]  #: pseudopotential for each type
    positions: torch.Tensor  #: fractional positions of each ion (n_ions x 3)
    velocities: Optional[torch.Tensor]  #: Cartesian velocities of each ion (n_ions x 3)
    types: torch.Tensor  #: type of each ion (n_ions, int)
    Q: Optional[torch.Tensor]  #: initial / Lowdin charge (oxidation state) for each ion
    M: Optional[torch.Tensor]  #: initial / Lowdin magnetic moment for each ion
    Z: torch.Tensor  #: charge of each ion type (n_types, float)
    Z_tot: float  #: total ionic charge
    rho_tilde: FieldH  #: ionic charge density (uses coulomb.ion_width)
    Vloc_tilde: FieldH  #: local potential due to ions (including from rho)
    n_core_tilde: FieldH  #: partial core electronic density (for XC)
    beta: dft.electrons.Wavefunction  #: pseudopotential projectors (split-basis only)
    beta_full: Optional[dft.electrons.Wavefunction]  #: full-basis version of `beta`
    beta_version: int  #: version of `beta` to invalidate cached projections
    D_all: torch.Tensor  #: nonlocal pseudopotential matrix (all atoms)
    dEtot_drho_basis: float  #: dE/d(basis function density) for Pulay correction
    effective_mix: list[Optional[dict]]  #: mix dictionaries for effective ions
    pulay_mixGrad: Optional[dict]  #: Pulay EAT gradient

    _get_projectors = _ions_projectors.get_projectors
    _projectors_grad = _ions_projectors.projectors_grad
    get_atomic_orbital_index = _ions_atomic.get_atomic_orbital_index
    get_atomic_orbitals = _ions_atomic.get_atomic_orbitals
    get_atomic_density = _ions_atomic.get_atomic_density
    update = _ions_update.update
    accumulate_geometry_grad = _ions_update.accumulate_geometry_grad
    _collect_ps_matrix = _ions_update.collect_ps_matrix

    def __init__(
        self,
        *,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        lattice: Lattice,
        fractional: bool = True,
        coordinates: Optional[list] = None,
        pseudopotentials: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """Initialize geometry and pseudopotentials.

        Parameters
        ----------
        fractional
            :yaml:`Whether to use fractional coordinates for input/output.`
            Note that positions in memory and checkpoint files are always fractional.
        coordinates
            :yaml:`List of [symbol, x, y, z, args] for each ion in unit cell.`
            Here, `symbol` is the chemical symbol of the element, while
            `x`, `y` and `z` are positions in fractional or Cartesian
            coordinates for `fractional` = True or False respectively.
            Optional `args` is a dictionary of additional per-ion
            parameters that may include:

                * `Q`: initial oxidation state / charge for the ion.
                  This can be used to tune initial density for LCAO,
                  or to specify charge disproportionation for symmetry detection.
                * `M`: initial magnetic moment for the ion, which would be a
                  single Mz or [Mx, My, Mz] depending on if the calculation
                  is spinorial. Only specify in spin-polarized calculations.
                  This can be used to tune initial magnetization for LCAO,
                  or to specify magnetic order for symmetry detection.
                * `v`: initial velocities [vx, vy, vz] in Cartesian coordinates.
                * Relaxation constraints: TODO

            Ions of the same type (symbol) must be specified consecutively.
        pseudopotentials
            :yaml:`Pseudopotential filenames or filename templates.`
            Templates for families of pseudopotentials are specified by
            including a $ID in the name which is replaced by the chemical
            symbol of the element. The list of specified file names and
            templates is processed in order, and the first match for
            each element takes precedence.
        """
        super().__init__()
        log.info("\n--- Initializing Ions ---")
        self.lattice = lattice
        self.fractional = fractional
        if checkpoint_in:
            self._read_checkpoint(checkpoint_in)
        else:
            self._process_coordinates(coordinates)
        self.report(report_grad=False)  # reports read-in coordinates

        # Initialize pseudopotentials:
        if pseudopotentials is None:
            pseudopotentials = []
        if isinstance(pseudopotentials, str):
            pseudopotentials = [pseudopotentials]
        prefixes = [""]
        if "QIMPY_PSEUDO_DIR" in os.environ:
            prefixes.extend(os.environ["QIMPY_PSEUDO_DIR"].split(":"))
        
        self.pseudopotential_filenames = []
        self.pseudopotentials = []
        pseudopotentials_encountered = {}
        for i, symbol in enumerate(self.symbols):
            if self.effective_mix[i] is None:
                # Standard ion: load as usual
                filename = self._get_pseudopotential_filename(symbol, pseudopotentials, prefixes)
                self.pseudopotential_filenames.append(filename)
                if symbol not in pseudopotentials_encountered.keys():
                    pseudopotential_current = Pseudopotential(filename)
                    pseudopotentials_encountered[symbol] = pseudopotential_current
                else:
                    pseudopotential_current = pseudopotentials_encountered[symbol]
                self.pseudopotentials.append(pseudopotential_current)
            else:
                # Effective ion: load pseudopotentials for each element in the mix.
                effective_mix = self.effective_mix[i]  # e.g. {'Ag': 0.5, 'Cu': 0.5}
                mix_filenames = []
                mix_potentials = []
                for elem in effective_mix.keys():
                    filename = self._get_pseudopotential_filename(elem, pseudopotentials, prefixes)
                    mix_filenames.append(filename)
                    if elem not in pseudopotentials_encountered.keys():
                        pseudopotential_current = Pseudopotential(filename)
                        pseudopotentials_encountered[elem] = pseudopotential_current
                    else:
                        pseudopotential_current = pseudopotentials_encountered[elem]
                    mix_potentials.append(pseudopotential_current)
                self.pseudopotential_filenames.append(mix_filenames)
                self.pseudopotentials.append(mix_potentials)
        
        self.beta_version = 0

        # Calculate total ionic charge (needed for number of electrons):
        # For standard ions, use the pseudopotential’s Z.
        # For effective ions, compute a weighted sum of the Z values.
        Z_list = []
        self.effective_Z_list = []
        for i, ps in enumerate(self.pseudopotentials):
            if isinstance(ps, list):
                # Effective ion: mix is non-None.
                mix = self.effective_mix[i]  # e.g. {'Ag': 0.5, 'Cu': 0.5}
                effective_Z_list = [p.Z for p in ps]
                effective_Z = sum(mix[elem] * p.Z for elem, p in zip(mix.keys(), ps))
                Z_list.append(effective_Z)
                self.effective_Z_list.append(torch.tensor(effective_Z_list, device=rc.device))
            else:
                # Standard ion.
                self.effective_Z_list.append(ps.Z)
                Z_list.append(ps.Z)
        self.Z = torch.tensor(Z_list, device=rc.device)
        self.Z_tot = self.Z[self.types].sum().item()
        log.info(f"\nTotal ion charge, Z_tot: {self.Z_tot:g}")

    @property
    def n_replicas(self) -> int:
        """Number of replicas used in future NEB / phonons support."""
        return 1

    def _process_coordinates(self, coordinates: Optional[list]) -> None:
        if coordinates is None:
            coordinates = []
        assert isinstance(coordinates, list)
        symbols: list[str] = []  # For standard ions, the element symbol for each ion type
        effective_mix: list[Optional[dict]] = []  # For effective ions, store the mix dict
        n_types = 0  # number of ion types encountered so far
        n_mix = 0  # number of effective ions encountered so far
        positions = []  # position of each ion
        types = []  # type of each ion (index into symbols)
        velocities = []  # initial velocities
        Q_initial = []  # initial charge / oxidation states
        M_initial = []  # initial magnetic moments
        moveable = [] # moveable flag for each ion
        for coord in coordinates:
            # Check for optional attributes:
            if len(coord) == 4:
                attrib = {}
            elif len(coord) == 5:
                attrib = coord[4]
                if not isinstance(attrib, dict):
                    raise ValueError("ion attributes must be a dict")
            else:
                raise ValueError("each ion must be 4 entries + optional dict")
            # Determine if the ion is effective or standard:
            if isinstance(coord[0], dict):
                # Effective ion case: the first element is a mix dictionary
                mix = coord[0]  # e.g., {'Ag': 0.5, 'Cu': 0.5}
                symbol = f"mix{n_mix}"  # symbol indicating an effective atom
                effective_mix.append(mix)
                n_mix += 1
            else:
                symbol = str(coord[0])
                effective_mix.append(None)
            # Add new symbol or append to existing:
            if (not symbols) or (symbol != symbols[-1]):
                symbols.append(symbol)
                n_types += 1
            # Add type and position of current ion:
            types.append(n_types - 1)
            positions.append([float(x) for x in coord[1:4]])
            velocities.append(attrib.get("v", None))
            Q_initial.append(attrib.get("Q", None))
            M_initial.append(attrib.get("M", None))
            moveable.append(attrib.get("moveable", 0))

        # Convert to tensors before storing in class object:
        self.positions = (
            torch.tensor(positions, device=rc.device)
            if positions
            else torch.empty((0, 3), device=rc.device)
        )
        if not self.fractional:
            # Convert Cartesian input to fractional coordinates:
            self.positions = self.positions @ self.lattice.invRbasisT
        self.positions.grad = None
        self.types = torch.tensor(types, device=rc.device, dtype=torch.long)
        self.symbols = symbols
        self.effective_mix = effective_mix # store the effective atom dictionaries
        self._set_counts_slices()  # uses types and symbols, sets n_ions, slices etc.
        self.velocities = self._process_velocities(velocities)
        self.Q = self._process_Q_initial(Q_initial)
        self.M = self._process_M_initial(M_initial)
        self.moveable = torch.tensor(moveable, device=rc.device) # store whether ion is moveable

    def _process_velocities(self, velocities: list) -> Optional[torch.Tensor]:
        """Fill in missing velocities (if any specified)."""
        if velocities.count(None) == self.n_ions:
            return None  # no velocities specified
        v_default = [0.0, 0.0, 0.0]
        return torch.tensor(
            [(v_default if (v is None) else v) for v in velocities],
            device=rc.device,
            dtype=torch.double,
        )

    def _process_Q_initial(self, Q_initial: list) -> Optional[torch.Tensor]:
        """Fill in missing oxidation states (if any specified)."""
        if Q_initial.count(None) == self.n_ions:
            return None  # no charge specified
        return torch.tensor(
            [(0.0 if (Q is None) else Q) for Q in Q_initial],
            device=rc.device,
            dtype=torch.double,
        )

    def _process_M_initial(self, M_initial: list) -> Optional[torch.Tensor]:
        """Fill in missing magnetizations (if any specified)."""
        M_lengths = set(
            [(len(M) if isinstance(M, list) else 1) for M in M_initial if M]
        )
        if not M_lengths:
            return None
        if len(M_lengths) > 1:
            raise ValueError("All M must be same type: 3-vector or scalar")
        M_length = next(iter(M_lengths))
        assert (M_length == 1) or (M_length == 3)
        M_default = [0.0, 0.0, 0.0] if (M_length == 3) else 0.0
        return torch.tensor(
            [(M_default if (M is None) else M) for M in M_initial],
            device=rc.device,
            dtype=torch.double,
        )

    def _get_pseudopotential_filename(
        self, symbol: str, pseudopotentials: list[str], prefixes: list[str]
    ) -> str:
        """Find exact pseudopotential filename for symbol from filename templates."""
        symbol_variants = [symbol.lower(), symbol.upper(), symbol.capitalize()]
        for ps_name in pseudopotentials:
            if ps_name.count("$ID"):
                # wildcard syntax
                for prefix in prefixes:
                    template = os.path.join(prefix, ps_name) if prefix else ps_name
                    for symbol_variant in symbol_variants:
                        filename_test = template.replace("$ID", symbol_variant)
                        if pathlib.Path(filename_test).exists():
                            return filename_test  # found
            else:
                # specific filename
                basename = pathlib.PurePath(ps_name).stem
                ps_symbol = re.split(r"[_\-\.]+", basename)[0]
                if ps_symbol in symbol_variants:
                    for prefix in prefixes:
                        filename = os.path.join(prefix, ps_name) if prefix else ps_name
                        if pathlib.Path(filename).exists():
                            return filename
                    raise FileNotFoundError(ps_name)
        raise ValueError(f"no pseudopotential found for {symbol}")

    def report(self, report_grad: bool) -> None:
        """Report ionic positions / attributes, and optionally forces if `report_grad`."""
        log.info(
            f"{self.n_ions} total ions of {self.n_types} types; positions:"
            f"  # in {'fractional' if self.fractional else 'Cartesian [a0]'} coordinates"
        )
        # Fetch to CPU in required coordinate system for reporting:
        positions = (
            (
                self.positions
                if self.fractional
                else self.positions @ self.lattice.Rbasis.T
            )
            .to(rc.cpu)
            .numpy()
        )
        types = self.types.to(rc.cpu).numpy()
        v = None if (self.velocities is None) else self.velocities.to(rc.cpu).numpy()
        Q = None if (self.Q is None) else self.Q.to(rc.cpu).numpy()
        M = None if (self.M is None) else self.M.to(rc.cpu).numpy()
        moveable = None if (self.moveable is None) else self.moveable.to(rc.cpu).numpy()
        any_attribs = not ((v is None) and (Q is None) and (M is None) and (moveable is None))
        for i_ion, (pos_x, pos_y, pos_z) in enumerate(positions):
            if any_attribs:
                # Generate attribute string:
                attribs = {}
                if v is not None:
                    vx, vy, vz = v[i_ion]
                    attribs["v"] = f"[{vx:+.3e}, {vy:+.3e}, {vz:+.3e}]"
                if Q is not None:
                    attribs["Q"] = f"{Q[i_ion]:+.5f}"
                if M is not None:
                    attribs["M"] = fmt(M[i_ion], floatmode="fixed", precision=5)
                if moveable is not None:
                    attribs["moveable"] = f"{moveable[i_ion]}"
                attrib_str = str(attribs).replace("'", "")
                attrib_str = f", {attrib_str}"
            else:
                attrib_str = ""
            # Report:
            log.info(
                f"- [{self.symbols[types[i_ion]]},"
                f" {pos_x:11.8f}, {pos_y:11.8f}, {pos_z:11.8f}{attrib_str}]"
            )

        # Report forces / stresses if requested:
        if report_grad:
            forces = self.forces.detach().to(rc.cpu).numpy()
            log.info("\nforces:  # in Cartesian [Eh/a0] coordinates")
            for type_i, (fx, fy, fz) in zip(types, forces):
                log.info(
                    f"- [{self.symbols[type_i]}, {fx:11.8f}, {fy:11.8f}, {fz:11.8f}]"
                )
        log.info("")

    def translation_phase(
        self, iG: torch.Tensor, atom_slice: slice = slice(None)
    ) -> torch.Tensor:
        """Get translation phases at `iG` for a slice of atoms.
        The result has atoms as the final dimension; summing over that
        dimension yields the structure factor corresponding to these atoms.
        """
        return cis((-2 * np.pi) * (iG @ self.positions[atom_slice].T))

    @property
    def n_projectors(self) -> int:
        """Total number of pseudopotential projectors."""
        n_proj = 0 # number of projectors
        for i_ps, ps in enumerate(self.pseudopotentials):
            # Chek if pseudopotential corresponds to an effective atom
            if isinstance(ps, list):
                for p in ps:
                    # Just add total number of projectors for all constituent elements in pseudopotential
                    n_proj += p.n_projectors * self.n_ions_type[i_ps] 
            else:
                n_proj += ps.n_projectors * self.n_ions_type[i_ps]
        
        return n_proj

    @property
    def n_orbital_projectors(self) -> int:
        """Total number of projectors used to generate atomic orbitals."""
        n_orb_proj = 0 # number of orbital projectors
        for i_ps, (ps, mix) in enumerate(zip(self.pseudopotentials, self.effective_mix)):
            # Chek if pseudopotential corresponds to an effective atom
            if isinstance(ps, list):
                max_mix_index = np.argmax(mix.values())
                for ip, p in enumerate(ps):
                    if ip == max_mix_index:
                        # Just count the element with largest mix coefficient
                        n_orb_proj += p.n_orbital_projectors * self.n_ions_type[i_ps] 
            else:
                n_orb_proj += ps.n_orbital_projectors * self.n_ions_type[i_ps]
        
        return n_orb_proj

    def n_atomic_orbitals(self, n_spinor: int) -> int:
        """Total number of atomic orbitals. This depends on the number
        of spinorial components `n_spinor`."""
        n_orb = 0 # number of atomic orbitals
        for i_ps, (ps, mix) in enumerate(zip(self.pseudopotentials, self.effective_mix)):
            # Chek if pseudopotential corresponds to an effective atom
            if isinstance(ps, list):
                max_mix_index = np.argmax(mix.values())
                for ip, p in enumerate(ps):
                    if ip == max_mix_index:
                        # Just count the element with largest mix coefficient
                        n_orb += p.n_atomic_orbitals(n_spinor) * self.n_ions_type[i_ps] 
            else:
                n_orb += ps.n_atomic_orbitals(n_spinor) * self.n_ions_type[i_ps]
        
        return n_orb

    @property
    def forces(self) -> torch.Tensor:
        """Cartesian forces [in Eh/a0] of each ion (n_ions x 3).
        This converts from the fractional energy gradient in `positions.grad`,
        which should have already been calculated.
        """
        assert self.positions.grad is not None
        return -self.positions.grad.detach() @ self.lattice.invRbasis

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        if any(item is not None for item in self.effective_mix):
            # Serialize the mixed list of lists/strings to a JSON string
            serialized_data = json.dumps(self.pseudopotential_filenames)
            # Save using write_str, which handles strings
            cp_path.write_str("pseudopotentials", serialized_data)
        else:
            cp_path.attrs["pseudopotentials"] = self.pseudopotential_filenames

        saved_list = [
            cp_path.write_str("symbols", ",".join(self.symbols)),
            cp_path.write("types", self.types),
            cp_path.write("positions", self.positions.detach()),
        ]
        if any(item is not None for item in self.effective_mix):
            # Store effective mix
            serialized_data = json.dumps(self.effective_mix)
            cp_path.write_str("effective_mix", serialized_data)
            # Store the total energy EAT gradient
            self.E_mixGrad = {}
            for (symbol, mix) in zip(self.symbols, self.effective_mix):
                if mix is not None:
                    self.E_mixGrad[symbol] = (np.array(self.Eloc_mixGrad[symbol]) + np.array(self.Enl_mixGrad[symbol]) 
                                            + np.array(self.Eewald_mixGrad[symbol]) + np.array(self.Epulay_mixGrad[symbol])
                                            + np.array(self.Efillings_mixGrad[symbol])).tolist()
            # Use json to serialize the dictionary
            serialized_data = json.dumps(self.E_mixGrad)
            cp_path.write_str("E_mixGrad", serialized_data)
        if self.velocities is not None:
            saved_list.append(cp_path.write("velocities", self.velocities))
        if self.positions.grad is not None:
            saved_list.append(cp_path.write("forces", self.forces))
        if self.Q is not None:
            saved_list.append(cp_path.write("Q", self.Q))
        if self.M is not None:
            saved_list.append(cp_path.write("M", self.M))
        if self.moveable is not None:
            saved_list.append(cp_path.write("moveable", self.moveable))
        return saved_list

    def _read_checkpoint(self, cp_path: CheckpointPath) -> None:
        symbol_str = cp_path.read_str("symbols")
        self.symbols = symbol_str.split(",") if symbol_str else list[str]()
        self.types = cp_path.read("types")
        self.positions = cp_path.read("positions")
        # Read in effective_mix as a JSON string and convert to a list of dictionaries
        try:
            data_str = cp_path.read_str("effective_mix")
            self.effective_mix = json.loads(data_str)
        except:
            self.effective_mix = [None] * len(self.symbols)

        self.velocities = cp_path.read_optional("velocities")
        forces = cp_path.read_optional("forces")
        if forces is not None:
            self.positions.grad = -forces @ self.lattice.Rbasis
        self.Q = cp_path.read_optional("Q")
        self.M = cp_path.read_optional("M")
        self.moveable = cp_path.read_optional("moveable")
        self._set_counts_slices()

    def _set_counts_slices(self) -> None:
        """Update all counts and slices based on `self.types`"""
        unique, counts = torch.unique_consecutive(self.types, return_counts=True)
        self.n_ions = len(self.types)
        self.n_types = len(unique)
        self.n_ions_type = counts.tolist()
        if len(set(self.symbols)) < self.n_types:
            raise ValueError("coordinates must group ions of same type together")
        slice_ends = counts.cumsum(dim=0).tolist()
        self.slices = [
            slice(slice_end - slice_len, slice_end)
            for slice_end, slice_len in zip(slice_ends, self.n_ions_type)
        ]

    @property
    def labeled_positions(self) -> Optional[LabeledPositions]:
        """Labeled positions for symmetry detection."""
        if not self.n_ions:
            return None
        scalars = [self.types.to(torch.double)]  # atom type always a label
        pseudovectors: Optional[torch.Tensor] = None  # only if vector magnetization
        if self.Q is not None:
            scalars.append(self.Q)
        if self.M is not None:
            if len(self.M.shape) == 1:  # scalar magnetization
                scalars.append(self.M)
            else:  # vector magnetization
                pseudovectors = self.M[None]
        return LabeledPositions(
            positions=self.positions,
            scalars=torch.stack(scalars),
            pseudovectors=pseudovectors,
        )
