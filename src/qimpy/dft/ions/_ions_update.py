from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, dft
from qimpy.profiler import stopwatch
from qimpy.grid import FieldH
from qimpy.dft import ions
from qimpy.math import RadialFunction
from qimpy.math.quintic_spline import Interpolator


def update(self: ions.Ions, system: dft.System) -> None:
    """Update ionic potentials, projectors and energy components.
    The grids used for the potentials are derived from `system`,
    and the energy components are stored within `system.E`.
    """
    grid = system.grid
    n_densities = system.electrons.n_densities
    self.rho_tilde = FieldH(grid)  # initialize zero ionic charge
    self.Vloc_tilde = FieldH(grid)  # initialize zero local potential
    self.n_core_tilde = FieldH(
        grid, shape_batch=(n_densities,)  # initialize zero core density
    )
    if not self.n_ions:
        nk_mine = system.electrons.kpoints.division.n_mine
        n_basis_mine = system.electrons.basis.division.n_mine
        self.beta = dft.electrons.Wavefunction(
            system.electrons.basis,
            coeff=torch.empty(
                (1, nk_mine, 0, 1, n_basis_mine),
                dtype=torch.complex128,
                device=rc.device,
            ),
        )
        self.D_all = torch.empty((0, 0), dtype=torch.complex128, device=rc.device)
        if system.electrons.need_full_projectors:
            n_basis_tot = system.electrons.basis.n_tot
            self.beta_full = dft.electrons.Wavefunction(
                system.electrons.basis,
                coeff=torch.empty(
                    (1, nk_mine, 0, 1, n_basis_tot),
                    dtype=torch.complex128,
                    device=rc.device,
                ),
            )
        return  # no contributions below if no ions!
    system.energy["Eewald"] = system.coulomb.ewald(self.positions, self.Z[self.types])
    system.energy["Epulay"] = _update_pulay(self, system.electrons.basis)
    # EAT gradients
    if any(item is not None for item in self.effective_mix):
        # Update pulay EAT gradient
        self.Epulay_mixGrad = _update_pulay_mixGrad(self, system.electrons.basis)
        # Update ewald EAT gradient
        Z_list = []
        for i, ps in enumerate(self.pseudopotentials):
            if isinstance(ps, list):
                # Effective ion: mix is non-None.
                mix = self.effective_mix[i]  # e.g. {'Ag': 0.5, 'Cu': 0.5}
                mix_weights = torch.tensor([weights for weights in mix.values()], device=rc.device)
                effective_Z_list = torch.tensor([p.Z for p in ps], device=rc.device)
                effective_Z = (mix_weights * effective_Z_list).sum()
                Z_list.append(effective_Z)
            else:
                # Standard ion.
                Z_list.append(ps.Z)
        Zeff = torch.tensor(Z_list, device=rc.device, requires_grad=True)
        # Now compute the Ewald energy using Zeff.
        E = system.coulomb.ewald(self.positions, Zeff[self.types])  # ewald_periodic is an instance of EwaldPeriodic
        # Get gradient dE/dZeff:
        grad_mix_Zeff, = torch.autograd.grad(E, Zeff)
        # Now by chain rule, the gradient with respect to each mixing weight is:
        self.Eewald_mixGrad = {}
        for i, (symbol, mix, eff_Z_list) in enumerate(zip(self.symbols, self.effective_mix, self.effective_Z_list)):
            if mix is not None:
                grad_mix_temp = []
                for Z_entry in eff_Z_list:
                    grad_mix_temp.append((grad_mix_Zeff[i] * Z_entry).item())
                self.Eewald_mixGrad[symbol] = grad_mix_temp
        # Update fillings EAT gradient
        self.Efillings_mixGrad = {}
        fillings = system.electrons.fillings
        for i, (symbol, mix, eff_Z_list) in enumerate(zip(self.symbols, self.effective_mix, self.effective_Z_list)):
            if mix is not None:
                grad_mix_temp = []
                for Z_entry in eff_Z_list:
                    grad_mix_temp.append((fillings.mu * Z_entry).item())
                self.Efillings_mixGrad[symbol] = grad_mix_temp

    # Update ionic densities and potentials:
    _LocalTerms(self, system).update()

    # Update pseudopotential matrix and projectors:
    self._collect_ps_matrix(system.electrons.n_spinor)
    if system.electrons.need_full_projectors:
        beta_full = self._get_projectors(system.electrons.basis, full_basis=True)
        self.beta_full = beta_full
        self.beta = beta_full[..., system.electrons.basis.mine]
    else:
        self.beta = self._get_projectors(system.electrons.basis)
        self.beta_full = None
    self.beta_version += 1  # will auto-invalidate cached projections


def accumulate_geometry_grad(self: ions.Ions, system: dft.System) -> None:
    """Accumulate geometry gradient contributions of total energy.
    Each contribution is accumulated to a `grad` attribute,
    only if the corresponding `requires_grad` is enabled.
    Force contributions are collected in `self.positions.grad`.
    Stress contributions are collected in `system.lattice.grad`.
    Assumes Hellman-Feynman theorem, i.e., electronic system must be converged.
    Note that this invokes `system.electrons.accumulate_geometry_grad`
    as a dependency and therefore includes electronic force / stress contributions.
    """
    # Electronic contributions (direct and through ion-dependent scalar fields, beta):
    self.beta.requires_grad_(True)  # don't zero-initialize to save memory
    self.rho_tilde.requires_grad_(True, clear=True)
    self.Vloc_tilde.requires_grad_(True, clear=True)
    self.n_core_tilde.requires_grad_(True, clear=True)
    system.electrons.accumulate_geometry_grad(system)

    # Ionic contributions:
    if self.n_ions:
        self._projectors_grad(self.beta)
        _LocalTerms(self, system).update_grad()
        system.coulomb.ewald(self.positions, self.Z[self.types])
        if system.lattice.requires_grad and self.dEtot_drho_basis:
            # Pulay stress:
            eye3 = torch.eye(3, device=rc.device)
            system.lattice.grad += (
                self.dEtot_drho_basis
                * system.electrons.basis.n_avg_weighted
                / system.lattice.volume
            ) * eye3

    # Clean up intermediate gradients:
    self.beta.requires_grad_(False, clear=True)
    self.rho_tilde.requires_grad_(False, clear=True)
    self.Vloc_tilde.requires_grad_(False, clear=True)
    self.n_core_tilde.requires_grad_(False, clear=True)

    # Symmetrize:
    assert self.positions.grad is not None
    self.positions.grad = system.symmetries.symmetrize_forces(self.positions.grad)
    if system.lattice.requires_grad:
        system.lattice.grad = system.symmetries.symmetrize_matrix(
            0.5 * (system.lattice.grad + system.lattice.grad.transpose(-2, -1))
        )


class _LocalTerms:
    """
    Handle generation and gradient propagation of ionic scalar fields (local terms).
    """

    @stopwatch(name="Ions.LocalTerms.init")
    def __init__(self, ions: ions.Ions, system: dft.System):
        self.ions = ions
        self.system = system

        # Prepare interpolator for grid:
        grid = system.grid
        self.iG = grid.get_mesh("H").to(torch.double)  # half-space
        G = self.iG @ grid.lattice.Gbasis.T
        Gsq = G.square().sum(dim=-1)
        Gmag = Gsq.sqrt()
        self.Ginterp = Interpolator(Gmag, RadialFunction.DG)

        # Collect structure factor and radial coefficients:
        Vloc_coeff = []
        n_core_coeff = []
        Gmax = grid.get_Gmax()
        ion_width = system.coulomb.ion_width
        effective_mix = ions.effective_mix

        Vloc_coeffs_mixGrad = {} # Assemble Vloc_tilde gradient for effective atoms
        for i_type, (ps, symbol) in enumerate(zip(ions.pseudopotentials, ions.symbols)):
            # Chek if pseudopotential corresponds to an effective atom
            if isinstance(ps, list):
                Vloc_coeff_mix_list = []
                Vloc_coeff_grad_temp = []
                n_core_coeff_mix_list = []
                for mix_weight, p in zip(effective_mix[i_type].values(), ps):
                    p.update(Gmax, ion_width, system.electrons.comm)
                    Vloc_coeff_mix_list.append(p.Vloc.f_tilde_coeff * mix_weight)
                    Vloc_coeff_grad_temp.append(p.Vloc.f_tilde_coeff)
                    n_core_coeff_mix_list.append(p.n_core.f_tilde_coeff * mix_weight)
                Vloc_coeff.append(sum(Vloc_coeff_mix_list))  # Append weighted sum of Vloc fourier coefficients
                Vloc_coeffs_mixGrad[symbol] = Vloc_coeff_grad_temp # Each entry in mix grad dict is a list of Vloc_tilde gradients for each species of the effective atom
                n_core_coeff.append(sum(n_core_coeff_mix_list))  # Append weighted sum of n_core fourier coefficients
            else:
                ps.update(Gmax, ion_width, system.electrons.comm)
                Vloc_coeff.append(ps.Vloc.f_tilde_coeff)
                n_core_coeff.append(ps.n_core.f_tilde_coeff)
        self.Vloc_coeff = torch.hstack(Vloc_coeff)
        self.Vloc_coeffs_mixGrad = Vloc_coeffs_mixGrad
        self.n_core_coeff = torch.hstack(n_core_coeff)

        # Assemble rho_kernel gradient for effective atoms
        self.rho_kernel_mixGrad = {} 
        for i_type, (ps, symbol, Z_list) in enumerate(zip(ions.pseudopotentials, ions.symbols, ions.effective_Z_list)):
            # Chek if pseudopotential corresponds to an effective atom
            if isinstance(ps, list):
                rho_kernel_grad_temp = []
                for mix_weight, p, Z_eff in zip(effective_mix[i_type].values(), ps, Z_list):
                    Z_eff_tensor = torch.zeros(1, device=rc.device)
                    Z_eff_tensor[0] = Z_eff
                    rho_kernel_grad_temp_comp = -Z_eff_tensor.view(-1, 1, 1, 1) * torch.exp(
                                                (-0.5 * (ion_width**2)) * Gsq
                                        )
                    rho_kernel_grad_temp.append(rho_kernel_grad_temp_comp)
                self.rho_kernel_mixGrad[symbol] = rho_kernel_grad_temp

        # Assemble rho_kernel for ordinary atoms
        self.rho_kernel = -ions.Z.view(-1, 1, 1, 1) * torch.exp(
            (-0.5 * (ion_width**2)) * Gsq
        )

        # Extra requirements for lattice gradient:
        if ions.lattice.requires_grad:
            self.Ginterp_prime = Interpolator(Gmag, RadialFunction.DG, deriv=1)
            self.rho_kernel_prime = self.rho_kernel * (-(ion_width**2)) * Gmag
            G = G.permute(3, 0, 1, 2)  # bring gradient direction to front
            self.stress_kernel = FieldH(
                grid,
                data=(
                    torch.where(Gmag == 0.0, 0.0, -1.0 / Gmag) * G[None] * G[:, None]
                ).to(dtype=torch.cdouble),
            )

    @stopwatch(name="Ions.LocalTerms.update")
    def update(self) -> None:
        """Update ionic densities and potentials."""
        ions = self.ions
        SF = self.get_structure_factor()
        ions.Vloc_tilde.data = (SF * self.Ginterp(self.Vloc_coeff)).sum(dim=0)

        if any(item is not None for item in ions.effective_mix):
            # Assemble EAT Vloc gradient terms
            SF_mixGrad = self.get_structure_factor_mixGrad()
            ions.Vloc_tilde_mixGrad = {}
            for (symbol, Vloc_coeffs_grad), rho_kernel_grad in zip(self.Vloc_coeffs_mixGrad.items(), self.rho_kernel_mixGrad.values()):
                grad_list = []
                for vcgrad, rkgrad in zip(Vloc_coeffs_grad, rho_kernel_grad):
                    Vloc_tilde_mix = FieldH(self.system.grid)
                    Vloc_tilde_mix.data = (SF_mixGrad[symbol] * self.Ginterp(vcgrad)).sum(dim=0)
                    rho_tilde_mix = FieldH(self.system.grid)
                    rho_tilde_mix.o = ions.rho_tilde.o.clone()
                    rho_tilde_mix.data = (SF_mixGrad[symbol] * rkgrad).sum(dim=0)
                    grad_list.append(Vloc_tilde_mix + self.system.coulomb.kernel(rho_tilde_mix, correct_G0_width=True))
                ions.Vloc_tilde_mixGrad[symbol] = grad_list

        ions.n_core_tilde.data[0] = (SF * self.Ginterp(self.n_core_coeff)).sum(dim=0)
        ions.rho_tilde.data = (SF * self.rho_kernel).sum(dim=0)
        # Add long-range part of local potential from ionic charge:
        ions.Vloc_tilde += self.system.coulomb.kernel(
            ions.rho_tilde, correct_G0_width=True
        )

    @stopwatch(name="Ions.LocalTerms.update_grad")
    def update_grad(self) -> None:
        """Accumulate local-pseudopotential force / stress contributions."""
        # Propagate long-range local-potential gradient to ionic charge gradient:
        ions = self.ions
        ions.rho_tilde.grad += self.system.coulomb.kernel(
            ions.Vloc_tilde.grad, correct_G0_width=True
        )
        if ions.lattice.requires_grad:
            ions.lattice.grad += self.system.coulomb.kernel.stress(
                ions.Vloc_tilde.grad, ions.rho_tilde
            )

        # Propagate to structure factor gradient:
        SF_grad = (
            self.Ginterp(self.Vloc_coeff) * ions.Vloc_tilde.grad.data
            + self.Ginterp(self.n_core_coeff) * ions.n_core_tilde.grad.data[0]
            + self.rho_kernel * ions.rho_tilde.grad.data
        )
        # Propagate to ionic gradient:
        self.accumulate_structure_factor_forces(SF_grad)

        if ions.lattice.requires_grad:
            # Propagate to radial function gradient:
            SF = self.get_structure_factor()
            radial_part = (
                self.Ginterp_prime(self.Vloc_coeff) * ions.Vloc_tilde.grad.data
                + self.Ginterp_prime(self.n_core_coeff) * ions.n_core_tilde.grad.data[0]
                + self.rho_kernel_prime * ions.rho_tilde.grad.data
            )
            radial_grad = FieldH(
                self.system.grid, data=(radial_part * SF.conj()).sum(dim=0)
            )
            # Propagate to lattice gradient:
            ions.lattice.grad += radial_grad ^ self.stress_kernel

    def get_structure_factor(self) -> torch.Tensor:
        """Compute structure factor."""
        inv_volume = 1.0 / self.system.lattice.volume
        return torch.stack(
            [
                self.ions.translation_phase(self.iG, slice_i).sum(dim=-1) * inv_volume
                for slice_i in self.ions.slices
            ]
        )
    
    def get_structure_factor_mixGrad(self) -> torch.Tensor:
        """Compute structure factor for each EAT atom."""
        inv_volume = 1.0 / self.system.lattice.volume
        SF_mixGrad = {}
        for symbol, slice_i, ps in zip(self.ions.symbols, self.ions.slices, self.ions.pseudopotentials):
            # Only append if effective atom for consistency
            if isinstance(ps, list):
                SF_mixGrad[symbol] = self.ions.translation_phase(self.iG, slice_i).sum(dim=-1) * inv_volume
        return SF_mixGrad

    def accumulate_structure_factor_forces(self, SF_grad: torch.Tensor) -> None:
        """Propagate structure factor gradient to forces."""
        grid = self.system.grid
        pos_grad = self.ions.positions.grad
        assert pos_grad is not None
        inv_volume = 1.0 / grid.lattice.volume
        d_by_dpos = self.iG.permute(3, 0, 1, 2)[None] * (-2j * np.pi * inv_volume)
        for slice_i, SF_grad_i in zip(self.ions.slices, SF_grad):
            phase = self.ions.translation_phase(self.iG, slice_i)
            phase = phase.permute(3, 0, 1, 2)[:, None]  # bring atom dim to front
            dphase_by_dpos = FieldH(grid, data=d_by_dpos * phase)
            pos_grad[slice_i] += FieldH(grid, data=SF_grad_i) ^ dphase_by_dpos


def collect_ps_matrix(self: ions.Ions, n_spinor: int) -> None:
    """Collect pseudopotential matrices across species and atoms.
    Initializes `D_all`."""
    n_proj = self.n_projectors * n_spinor
    self.D_all = torch.zeros((n_proj, n_proj), device=rc.device, dtype=torch.complex128)
    i_proj_start = 0
    effective_mix = self.effective_mix
    D_to_ion = [] # keeps track of which indices of D_all correspond to which ion
    D2i_index = 0
    for i_ps, ps in enumerate(self.pseudopotentials):
        # Check if pseudopotential corresponds to an effective atom
        if isinstance(ps, list):
            for mix_weight, p in zip(effective_mix[i_ps].values(), ps):
                # Add D matrices for each species of each effective atom on the block diagonal of D_all
                # Note that we have to keep track of which block corresponds to which atom since an effective atom will occupy multiple blocks
                D_nlms = p.pqn_beta.expand_matrix(p.D, n_spinor)
                n_proj_atom = D_nlms.shape[0]
                # Each effective atom is treated as a unique element and so self.n_ions_type[i_ps] must equal one
                i_proj_stop = i_proj_start + n_proj_atom
                slice_cur = slice(i_proj_start, i_proj_stop)
                # Multiply diagonal block by effective atom mix weight
                self.D_all[slice_cur, slice_cur] = D_nlms * mix_weight
                i_proj_start = i_proj_stop
                for _ in range(n_proj_atom):
                    D_to_ion.append(D2i_index)
            D2i_index += 1
        else:
            # Ordinary atom procedure
            D_nlms = ps.pqn_beta.expand_matrix(ps.D, n_spinor)
            n_proj_atom = D_nlms.shape[0]
            # Set diagonal block for each atom:
            for i_atom in range(self.n_ions_type[i_ps]):
                i_proj_stop = i_proj_start + n_proj_atom
                slice_cur = slice(i_proj_start, i_proj_stop)
                self.D_all[slice_cur, slice_cur] = D_nlms
                i_proj_start = i_proj_stop
                for _ in range(n_proj_atom):
                    D_to_ion.append(D2i_index)
                D2i_index += 1
    self.D_to_ion = D_to_ion # store D to ion mapping

    # Create EAT gradient D_all dictionary
    self.D_all_mixGrad = {}
    i_proj_start = 0
    for i_ps, (ps, symbols) in enumerate(zip(self.pseudopotentials, self.symbols)):
        # Check if pseudopotential corresponds to an effective atom
        if isinstance(ps, list):
            D_all_mixGrad_list = []
            for mix_weight, p in zip(effective_mix[i_ps].values(), ps):
                D_all_temp = torch.zeros((n_proj, n_proj), device=rc.device, dtype=torch.complex128)
                # Add D matrices for species of each effective atom on the block diagonal of D_all_temp corresponding to EAT gradient
                # Note that we have to keep track of which block corresponds to which atom since an effective atom will occupy multiple blocks
                D_nlms = p.pqn_beta.expand_matrix(p.D, n_spinor)
                n_proj_atom = D_nlms.shape[0]
                # Each effective atom is treated as a unique element and so self.n_ions_type[i_ps] must equal one
                i_proj_stop = i_proj_start + n_proj_atom
                slice_cur = slice(i_proj_start, i_proj_stop)
                # Multiply diagonal block by effective atom mix weight
                D_all_temp[slice_cur, slice_cur] = D_nlms
                i_proj_start = i_proj_stop
                D_all_mixGrad_list.append(D_all_temp)
            self.D_all_mixGrad[symbols] = D_all_mixGrad_list
        else:
            # Mimick ordinary atom procedure to ensure indexing consistency
            for i_atom in range(self.n_ions_type[i_ps]):
                i_proj_stop = i_proj_start + n_proj_atom
                i_proj_start = i_proj_stop



def _update_pulay(ions: ions.Ions, basis: dft.electrons.Basis) -> float:
    "Update `ions.dEtot_drho_basis` and return Pulay correction."
    dE_drho_basis = 0 # dEtot_drho_basis
    for eff_mix, n_ions_i, ps in zip(ions.effective_mix, ions.n_ions_type, ions.pseudopotentials):
        # Chek if effective atom
        if eff_mix is not None:
            dE_drho_mix = []
            for mix_coeff, p in zip(eff_mix.values(), ps):
                dE_drho_mix.append(p.dE_drho_basis(basis.ke_cutoff) * n_ions_i * mix_coeff)
            dE_drho_basis += sum(dE_drho_mix)
        else:
            dE_drho_basis += n_ions_i * ps.dE_drho_basis(basis.ke_cutoff)
    ions.dEtot_drho_basis = dE_drho_basis

    return (
        ions.dEtot_drho_basis
        * (basis.n_ideal - basis.n_avg_weighted)
        / basis.lattice.volume
    )


def _update_pulay_mixGrad(ions: ions.Ions, basis: dft.electrons.Basis) -> float:
    "Return Pulay correction EAT gradient."
    grad = {}
    for eff_mix, n_ions_i, ps, symbol in zip(ions.effective_mix, ions.n_ions_type, ions.pseudopotentials, ions.symbols):
        # Chek if effective atom
        if eff_mix is not None:
            grad_temp = []
            for mix_coeff, p in zip(eff_mix.values(), ps):
                dE_drho_mixGrad = p.dE_drho_basis(basis.ke_cutoff) * n_ions_i
                grad_temp.append(dE_drho_mixGrad * (basis.n_ideal - basis.n_avg_weighted) / basis.lattice.volume)
            # Assign EAT atom with name "symbol" its mix gradient
            grad[symbol] = grad_temp
    
    return grad