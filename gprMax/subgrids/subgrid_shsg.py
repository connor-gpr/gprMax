# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""Switched Huygens Subgridding (SHSG).

Implements the SHSG of Hartley, Giannopoulos & Davidson, "Switched Huygens
Subgridding for the FDTD Method", IEEE Trans. Antennas Propag. 70(8), 2022,
and ch. 9-10 of Hartley's 2020 PhD thesis (University of Edinburgh).

The SHSG reverses the roles of the two Huygens surfaces relative to the HSG:

- The Outer Surface (OS) radiates the main grid INTO the subgrid. The
  spatially/temporally interpolated (and optionally filtered) main-grid
  fields - the precursor machinery the HSG uses at its Inner Surface -
  are applied at the subgrid's OS ring: E components on the OS planes and
  H components half a cell outside them.
- The Inner Surface (IS) radiates the subgrid INTO the main grid
  ("anti-Huygens"): subgrid fields sampled at collocated coarse positions
  update the main-grid E on the IS planes and H half a cell outside them.

Consequences (verified in the 1-D prototype, Phase 0):
- The main-grid solution strictly inside the IS and the subgrid solution
  beyond the OS are identically zero; artificial loss applied there
  (via per-component lossy materials, see apply_shsg_loss) suppresses the
  HSG's late-time instability without touching the physical solution.
- Fields in the IS-OS gap equal the physical solution in EITHER grid.
- The subgrid needs no PML and only a single fine cell beyond the OS:
  n_boundary_cells = is_os_sep * ratio + 1.

The correction signs are identical to the HSG kernel calls; only the
surface positions differ. The IS-ring main-grid corrections reuse the
update_electric_os/update_magnetic_os kernels with the IS box and an
is_os_sep argument of 0 (which lands the subgrid sample positions exactly
on the IS ring). The OS-ring subgrid corrections reuse the update_is
kernel with the ring offset (1 with the SHSG halo) and the OS-box extents.
"""

import logging

import numpy as np

import gprMax.config as config

from ..cython.fields_updates_hsg import update_electric_os, update_is, update_magnetic_os
from ..materials import Material
from .grid import SubGridBaseGrid

logger = logging.getLogger(__name__)


class SubGridSHSG(SubGridBaseGrid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.le = kwargs["le"]
        self.lm = kwargs["lm"]
        self.les = kwargs["les"]
        self.lms = kwargs["lms"]
        # Fine index of the OS ring inside the subgrid arrays
        self.os_f = self.n_boundary_cells - self.s_is_os_sep
        self._loss_applied = False

    # ------------------------------------------------------------------
    # B-side: incident main-grid fields injected at the OS ring.
    # Same kernel, signs and structure as the HSG's IS injection, but at
    # the OS ring (offset os_f) with the OS-box extents. The updater call
    # slots are unchanged, so these keep the update_*_is method names.
    # ------------------------------------------------------------------

    def _os_extents(self):
        s2 = 2 * self.s_is_os_sep
        return self.nwx + s2, self.nwy + s2, self.nwz + s2

    def update_magnetic_is(self, precursors):
        """Updates subgrid H half a cell outside the OS planes with the
        temporally interpolated incident main-grid E (thesis eqs 9.9/9.10;
        the node's own lms-lossy material coefficient provides 1/(1+lms))."""

        owx, owy, owz = self._os_extents()
        n = self.os_f
        nthreads = config.get_model_config().ompthreads

        # Bottom and top
        update_is(owx, owy, owz, self.updatecoeffsH, self.ID, n, -1,
                  owx, owy + 1, owz, 1, self.Hy,
                  precursors.ex_bottom, precursors.ex_top,
                  self.IDlookup["Hy"], 1, -1, 3, nthreads)
        update_is(owx, owy, owz, self.updatecoeffsH, self.ID, n, -1,
                  owx + 1, owy, owz, 1, self.Hx,
                  precursors.ey_bottom, precursors.ey_top,
                  self.IDlookup["Hx"], -1, 1, 3, nthreads)
        # Left and right
        update_is(owx, owy, owz, self.updatecoeffsH, self.ID, n, -1,
                  owy, owz + 1, owx, 2, self.Hz,
                  precursors.ey_left, precursors.ey_right,
                  self.IDlookup["Hz"], 1, -1, 1, nthreads)
        update_is(owx, owy, owz, self.updatecoeffsH, self.ID, n, -1,
                  owy + 1, owz, owx, 2, self.Hy,
                  precursors.ez_left, precursors.ez_right,
                  self.IDlookup["Hy"], -1, 1, 1, nthreads)
        # Front and back
        update_is(owx, owy, owz, self.updatecoeffsH, self.ID, n, -1,
                  owx, owz + 1, owy, 3, self.Hz,
                  precursors.ex_front, precursors.ex_back,
                  self.IDlookup["Hz"], -1, 1, 2, nthreads)
        update_is(owx, owy, owz, self.updatecoeffsH, self.ID, n, -1,
                  owx + 1, owz, owy, 3, self.Hx,
                  precursors.ez_front, precursors.ez_back,
                  self.IDlookup["Hx"], 1, -1, 2, nthreads)

    def update_electric_is(self, precursors):
        """Updates subgrid E on the OS planes with the spatially and
        temporally interpolated incident main-grid H (working-region
        nodes - their materials carry no loss)."""

        owx, owy, owz = self._os_extents()
        n = self.os_f
        nthreads = config.get_model_config().ompthreads

        # Bottom and top
        update_is(owx, owy, owz, self.updatecoeffsE, self.ID, n, 0,
                  owx, owy + 1, owz, 1, self.Ex,
                  precursors.hy_bottom, precursors.hy_top,
                  self.IDlookup["Ex"], 1, -1, 3, nthreads)
        update_is(owx, owy, owz, self.updatecoeffsE, self.ID, n, 0,
                  owx + 1, owy, owz, 1, self.Ey,
                  precursors.hx_bottom, precursors.hx_top,
                  self.IDlookup["Ey"], -1, 1, 3, nthreads)
        # Left and right
        update_is(owx, owy, owz, self.updatecoeffsE, self.ID, n, 0,
                  owy, owz + 1, owx, 2, self.Ey,
                  precursors.hz_left, precursors.hz_right,
                  self.IDlookup["Ey"], 1, -1, 1, nthreads)
        update_is(owx, owy, owz, self.updatecoeffsE, self.ID, n, 0,
                  owy + 1, owz, owx, 2, self.Ez,
                  precursors.hy_left, precursors.hy_right,
                  self.IDlookup["Ez"], -1, 1, 1, nthreads)
        # Front and back
        update_is(owx, owy, owz, self.updatecoeffsE, self.ID, n, 0,
                  owx, owz + 1, owy, 3, self.Ex,
                  precursors.hz_front, precursors.hz_back,
                  self.IDlookup["Ex"], -1, 1, 2, nthreads)
        update_is(owx, owy, owz, self.updatecoeffsE, self.ID, n, 0,
                  owx + 1, owz, owy, 3, self.Ez,
                  precursors.hx_front, precursors.hx_back,
                  self.IDlookup["Ez"], 1, -1, 2, nthreads)

    # ------------------------------------------------------------------
    # A-side: subgrid fields injected into the main grid at the IS ring.
    # Identical kernel calls to the HSG's OS correction, but over the IS
    # box with is_os_sep = 0 (sample positions land on the IS ring). The
    # main-grid IS-plane nodes carry le-lossy materials, so the kernel's
    # updatecoeffs lookup supplies 1/(1+le) automatically (thesis eqs
    # 9.11/9.12).
    # ------------------------------------------------------------------

    def update_electric_os(self, main_grid):
        i_l, i_u = self.i0, self.i1
        j_l, j_u = self.j0, self.j1
        k_l, k_u = self.k0, self.k1
        nthreads = config.get_model_config().ompthreads

        # Front and back
        update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 3,
                           i_l, i_u, k_l, k_u + 1, j_l, j_u, self.nwy,
                           main_grid.IDlookup["Ex"], main_grid.Ex, self.Hz,
                           2, 1, -1, 1, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 3,
                           i_l, i_u + 1, k_l, k_u, j_l, j_u, self.nwy,
                           main_grid.IDlookup["Ez"], main_grid.Ez, self.Hx,
                           2, -1, 1, 0, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        # Left and right
        update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 2,
                           j_l, j_u, k_l, k_u + 1, i_l, i_u, self.nwx,
                           main_grid.IDlookup["Ey"], main_grid.Ey, self.Hz,
                           1, -1, 1, 1, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 2,
                           j_l, j_u + 1, k_l, k_u, i_l, i_u, self.nwx,
                           main_grid.IDlookup["Ez"], main_grid.Ez, self.Hy,
                           1, 1, -1, 0, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        # Bottom and top
        update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 1,
                           i_l, i_u, j_l, j_u + 1, k_l, k_u, self.nwz,
                           main_grid.IDlookup["Ex"], main_grid.Ex, self.Hy,
                           3, -1, 1, 1, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        update_electric_os(main_grid.updatecoeffsE, main_grid.ID, 1,
                           i_l, i_u + 1, j_l, j_u, k_l, k_u, self.nwz,
                           main_grid.IDlookup["Ey"], main_grid.Ey, self.Hx,
                           3, 1, -1, 0, self.ratio, 0,
                           self.n_boundary_cells, nthreads)

    def update_magnetic_os(self, main_grid):
        i_l, i_u = self.i0, self.i1
        j_l, j_u = self.j0, self.j1
        k_l, k_u = self.k0, self.k1
        nthreads = config.get_model_config().ompthreads

        # Front and back
        update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 3,
                           i_l, i_u, k_l, k_u + 1, j_l - 1, j_u, self.nwy,
                           main_grid.IDlookup["Hz"], main_grid.Hz, self.Ex,
                           2, 1, -1, 1, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 3,
                           i_l, i_u + 1, k_l, k_u, j_l - 1, j_u, self.nwy,
                           main_grid.IDlookup["Hx"], main_grid.Hx, self.Ez,
                           2, -1, 1, 0, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        # Left and right
        update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 2,
                           j_l, j_u, k_l, k_u + 1, i_l - 1, i_u, self.nwx,
                           main_grid.IDlookup["Hz"], main_grid.Hz, self.Ey,
                           1, -1, 1, 1, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 2,
                           j_l, j_u + 1, k_l, k_u, i_l - 1, i_u, self.nwx,
                           main_grid.IDlookup["Hy"], main_grid.Hy, self.Ez,
                           1, 1, -1, 0, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        # Bottom and top
        update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 1,
                           i_l, i_u, j_l, j_u + 1, k_l - 1, k_u, self.nwz,
                           main_grid.IDlookup["Hy"], main_grid.Hy, self.Ex,
                           3, -1, 1, 1, self.ratio, 0,
                           self.n_boundary_cells, nthreads)
        update_magnetic_os(main_grid.updatecoeffsH, main_grid.ID, 1,
                           i_l, i_u + 1, j_l, j_u, k_l - 1, k_u, self.nwz,
                           main_grid.IDlookup["Hx"], main_grid.Hx, self.Ey,
                           3, 1, -1, 0, self.ratio, 0,
                           self.n_boundary_cells, nthreads)

    # ------------------------------------------------------------------
    # Artificial loss in the non-working regions
    # ------------------------------------------------------------------

    def apply_shsg_loss(self, main_grid):
        """Paints per-component lossy materials on the non-working regions:
        the subgrid beyond its OS box (les/lms) and the main grid inside
        the IS box (le/lm). Loss factor l = sigma*dt/(2*eps) is realised
        as material conductivity so the standard update kernels apply the
        (1-l)/(1+l) form with no new code (thesis eqs 9.5-9.8)."""

        if self._loss_applied:
            return
        self._loss_applied = True

        self._validate_main_grid_placements(main_grid)

        if self.les > 0 or self.lms > 0:
            self._paint_subgrid_loss()
        if self.le > 0 or self.lm > 0:
            self._paint_main_loss(main_grid)

    def _validate_main_grid_placements(self, main_grid):
        """SHSG proposition 1: fields sourced in a non-working region DO
        radiate into the working regions (unlike the HSG), and the
        main-grid solution inside the IS is identically zero. Reject
        main-grid sources inside the IS box and warn about receivers."""

        def inside(obj):
            return (self.i0 <= obj.xcoord <= self.i1
                    and self.j0 <= obj.ycoord <= self.j1
                    and self.k0 <= obj.zcoord <= self.k1)

        sources = (main_grid.hertziandipoles + main_grid.magneticdipoles
                   + main_grid.voltagesources + main_grid.transmissionlines)
        for src in sources:
            if inside(src):
                logger.exception(
                    f"[{self.name}] main-grid source '{src.ID}' lies inside "
                    "the subgrid's Inner Surface. In the switched HSG this "
                    "region is non-working (zero field) and sources placed "
                    "there corrupt the physical solution - move the source "
                    "into the subgrid or outside the IS."
                )
                raise ValueError
        for rx in main_grid.rxs:
            if inside(rx):
                logger.warning(
                    f"[{self.name}] main-grid receiver '{rx.ID}' lies inside "
                    "the subgrid's Inner Surface: in the switched HSG it "
                    "will record ~zero field. Place receivers in the "
                    "subgrid or outside the IS."
                )

    @staticmethod
    def _add_loss_material(grid, l_e, l_m, name):
        """Appends a material with electric loss factor l_e and magnetic
        loss factor l_m to a built grid, returning its numID."""
        e0 = config.sim_config.em_consts["e0"]
        m0 = config.sim_config.em_consts["m0"]
        numID = grid.updatecoeffsE.shape[0]
        mat = Material(numID, name)
        mat.type = "builtin"
        mat.averagable = False
        mat.er, mat.mr = 1.0, 1.0
        mat.se = 2.0 * l_e * e0 / grid.dt
        mat.sm = 2.0 * l_m * m0 / grid.dt
        mat.calculate_update_coeffsE(grid)
        mat.calculate_update_coeffsH(grid)
        rowE = np.array([[mat.CA, mat.CBx, mat.CBy, mat.CBz, mat.srce]],
                        dtype=grid.updatecoeffsE.dtype)
        rowH = np.array([[mat.DA, mat.DBx, mat.DBy, mat.DBz, mat.srcm]],
                        dtype=grid.updatecoeffsH.dtype)
        grid.updatecoeffsE = np.concatenate((grid.updatecoeffsE, rowE))
        grid.updatecoeffsH = np.concatenate((grid.updatecoeffsH, rowH))
        if config.get_model_config().materials["maxpoles"] > 0:
            zrow = np.zeros((1, grid.updatecoeffsdispersive.shape[1]),
                            dtype=grid.updatecoeffsdispersive.dtype)
            grid.updatecoeffsdispersive = np.concatenate(
                (grid.updatecoeffsdispersive, zrow))
        grid.materials.append(mat)
        return numID

    # Component -> (ID index, axes in which the node is staggered by 1/2)
    _E_COMPONENTS = {"Ex": (0,), "Ey": (1,), "Ez": (2,)}
    _H_COMPONENTS = {"Hx": (1, 2), "Hy": (0, 2), "Hz": (0, 1)}
    _AXIS = {"x": 0, "y": 1, "z": 2}

    @staticmethod
    def _inside_mask(shape3, box_lo, box_hi, staggered_axes):
        """Boolean mask of nodes whose positions lie within the closed box
        (node positions: index + 0.5 along staggered axes)."""
        mask = np.ones(shape3, dtype=bool)
        for ax in range(3):
            pos = np.arange(shape3[ax], dtype=float)
            if ax in staggered_axes:
                pos = pos + 0.5
            ins = (pos >= box_lo[ax]) & (pos <= box_hi[ax])
            shape = [1, 1, 1]
            shape[ax] = shape3[ax]
            mask &= ins.reshape(shape)
        return mask

    @classmethod
    def _outside_mask(cls, shape3, box_lo, box_hi, staggered_axes):
        """Boolean mask of nodes strictly outside the closed box."""
        return ~cls._inside_mask(shape3, box_lo, box_hi, staggered_axes)

    def _paint_subgrid_loss(self):
        owx, owy, owz = self._os_extents()
        lo = (self.os_f,) * 3
        hi = (self.os_f + owx, self.os_f + owy, self.os_f + owz)
        shape3 = self.ID.shape[1:]
        e_mat = self._add_loss_material(self, self.les, 0.0, f"shsg_les_{self.name}")
        h_mat = self._add_loss_material(self, 0.0, self.lms, f"shsg_lms_{self.name}")
        lookup = self.IDlookup
        for comp, stag in self._E_COMPONENTS.items():
            mask = self._outside_mask(shape3, lo, hi, stag)
            self.ID[lookup[comp]][mask] = e_mat
        for comp, stag in self._H_COMPONENTS.items():
            mask = self._outside_mask(shape3, lo, hi, stag)
            self.ID[lookup[comp]][mask] = h_mat

    def _paint_main_loss(self, main_grid):
        lo = (self.i0, self.j0, self.k0)
        hi = (self.i1, self.j1, self.k1)
        shape3 = main_grid.ID.shape[1:]
        e_mat = self._add_loss_material(main_grid, self.le, 0.0, f"shsg_le_{self.name}")
        h_mat = self._add_loss_material(main_grid, 0.0, self.lm, f"shsg_lm_{self.name}")
        lookup = main_grid.IDlookup
        for comp, stag in self._E_COMPONENTS.items():
            mask = self._inside_mask(shape3, lo, hi, stag)
            main_grid.ID[lookup[comp]][mask] = e_mat
        for comp, stag in self._H_COMPONENTS.items():
            mask = self._inside_mask(shape3, lo, hi, stag)
            main_grid.ID[lookup[comp]][mask] = h_mat

    def print_info(self):
        xs, ys, zs = self.round_to_grid(
            (self.i0 * self.dx * self.ratio,
             self.j0 * self.dy * self.ratio,
             self.k0 * self.dz * self.ratio))
        xf, yf, zf = self.round_to_grid(
            (self.i1 * self.dx * self.ratio,
             self.j1 * self.dy * self.ratio,
             self.k1 * self.dz * self.ratio))
        logger.info("")
        logger.info(f"[{self.name}] Type: {self.__class__.__name__} (switched HSG)")
        logger.info(f"[{self.name}] Ratio: 1:{self.ratio}")
        logger.info(
            f"[{self.name}] Spatial discretisation: {self.dx:g} x "
            + f"{self.dy:g} x {self.dz:g}m"
        )
        logger.info(
            f"[{self.name}] Extent (working region): {xs}m, {ys}m, {zs}m to "
            + f"{xf}m, {yf}m, {zf}m "
            + f"(({self.nwx} x {self.nwy} x {self.nwz} = "
            + f"{self.nwx * self.nwy * self.nwz} cells)"
        )
        logger.info(
            f"[{self.name}] Loss factors: le={self.le:g}, lm={self.lm:g}, "
            + f"les={self.les:g}, lms={self.lms:g}; no subgrid PML"
        )
        logger.info(f"[{self.name}] Time step: {self.dt:g} secs")
