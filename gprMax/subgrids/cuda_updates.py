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

"""CUDA (device-resident) solver for SHSG sub-grids (Phase 1).

Architecture: one shared CUDA context; the main grid is driven by
CUDASubgridUpdates (a CUDAUpdates) and each sub-grid by a
CUDASubgridUpdater (also a CUDAUpdates, on the sub-grid's own compiled
module set with the sub-grid's dims and loss-painted material
coefficients baked in). The hsg_1/hsg_2 substep sequences are
transliterated from the CPU SubgridUpdater; the ring-correction call
tables mirror SubGridSHSG.update_*_is/os argument for argument.

Kernel/module placement (the cross-grid indexing contract):
- update_is_e/h write the SUB-GRID: compiled per sub-grid module set;
  the 2-D precursor buffers are foreign (runtime row length).
- update_electric_os/update_magnetic_os write the MAIN grid: compiled
  once into the main-grid module set; the 3-D sub-grid field is foreign
  (runtime dims), so one compilation serves all sub-grids.
- pack_planes reads the main grid: compiled with the main-grid module.

Ordering constraint: apply_shsg_loss appends material rows and repaints
ID in the non-working regions, and the coefficient array sizes are baked
into module source at render time - so loss painting MUST precede
construction of every updates object here (asserted in create_updates).

Precision: the sub-grid rule (double) applies; kernels are $REAL-clean
so the experimental single-precision mode is a config flip.
"""

import logging

import numpy as np

import gprMax.config as config
from gprMax.cuda_opencl import knl_precursors, knl_subgrid_coupling
from gprMax.model import Model
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.updates import HSGCapable

from .cuda_grid import CUDASubGridSHSG
from .cuda_precursors import CUDAPrecursorNodes, CUDAPrecursorNodesFiltered
from .updates import OSSurfaceView

logger = logging.getLogger(__name__)


def _build_multi_knl(updates, knl_funcs):
    """Renders several kernels into one module source sharing the updates
    object's knl_common (baked dims + __constant__ coefficient arrays)."""
    parts = [updates.knl_common]
    for kf in knl_funcs:
        parts.append(
            kf["args_cuda"].substitute(updates.subs_name_args)
            + "{"
            + kf["func"].substitute(updates.subs_func)
            + "}"
        )
    return "\n".join(parts)


def create_updates(model: Model):
    """Returns the CUDA updates object for a sub-gridded model."""

    if config.get_model_config().materials["maxpoles"] > 0:
        logger.exception(
            "Dispersive materials anywhere in the model (including "
            "main-grid-only) are not yet supported with subgrid_gpu - "
            "dispersive sub-grid support is a later phase. Use the CPU "
            "solver."
        )
        raise ValueError

    for sg in model.subgrids:
        if not isinstance(sg, CUDASubGridSHSG):
            logger.exception(
                f"[{sg.name}] the CUDA sub-grid solver supports SubGridSHSG "
                "only (the plain HSG remains CPU-only). Use #subgrid_shsg / "
                "gprMax.SubGridSHSG, or the CPU solver."
            )
            raise ValueError

    # Loss painting appends material rows and repaints ID; it must precede
    # any kernel-module render/compile or device upload (sizes are baked at
    # render time and constants are copied per module).
    for sg in model.subgrids:
        sg.apply_shsg_loss(model.G)
        assert sg._loss_applied

    updates = CUDASubgridUpdates(model.G)
    updaters = []
    for sg in model.subgrids:
        surface = OSSurfaceView(sg)
        if sg.filter:
            precursors = CUDAPrecursorNodesFiltered(model.G, surface)
        else:
            precursors = CUDAPrecursorNodes(model.G, surface)
        updaters.append(CUDASubgridUpdater(sg, precursors, model.G, updates))
    updates.updaters = updaters
    return updates


class CUDASubgridUpdates(CUDAUpdates, HSGCapable):
    """Main-grid CUDA updates plus the sub-grid phase delegation and the
    main-grid-side coupling kernels (IS-ring corrections, plane packing)."""

    def __init__(self, G):
        super().__init__(G)  # owns the context
        self.updaters = []
        self._set_coupling_knls()

    def _set_coupling_knls(self):
        bld = _build_multi_knl(
            self,
            [
                knl_subgrid_coupling.update_electric_os,
                knl_subgrid_coupling.update_magnetic_os,
                knl_subgrid_coupling.update_os_faces_electric,
                knl_subgrid_coupling.update_os_faces_magnetic,
                knl_subgrid_coupling.pack_planes,
                knl_precursors.gather_weighted_planes,
                knl_precursors.interp_faces,
            ],
        )
        knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
        self.update_electric_os_dev = knl.get_function("update_electric_os")
        self.update_magnetic_os_dev = knl.get_function("update_magnetic_os")
        self.update_os_faces_electric_dev = knl.get_function("update_os_faces_electric")
        self.update_os_faces_magnetic_dev = knl.get_function("update_os_faces_magnetic")
        self.pack_planes_dev = knl.get_function("pack_planes")
        self.gather_weighted_planes_dev = knl.get_function("gather_weighted_planes")
        self.interp_faces_dev = knl.get_function("interp_faces")
        # Constants are per-module: this module's coefficients must include
        # the SHSG loss rows painted on the main grid.
        self._copy_mat_coeffs(knl, knl)

    def hsg_1(self):
        """Updates the sub-grids over the first phase (electric exchange)."""
        for updater in self.updaters:
            updater.hsg_1()

    def hsg_2(self):
        """Updates the sub-grids over the second phase (magnetic exchange)."""
        for updater in self.updaters:
            updater.hsg_2()

    def finalise(self):
        super().finalise()
        for updater in self.updaters:
            updater.finalise()

    def cleanup(self):
        for updater in self.updaters:
            updater.cleanup()  # non-owners: no context pop
        super().cleanup()


class CUDASubgridUpdater(CUDAUpdates):
    """Drives one SHSG sub-grid on the device: volume updates via the
    inherited CUDAUpdates machinery on the sub-grid's module set, ring
    corrections via the coupling kernels, sources/receivers on device
    with the sub-grid-local iteration counter."""

    def __init__(self, subgrid: CUDASubGridSHSG, precursors, G, parent):
        # Render/compile bakes coefficient-array sizes: loss must be painted
        assert subgrid._loss_applied, "apply_shsg_loss must run before device setup"
        super().__init__(subgrid, ctx=parent.ctx)
        self.precursors = precursors
        self.G = G
        self.parent = parent
        self.iteration = 0
        self._real = config.sim_config.dtypes["float_or_double"]

        self._set_is_knls()
        precursors.attach_device(parent)
        self._build_call_tables()
        self._build_fused_specs()

    # ------------------------------------------------------------------
    # Kernels and call tables
    # ------------------------------------------------------------------

    def _set_is_knls(self):
        bld = _build_multi_knl(
            self,
            [
                knl_subgrid_coupling.update_is_e,
                knl_subgrid_coupling.update_is_h,
                knl_subgrid_coupling.update_is_faces_e,
                knl_subgrid_coupling.update_is_faces_h,
            ],
        )
        knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
        self.update_is_e_dev = knl.get_function("update_is_e")
        self.update_is_h_dev = knl.get_function("update_is_h")
        self.update_is_faces_e_dev = knl.get_function("update_is_faces_e")
        self.update_is_faces_h_dev = knl.get_function("update_is_faces_h")
        # This module's constants are the sub-grid's loss-painted coefficients
        self._copy_mat_coeffs(knl, knl)

    def _build_call_tables(self):
        """Mirrors SubGridSHSG.update_*_is/os argument for argument."""
        sg = self.grid
        owx, owy, owz = sg._os_extents()
        self._ow = (owx, owy, owz)
        idl = sg.IDlookup

        # (face, nwl, nwm, field, name_l, name_u, lookup, sign_l, sign_u, co)
        # E ring: precursor H buffers; offset 0
        self._is_e_calls = [
            (1, owx, owy + 1, "Ex", "hy_bottom", "hy_top", idl["Ex"], 1, -1, 3),
            (1, owx + 1, owy, "Ey", "hx_bottom", "hx_top", idl["Ey"], -1, 1, 3),
            (2, owy, owz + 1, "Ey", "hz_left", "hz_right", idl["Ey"], 1, -1, 1),
            (2, owy + 1, owz, "Ez", "hy_left", "hy_right", idl["Ez"], -1, 1, 1),
            (3, owx, owz + 1, "Ex", "hz_front", "hz_back", idl["Ex"], -1, 1, 2),
            (3, owx + 1, owz, "Ez", "hx_front", "hx_back", idl["Ez"], 1, -1, 2),
        ]
        # H ring: precursor E buffers; offset -1
        self._is_h_calls = [
            (1, owx, owy + 1, "Hy", "ex_bottom", "ex_top", idl["Hy"], 1, -1, 3),
            (1, owx + 1, owy, "Hx", "ey_bottom", "ey_top", idl["Hx"], -1, 1, 3),
            (2, owy, owz + 1, "Hz", "ey_left", "ey_right", idl["Hz"], 1, -1, 1),
            (2, owy + 1, owz, "Hy", "ez_left", "ez_right", idl["Hy"], -1, 1, 1),
            (3, owx, owz + 1, "Hz", "ex_front", "ex_back", idl["Hz"], -1, 1, 2),
            (3, owx + 1, owz, "Hx", "ez_front", "ez_back", idl["Hx"], 1, -1, 2),
        ]

        G = self.G
        i0, i1 = sg.i0, sg.i1
        j0, j1 = sg.j0, sg.j1
        k0, k1 = sg.k0, sg.k1
        gidl = G.IDlookup
        # (face, l_l, l_u, m_l, m_u, n_l, n_u, nwn, lookup, field, inc, co,
        #  sign_n, sign_f, mid) - SHSG argument set: IS box, s=0
        self._os_e_calls = [
            (3, i0, i1, k0, k1 + 1, j0, j1, sg.nwy, gidl["Ex"], "Ex", "Hz", 2, 1, -1, 1),
            (3, i0, i1 + 1, k0, k1, j0, j1, sg.nwy, gidl["Ez"], "Ez", "Hx", 2, -1, 1, 0),
            (2, j0, j1, k0, k1 + 1, i0, i1, sg.nwx, gidl["Ey"], "Ey", "Hz", 1, -1, 1, 1),
            (2, j0, j1 + 1, k0, k1, i0, i1, sg.nwx, gidl["Ez"], "Ez", "Hy", 1, 1, -1, 0),
            (1, i0, i1, j0, j1 + 1, k0, k1, sg.nwz, gidl["Ex"], "Ex", "Hy", 3, -1, 1, 1),
            (1, i0, i1 + 1, j0, j1, k0, k1, sg.nwz, gidl["Ey"], "Ey", "Hx", 3, 1, -1, 0),
        ]
        self._os_h_calls = [
            (3, i0, i1, k0, k1 + 1, j0 - 1, j1, sg.nwy, gidl["Hz"], "Hz", "Ex", 2, 1, -1, 1),
            (3, i0, i1 + 1, k0, k1, j0 - 1, j1, sg.nwy, gidl["Hx"], "Hx", "Ez", 2, -1, 1, 0),
            (2, j0, j1, k0, k1 + 1, i0 - 1, i1, sg.nwx, gidl["Hz"], "Hz", "Ey", 1, -1, 1, 1),
            (2, j0, j1 + 1, k0, k1, i0 - 1, i1, sg.nwx, gidl["Hy"], "Hy", "Ez", 1, 1, -1, 0),
            (1, i0, i1, j0, j1 + 1, k0 - 1, k1, sg.nwz, gidl["Hy"], "Hy", "Ex", 3, -1, 1, 1),
            (1, i0, i1 + 1, j0, j1, k0 - 1, k1, sg.nwz, gidl["Hx"], "Hx", "Ey", 3, 1, -1, 0),
        ]

    @staticmethod
    def _split_conflict_free(calls, name_index):
        """Splits the six per-face calls into two groups such that no field
        component appears twice within a group. The ring-box EDGE nodes
        receive += contributions from two faces writing the same component;
        within one launch those would be a read-modify-write race (this is
        exactly the fused-kernel hazard - sequential launches are safe), so
        each conflicting pair is separated. Every component appears exactly
        twice, so a greedy first-fit 2-colouring always succeeds."""
        group_a, group_b, seen_a = [], [], set()
        for call in calls:
            fname = call[name_index]
            if fname not in seen_a:
                group_a.append(call)
                seen_a.add(fname)
            else:
                group_b.append(call)
        return group_a, group_b

    def _build_fused_specs(self):
        """Builds the spec tables for the fused ring kernels (Phase 3):
        6 per-face launches collapse into 2 conflict-free group launches."""
        sel_e = {"Ex": 0, "Ey": 1, "Ez": 2}
        sel_h = {"Hx": 0, "Hy": 1, "Hz": 2}
        self._fused_is = {}
        for key, calls, kind, sel in (
            ("e", self._is_e_calls, "h", sel_e),
            ("h", self._is_h_calls, "e", sel_h),
        ):
            offsets = self.precursors._fine[kind]["offsets"]
            groups = []
            for group in self._split_conflict_free(calls, 3):
                rows = []
                total = 0
                for face, nwl, nwm, fname, name_l, name_u, lookup, sign_l, sign_u, co in group:
                    off_l, shape_l = offsets[name_l]
                    off_u, _ = offsets[name_u]
                    rows.append(
                        [face, nwl, nwm, sel[fname], int(off_l), int(off_u),
                         int(shape_l[1]), lookup, sign_l, sign_u, co, total]
                    )
                    total += nwl * nwm
                groups.append(
                    {
                        "specs_dev": self.grid.gpuarray.to_gpu(
                            np.array(rows, dtype=np.int32)
                        ),
                        "n_specs": len(rows),
                        "total": total,
                    }
                )
            self._fused_is[key] = {"groups": groups, "kind": kind}

        self._fused_os = {}
        for key, calls, fsel, isel in (
            ("e", self._os_e_calls, sel_e, sel_h),
            ("h", self._os_h_calls, sel_h, sel_e),
        ):
            groups = []
            for group in self._split_conflict_free(calls, 9):
                rows = []
                total = 0
                for face, l_l, l_u, m_l, m_u, n_l, n_u, nwn, lookup, fname, incname, co, sign_n, sign_f, mid in group:
                    rows.append(
                        [face, int(l_l), int(l_u), int(m_l), int(m_u), int(n_l),
                         int(n_u), int(nwn), lookup, fsel[fname], isel[incname],
                         co, sign_n, sign_f, mid, total]
                    )
                    total += (l_u - l_l) * (m_u - m_l)
                groups.append(
                    {
                        "specs_dev": self.grid.gpuarray.to_gpu(
                            np.array(rows, dtype=np.int32)
                        ),
                        "n_specs": len(rows),
                        "total": total,
                    }
                )
            self._fused_os[key] = {"groups": groups}

    @staticmethod
    def _bpg(total):
        return (int(np.ceil(total / 128)), 1, 1)

    # ------------------------------------------------------------------
    # Ring corrections (kernel launches)
    # ------------------------------------------------------------------

    def _launch_is(self, knl_func, calls, kind, offset, weights):
        sg = self.grid
        owx, owy, owz = self._ow
        n = sg.os_f
        c1, c2 = weights
        for face, nwl, nwm, fname, name_l, name_u, lookup, sign_l, sign_u, co in calls:
            p0_l, p1_l, pre_nm = self.precursors.dev_ptrs(kind, name_l)
            p0_u, p1_u, _ = self.precursors.dev_ptrs(kind, name_u)
            knl_func(
                np.int32(owx),
                np.int32(owy),
                np.int32(owz),
                np.int32(n),
                np.int32(offset),
                np.int32(nwl),
                np.int32(nwm),
                np.int32(face),
                sg.ID_dev.gpudata,
                getattr(sg, f"{fname}_dev").gpudata,
                p0_l,
                p1_l,
                p0_u,
                p1_u,
                self._real(c1),
                self._real(c2),
                pre_nm,
                np.int32(lookup),
                np.int32(sign_l),
                np.int32(sign_u),
                np.int32(co),
                block=(128, 1, 1),
                grid=self._bpg(nwl * nwm),
            )

    def _launch_is_fused(self, knl_func, key, offset, weights):
        sg = self.grid
        owx, owy, owz = self._ow
        f = self._fused_is[key]
        fine = self.precursors._fine[f["kind"]]
        c1, c2 = weights
        pre = "E" if key == "e" else "H"
        for g in f["groups"]:
            knl_func(
                np.int32(owx),
                np.int32(owy),
                np.int32(owz),
                np.int32(sg.os_f),
                np.int32(offset),
                np.int32(g["n_specs"]),
                np.int32(g["total"]),
                g["specs_dev"].gpudata,
                sg.ID_dev.gpudata,
                getattr(sg, f"{pre}x_dev").gpudata,
                getattr(sg, f"{pre}y_dev").gpudata,
                getattr(sg, f"{pre}z_dev").gpudata,
                fine["dev_0"].gpudata,
                fine["dev_1"].gpudata,
                self._real(c1),
                self._real(c2),
                block=(128, 1, 1),
                grid=self._bpg(g["total"]),
            )

    def update_electric_is(self):
        """Sub-grid E on the OS planes from interpolated main-grid H."""
        self._launch_is_fused(
            self.update_is_faces_e_dev, "e", 0, self.precursors.h_weights
        )

    def update_magnetic_is(self):
        """Sub-grid H half a cell outside the OS planes from interpolated
        main-grid E."""
        self._launch_is_fused(
            self.update_is_faces_h_dev, "h", -1, self.precursors.e_weights
        )

    def _launch_os(self, knl_func, calls):
        sg = self.grid
        G = self.G
        r = np.int32(sg.ratio)
        s = np.int32(0)
        nb = np.int32(sg.n_boundary_cells)
        for face, l_l, l_u, m_l, m_u, n_l, n_u, nwn, lookup, fname, incname, co, sign_n, sign_f, mid in calls:
            inc = getattr(sg, f"{incname}_dev")
            knl_func(
                np.int32(face),
                np.int32(l_l),
                np.int32(l_u),
                np.int32(m_l),
                np.int32(m_u),
                np.int32(n_l),
                np.int32(n_u),
                np.int32(nwn),
                G.ID_dev.gpudata,
                getattr(G, f"{fname}_dev").gpudata,
                inc.gpudata,
                np.int32(inc.shape[1]),
                np.int32(inc.shape[2]),
                np.int32(lookup),
                np.int32(co),
                np.int32(sign_n),
                np.int32(sign_f),
                np.int32(mid),
                r,
                s,
                nb,
                block=(128, 1, 1),
                grid=self._bpg((l_u - l_l) * (m_u - m_l)),
            )

    def _launch_os_fused(self, knl_func, key):
        sg = self.grid
        G = self.G
        f = self._fused_os[key]
        fpre = "E" if key == "e" else "H"
        ipre = "H" if key == "e" else "E"
        for g in f["groups"]:
            knl_func(
                np.int32(g["n_specs"]),
                np.int32(g["total"]),
                g["specs_dev"].gpudata,
                G.ID_dev.gpudata,
                getattr(G, f"{fpre}x_dev").gpudata,
                getattr(G, f"{fpre}y_dev").gpudata,
                getattr(G, f"{fpre}z_dev").gpudata,
                getattr(sg, f"{ipre}x_dev").gpudata,
                getattr(sg, f"{ipre}y_dev").gpudata,
                getattr(sg, f"{ipre}z_dev").gpudata,
                np.int32(sg.Ex.shape[1]),
                np.int32(sg.Ex.shape[2]),
                np.int32(sg.ratio),
                np.int32(0),
                np.int32(sg.n_boundary_cells),
                block=(128, 1, 1),
                grid=self._bpg(g["total"]),
            )

    def update_electric_os(self):
        """Main-grid E on the IS planes from collocated sub-grid H."""
        self._launch_os_fused(self.parent.update_os_faces_electric_dev, "e")

    def update_magnetic_os(self):
        """Main-grid H half a cell outside the IS from collocated
        sub-grid E."""
        self._launch_os_fused(self.parent.update_os_faces_magnetic_dev, "h")

    # ------------------------------------------------------------------
    # Sub-grid-local iteration bookkeeping (mirrors CPU SubgridUpdater)
    # ------------------------------------------------------------------

    def store_outputs(self):
        super().store_outputs(self.iteration)

    def update_electric_sources(self):
        super().update_electric_sources(self.iteration)
        self.iteration += 1

    def update_magnetic_sources(self):
        super().update_magnetic_sources(self.iteration)

    # ------------------------------------------------------------------
    # The two substep phases - transliterated from the CPU SubgridUpdater
    # (subgrids/updates.py); same call order, same iteration increment
    # placement, same final-substep exact-time handling.
    # ------------------------------------------------------------------

    def hsg_1(self):
        """First half of the sub-grid update."""
        subgrid = self.grid
        precursors = self.precursors

        precursors.update_electric()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(int(m + subgrid.ratio / 2 - 0.5))
            self.update_electric_is()
            self.update_electric_sources()
            self.update_electric_b()
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(m)
            self.update_magnetic_is()
            self.update_magnetic_sources()

        self.store_outputs()
        self.update_electric_a()
        self.update_electric_pml()
        precursors.calc_exact_magnetic_in_time()
        self.update_electric_is()
        self.update_electric_sources()
        self.update_electric_b()
        self.update_electric_os()

    def hsg_2(self):
        """Second half of the sub-grid update."""
        subgrid = self.grid
        precursors = self.precursors

        precursors.update_magnetic()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(int(m + subgrid.ratio / 2 - 0.5))
            self.update_magnetic_is()
            self.update_magnetic_sources()
            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(m)
            self.update_electric_is()
            self.update_electric_sources()
            self.update_electric_b()

        self.update_magnetic()
        self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        self.update_magnetic_is()
        self.update_magnetic_sources()
        self.update_magnetic_os()
