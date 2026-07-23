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

"""Device bridge for the precursor nodes (Phase 1 of the CUDA sub-grid
solver).

Spatial interpolation (FIR filter, transverse weighting, SciPy spline)
stays on the CPU, byte-identical to the CPU solver: the coarse surface
planes it needs are gathered on device by the pack_planes kernel,
transferred in ONE packed copy per snapshot, and scattered into the
(otherwise stale) host main-grid arrays at exactly their slice positions
- so the unmodified PrecursorNodes[Filtered] code then reads what it
would have read on the CPU path.

The interpolated fine _1 face arrays are packed and uploaded in ONE
pinned copy per snapshot per field type. The _0 set stays on device from
the previous snapshot (buffer swap, no copy). Temporal blending moves
into the update_is kernels: interpolate_*_in_time / calc_exact_* only
record the scalar weights (c1, c2) here.

Steady-state PCIe traffic: 2 packed D2H + 2 packed H2D per main-grid
iteration, independent of substep count.
"""

import numpy as np

import gprMax.config as config
from .precursor_nodes import (
    PrecursorNodes,
    PrecursorNodesFiltered,
    calculate_weighting_coefficients,
)

_FIELD_IDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class DeviceBridgeMixin:
    """Bridges a CPU precursor-nodes class to the CUDA sub-grid solver."""

    def attach_device(self, parent):
        """Builds pack-spec tables and device/pinned buffers.

        Args:
            parent: CUDASubgridUpdates instance owning the main-grid module
                (pack kernel, main-grid device field arrays, pycuda handles).
        """
        self._parent = parent
        self._drv = parent.drv
        self._gpuarray = parent.grid.gpuarray
        # Device buffers must match the kernels' $REAL (the host precursor
        # arrays are always float64 regardless of the configured precision)
        self._real = config.sim_config.dtypes["float_or_double"]

        self.e_weights = (0.0, 1.0)
        self.h_weights = (0.0, 1.0)

        self._pack = {}
        self._build_pack("e", self.electric_slices)
        self._build_pack("h", self.magnetic_slices)

        self._fine = {}
        self._build_fine("e", self.fn_e)
        self._build_fine("h", self.fn_m)

    # ------------------------------------------------------------------
    # Coarse-plane gather (device -> host)
    # ------------------------------------------------------------------

    def _build_pack(self, kind, descriptors):
        """Builds the int32 spec table for pack_planes and the scatter list
        mapping each packed plane back to a host-array slice assignment."""
        specs = []
        scatter = []
        offset = 0
        for obj in descriptors:
            field = obj[-1]
            fid = next(
                i for i, name in enumerate(_FIELD_IDS) if getattr(self, name) is field
            )
            for slc in obj[2:-1]:
                starts = []
                counts = []
                for ax in slc:
                    if isinstance(ax, slice):
                        starts.append(int(ax.start))
                        counts.append(int(ax.stop - ax.start))
                    else:
                        starts.append(int(ax))
                        counts.append(1)
                size = counts[0] * counts[1] * counts[2]
                specs.append([fid] + starts + counts + [offset])
                # numpy squeezes int-indexed axes on assignment
                shape = tuple(c for ax, c in zip(slc, counts) if isinstance(ax, slice))
                scatter.append((field, slc, offset, size, shape))
                offset += size

        spec_arr = np.array(specs, dtype=np.int32)
        self._pack[kind] = {
            "n_specs": np.int32(len(specs)),
            "total": np.int32(offset),
            "specs_dev": self._gpuarray.to_gpu(spec_arr),
            "packed_dev": self._gpuarray.zeros(offset, dtype=self._real),
            "pinned": self._drv.pagelocked_empty(offset, dtype=self._real),
            "scatter": scatter,
        }

    def _refresh_host_planes(self, kind):
        """Gathers the coarse surface planes on device and scatters them
        into the host main-grid arrays at their slice positions."""
        p = self._pack[kind]
        g = self._parent.grid
        self._parent.pack_planes_dev(
            p["n_specs"],
            p["total"],
            p["specs_dev"].gpudata,
            p["packed_dev"].gpudata,
            g.Ex_dev.gpudata,
            g.Ey_dev.gpudata,
            g.Ez_dev.gpudata,
            g.Hx_dev.gpudata,
            g.Hy_dev.gpudata,
            g.Hz_dev.gpudata,
            block=(128, 1, 1),
            grid=(int(np.ceil(int(p["total"]) / 128)), 1, 1),
        )
        self._drv.memcpy_dtoh(p["pinned"], p["packed_dev"].gpudata)
        for field, slc, off, size, shape in p["scatter"]:
            field[slc] = p["pinned"][off : off + size].reshape(shape)

    # ------------------------------------------------------------------
    # Fine _0/_1 buffers (host -> device)
    # ------------------------------------------------------------------

    def _build_fine(self, kind, names):
        offsets = {}
        offset = 0
        for name in names:
            arr = getattr(self, f"{name}_1")
            offsets[name] = (offset, arr.shape)
            offset += arr.size
        self._fine[kind] = {
            "offsets": offsets,
            "size": offset,
            "dev_0": self._gpuarray.zeros(offset, dtype=self._real),
            "dev_1": self._gpuarray.zeros(offset, dtype=self._real),
            "pinned": self._drv.pagelocked_empty(offset, dtype=self._real),
        }

    def _upload_fine(self, kind, names):
        """Swaps the device _0/_1 buffers and uploads the freshly
        interpolated host _1 arrays into the new _1 buffer."""
        f = self._fine[kind]
        f["dev_0"], f["dev_1"] = f["dev_1"], f["dev_0"]
        pinned = f["pinned"]
        for name in names:
            off, _ = f["offsets"][name]
            arr = getattr(self, f"{name}_1")
            pinned[off : off + arr.size] = arr.ravel()
        self._drv.memcpy_htod(f["dev_1"].gpudata, pinned)

    def dev_ptrs(self, kind, name):
        """Returns (ptr_0, ptr_1, row_len) for a fine face buffer."""
        f = self._fine[kind]
        off, shape = f["offsets"][name]
        itemsize = np.dtype(self._real).itemsize
        p0 = np.uintp(int(f["dev_0"].gpudata) + off * itemsize)
        p1 = np.uintp(int(f["dev_1"].gpudata) + off * itemsize)
        return p0, p1, np.int32(shape[1])

    # ------------------------------------------------------------------
    # Overridden precursor entry points
    # ------------------------------------------------------------------

    def update_electric(self):
        self._refresh_host_planes("e")
        super().update_electric()
        self._upload_fine("e", self.fn_e)

    def update_magnetic(self):
        self._refresh_host_planes("h")
        super().update_magnetic()
        self._upload_fine("h", self.fn_m)

    # Temporal blending is fused into the update_is kernels - only the
    # scalar weights are recorded here.

    def interpolate_electric_in_time(self, m):
        self.e_weights = calculate_weighting_coefficients(m, self.ratio)

    def interpolate_magnetic_in_time(self, m):
        self.h_weights = calculate_weighting_coefficients(m, self.ratio)

    def calc_exact_electric_in_time(self):
        self.e_weights = (0.0, 1.0)

    def calc_exact_magnetic_in_time(self):
        self.h_weights = (0.0, 1.0)


class CUDAPrecursorNodesBridge(DeviceBridgeMixin, PrecursorNodes):
    pass


class CUDAPrecursorNodesFilteredBridge(DeviceBridgeMixin, PrecursorNodesFiltered):
    pass


class DeviceResidentMixin(DeviceBridgeMixin):
    """Fully device-resident precursors (Phase 2): the FIR filter and
    transverse weighting collapse into per-plane gather weights, and the
    SciPy spline becomes precomputed separable weight matrices applied in
    two stages (tmp = Wx . coarse, fine = tmp . Wy^T) built by pushing
    unit vectors through the exact SciPy code path - exact for any
    interpolation degree. The staging is a hard requirement, not a
    nicety: a one-pass Wx . C . Wy^T evaluation is O(n_cx*n_cy) per fine
    node and at flagship face sizes costs more than the volume updates
    of every grid combined. Steady state does no PCIe transfers at
    all."""

    def attach_device(self, parent):
        self._parent = parent
        self._drv = parent.drv
        self._gpuarray = parent.grid.gpuarray
        # Device buffers must match the kernels' $REAL (the host precursor
        # arrays are always float64 regardless of the configured precision)
        self._real = config.sim_config.dtypes["float_or_double"]

        self.e_weights = (0.0, 1.0)
        self.h_weights = (0.0, 1.0)

        self._fine = {}
        self._build_fine("e", self.fn_e)
        self._build_fine("h", self.fn_m)

        self._wcache = {}
        self._stage = {}
        self._build_stage("e", self.electric_slices, self.fn_e)
        self._build_stage("h", self.magnetic_slices, self.fn_m)

    # ------------------------------------------------------------------
    # Spec/weight construction (once per model build)
    # ------------------------------------------------------------------

    def _plane_weights(self, obj, n_planes):
        """Per-plane combination weights: FIR taps and, for H, the
        transverse bracketing-pair weights (mirrors get_transverse_* and
        the CPU update_magnetic weighting)."""
        name = obj[0]
        if name.startswith("e"):
            return [1.0] if n_planes == 1 else [0.25, 0.5, 0.25]
        w = self.l_weight if ("left" in name or "bottom" in name or "front" in name) else self.r_weight
        c1, c2 = calculate_weighting_coefficients(w, self.ratio)
        if n_planes == 2:
            return [c1, c2]
        return [0.25 * c1, 0.5 * c1 + 0.25 * c2, 0.25 * c1 + 0.5 * c2, 0.25 * c2]

    def _weight_matrices(self, coords):
        """Separable spline weight matrices for one face's coordinate set,
        probed through the exact SciPy interpolation path."""
        x, z, x_sg, z_sg = coords
        key = (len(x), len(z), float(x[0]), float(z[0]))
        if key in self._wcache:
            return self._wcache[key]
        n_cx, n_cy = len(x), len(z)
        n_fx, n_fy = len(x_sg), len(z_sg)
        Wx = np.empty((n_fx, n_cx))
        Wy = np.empty((n_fy, n_cy))
        for p in range(n_cx):
            F = np.zeros((n_cx, n_cy))
            F[p, :] = 1.0
            Wx[:, p] = self.interpolate_to_sub_grid(F, coords)[:, 0]
        for q in range(n_cy):
            F = np.zeros((n_cx, n_cy))
            F[:, q] = 1.0
            Wy[:, q] = self.interpolate_to_sub_grid(F, coords)[0, :]
        # Exactness check against the SciPy path on random data
        rng = np.random.default_rng(0)
        F = rng.standard_normal((n_cx, n_cy))
        ref = self.interpolate_to_sub_grid(F, coords)
        got = Wx @ F @ Wy.T
        scale = np.max(np.abs(ref)) or 1.0
        if np.max(np.abs(got - ref)) > 1e-11 * scale:
            raise ValueError("separable weight matrices do not reproduce SciPy")
        self._wcache[key] = (Wx, Wy)
        return Wx, Wy

    def _build_stage(self, kind, descriptors, names):
        gspecs = []
        gweights = []
        ispecs = []
        wx_all = []
        wyt_all = []
        coarse_off = 0
        wx_off = 0
        wy_off = 0
        fine_offsets = self._fine[kind]["offsets"]

        for obj in descriptors:
            field = obj[-1]
            fid = next(
                i for i, n in enumerate(_FIELD_IDS) if getattr(self, n) is field
            )
            planes = obj[2:-1]
            n_planes = len(planes)
            weights = self._plane_weights(obj, n_planes)
            starts = []
            counts = None
            for slc in planes:
                st, ct = [], []
                for ax in slc:
                    if isinstance(ax, slice):
                        st.append(int(ax.start))
                        ct.append(int(ax.stop - ax.start))
                    else:
                        st.append(int(ax))
                        ct.append(1)
                starts.append(st)
                counts = ct
            while len(starts) < 4:
                starts.append(starts[0])
            weights = weights + [0.0] * (4 - n_planes)
            size = counts[0] * counts[1] * counts[2]

            gspecs.append(
                [fid]
                + starts[0] + starts[1] + starts[2] + starts[3]
                + counts
                + [coarse_off, n_planes]
            )
            gweights.append(weights)

            coords = obj[1]
            Wx, Wy = self._weight_matrices(coords)
            n_fx, n_cx = Wx.shape
            n_fy, n_cy = Wy.shape
            assert n_cx * n_cy == size, "coarse plane/coord size mismatch"
            fine_name = obj[0][:-2]  # strip the _1 suffix
            fine_off, fine_shape = fine_offsets[fine_name]
            assert (n_fx, n_fy) == tuple(fine_shape), "fine shape mismatch"
            # Wy is stored transposed (n_cy x n_fy) for coalesced stage-2
            # reads; the packed offset accounting is unaffected.
            ispecs.append([coarse_off, int(fine_off), n_cx, n_cy, n_fx, n_fy, wx_off, wy_off])
            wx_all.append(Wx.ravel())
            wyt_all.append(np.ascontiguousarray(Wy.T).ravel())
            wx_off += Wx.size
            wy_off += Wy.size
            coarse_off += size

        # The offset-scans in the kernels need ascending output offsets;
        # assigning the tmp offsets in fine-offset order keeps both the
        # stage-2 (fine, col 1) and stage-1 (tmp, col 8) keys ascending.
        ispecs.sort(key=lambda s: s[1])
        tmp_total = 0
        for s in ispecs:
            s.append(tmp_total)
            tmp_total += s[4] * s[3]  # n_fx * n_cy

        self._stage[kind] = {
            "n_specs": np.int32(len(gspecs)),
            "coarse_total": np.int32(coarse_off),
            "fine_total": np.int32(self._fine[kind]["size"]),
            "tmp_total": np.int32(tmp_total),
            "gspecs_dev": self._gpuarray.to_gpu(np.array(gspecs, dtype=np.int32)),
            "gweights_dev": self._gpuarray.to_gpu(
                np.array(gweights, dtype=self._real)
            ),
            "coarse_dev": self._gpuarray.zeros(coarse_off, dtype=self._real),
            "tmp_dev": self._gpuarray.zeros(tmp_total, dtype=self._real),
            "ispecs_dev": self._gpuarray.to_gpu(np.array(ispecs, dtype=np.int32)),
            "wx_dev": self._gpuarray.to_gpu(
                np.concatenate(wx_all).astype(self._real)
            ),
            "wyt_dev": self._gpuarray.to_gpu(
                np.concatenate(wyt_all).astype(self._real)
            ),
        }

    # ------------------------------------------------------------------
    # Snapshots: three kernel launches, zero transfers
    # ------------------------------------------------------------------

    def _snapshot(self, kind):
        f = self._fine[kind]
        f["dev_0"], f["dev_1"] = f["dev_1"], f["dev_0"]
        st = self._stage[kind]
        g = self._parent.grid
        self._parent.gather_weighted_planes_dev(
            st["n_specs"],
            st["coarse_total"],
            st["gspecs_dev"].gpudata,
            st["gweights_dev"].gpudata,
            st["coarse_dev"].gpudata,
            g.Ex_dev.gpudata,
            g.Ey_dev.gpudata,
            g.Ez_dev.gpudata,
            g.Hx_dev.gpudata,
            g.Hy_dev.gpudata,
            g.Hz_dev.gpudata,
            block=(128, 1, 1),
            grid=(int(np.ceil(int(st["coarse_total"]) / 128)), 1, 1),
        )
        self._parent.interp_stage1_dev(
            st["n_specs"],
            st["tmp_total"],
            st["ispecs_dev"].gpudata,
            st["wx_dev"].gpudata,
            st["coarse_dev"].gpudata,
            st["tmp_dev"].gpudata,
            block=(128, 1, 1),
            grid=(int(np.ceil(int(st["tmp_total"]) / 128)), 1, 1),
        )
        self._parent.interp_stage2_dev(
            st["n_specs"],
            st["fine_total"],
            st["ispecs_dev"].gpudata,
            st["wyt_dev"].gpudata,
            st["tmp_dev"].gpudata,
            f["dev_1"].gpudata,
            block=(128, 1, 1),
            grid=(int(np.ceil(int(st["fine_total"]) / 128)), 1, 1),
        )

    def update_electric(self):
        self._snapshot("e")

    def update_magnetic(self):
        self._snapshot("h")


class CUDAPrecursorNodes(DeviceResidentMixin, PrecursorNodes):
    pass


class CUDAPrecursorNodesFiltered(DeviceResidentMixin, PrecursorNodesFiltered):
    pass
