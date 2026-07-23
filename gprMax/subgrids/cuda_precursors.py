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
        self._real = self.ex_front_1.dtype.type

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
