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

"""CUDA kernels for device-resident precursor nodes (Phase 2).

Replaces the CPU FIR/transverse-weighting/SciPy-spline pipeline:

- gather_weighted_planes: per face-component, gathers up to 4 coarse
  main-grid planes and combines them with fixed weights (the 3-tap FIR
  and the transverse H weighting collapse into per-plane weights
  computed once at init), writing the packed coarse buffer.
- interp_faces: per face-component, applies the precomputed separable
  spline weight matrices (fine = Wx . coarse . Wy^T), writing directly
  into the packed fine _1 buffer used by the update_is kernels. The
  matrices are built on host by pushing unit vectors through the exact
  SciPy code path, so any interpolation degree is reproduced.

Both are compiled into the main-grid module set (gather reads main-grid
fields via the baked IDX3D_FIELDS); all per-face geometry arrives in
spec tables, so a single launch covers all faces of one field type.
"""

from string import Template

# int32 geometry specs, one row of 18 per face-component:
#  0: fieldid (0-5 = Ex..Hz)
#  1-12: i,j,k start of up to 4 source planes (unused planes repeat
#        plane 0 and get weight 0)
#  13-15: extents n0, n1, n2 of the (identically shaped) planes
#  16: output offset into the packed coarse buffer
#  17: number of planes actually used
# REAL weights, one row of 4 per face-component.
gather_weighted_planes = {
    "args_cuda": Template(
        """
                __global__ void gather_weighted_planes(int n_specs,
                                int total,
                                const int* __restrict__ specs,
                                const $REAL* __restrict__ weights,
                                $REAL *coarse,
                                const $REAL* __restrict__ Ex,
                                const $REAL* __restrict__ Ey,
                                const $REAL* __restrict__ Ez,
                                const $REAL* __restrict__ Hx,
                                const $REAL* __restrict__ Hy,
                                const $REAL* __restrict__ Hz)
                    """
    ),
    "func": Template(
        """
    // Weighted plane gather: coarse[out] = sum_m w_m * plane_m[node]

    $CUDA_IDX

    if (i >= total) return;

    int sidx = 0;
    while (sidx + 1 < n_specs && specs[(sidx + 1) * 18 + 16] <= i) sidx++;
    const int* sp = specs + sidx * 18;
    const $REAL* w = weights + sidx * 4;

    int local = i - sp[16];
    int n1 = sp[14];
    int n2 = sp[15];
    int d0 = local / (n1 * n2);
    int rem = local % (n1 * n2);
    int d1 = rem / n2;
    int d2 = rem % n2;

    const $REAL* f = (sp[0] == 0) ? Ex
                   : (sp[0] == 1) ? Ey
                   : (sp[0] == 2) ? Ez
                   : (sp[0] == 3) ? Hx
                   : (sp[0] == 4) ? Hy
                   : Hz;

    $REAL val = 0.0;
    for (int m = 0; m < sp[17]; m++) {
        int x = sp[1 + 3 * m] + d0;
        int y = sp[2 + 3 * m] + d1;
        int z = sp[3 + 3 * m] + d2;
        val += w[m] * f[IDX3D_FIELDS(x, y, z)];
    }
    coarse[i] = val;
    """
    ),
}

# int32 specs, one row of 8 per face-component:
#  0: coarse offset  1: fine offset  2: n_cx  3: n_cy  4: n_fx  5: n_fy
#  6: Wx offset (row-major n_fx x n_cx)  7: Wy offset (n_fy x n_cy)
interp_faces = {
    "args_cuda": Template(
        """
                __global__ void interp_faces(int n_specs,
                                int total,
                                const int* __restrict__ specs,
                                const $REAL* __restrict__ wx,
                                const $REAL* __restrict__ wy,
                                const $REAL* __restrict__ coarse,
                                $REAL *fine)
                    """
    ),
    "func": Template(
        """
    // Separable spline interpolation: fine[a,b] = Wx[a,:] . C . Wy[b,:]^T

    $CUDA_IDX

    if (i >= total) return;

    int sidx = 0;
    while (sidx + 1 < n_specs && specs[(sidx + 1) * 8 + 1] <= i) sidx++;
    const int* sp = specs + sidx * 8;

    int local = i - sp[1];
    int n_cx = sp[2];
    int n_cy = sp[3];
    int n_fy = sp[5];
    int a = local / n_fy;
    int b = local % n_fy;

    const $REAL* wxr = wx + sp[6] + (size_t)a * n_cx;
    const $REAL* wyr = wy + sp[7] + (size_t)b * n_cy;
    const $REAL* c = coarse + sp[0];

    $REAL val = 0.0;
    for (int p = 0; p < n_cx; p++) {
        $REAL row = 0.0;
        for (int q = 0; q < n_cy; q++) {
            row += c[(size_t)p * n_cy + q] * wyr[q];
        }
        val += wxr[p] * row;
    }
    fine[i] = val;
    """
    ),
}
