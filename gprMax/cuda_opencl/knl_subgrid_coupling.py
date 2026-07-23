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

"""CUDA kernels coupling a Huygens sub-grid to the main grid.

Ports of cython/fields_updates_hsg.pyx (update_is, update_electric_os,
update_magnetic_os) plus a plane-gather kernel used to feed the CPU-side
precursor interpolation with a single packed transfer.

Cross-grid indexing contract: each kernel is compiled into the module set
of the grid it WRITES, so the baked IDX3D_FIELDS/IDX4D_ID macros and the
__constant__ updatecoeffs arrays resolve to that grid. Every array
belonging to the OTHER grid is indexed through dimensions passed as
runtime kernel arguments (pre_nm for the 2-D precursor buffers, inc_ny/
inc_nz for the 3-D sub-grid fields). The update equations and sign
conventions are identical to the Cython kernels; HSG vs SHSG is purely
the launch-argument set (see SHSG_CONVENTIONS.md).

Temporal blending is fused into update_is: instead of a host-side blended
array, the kernel takes the previous (_0) and current (_1) precursor
buffers and scalar weights c1/c2 (c1=0, c2=1 for the exact-time case).
"""

from string import Template


def _update_is(suffix, coeffs):
    """update_is writes the sub-grid ring (SHSG: the OS ring) from the
    interpolated main-grid precursor buffers. Compiled into the sub-grid
    module: field/ID macros and __constant__ coefficients are the
    sub-grid's own (loss rows included). One thread handles the matching
    node on the lower and upper face of the given axis, exactly like one
    (l, m) iteration of the Cython kernel."""
    return {
        "args_cuda": Template(
            """
                __global__ void update_is_"""
            + suffix
            + """(int nwx,
                                int nwy,
                                int nwz,
                                int n,
                                int offset,
                                int nwl,
                                int nwm,
                                int face,
                                const unsigned int* __restrict__ ID,
                                $REAL *field,
                                const $REAL* __restrict__ inc_l_0,
                                const $REAL* __restrict__ inc_l_1,
                                const $REAL* __restrict__ inc_u_0,
                                const $REAL* __restrict__ inc_u_1,
                                $REAL c1,
                                $REAL c2,
                                int pre_nm,
                                int lookup_id,
                                int sign_l,
                                int sign_u,
                                int co)
                    """
        ),
        "func": Template(
            """
    // Sub-grid ring correction (Cython update_is port). One thread per
    // (l, m) face node; writes the lower- and upper-face nodes.

    $CUDA_IDX

    int total = nwl * nwm;
    if (i >= total) return;

    int l = n + i / nwm;
    int m = n + i % nwm;
    // For inner faces H nodes are 1 cell before n boundary cells
    int n_o = n + offset;

    int i1, j1, k1, i2, j2, k2;
    if (face == 1) {
        // Bottom and top
        i1 = l; j1 = m; k1 = n_o;
        i2 = l; j2 = m; k2 = n + nwz;
    } else if (face == 2) {
        // Left and right
        i1 = n_o; j1 = l; k1 = m;
        i2 = n + nwx; j2 = l; k2 = m;
    } else {
        // Front and back
        i1 = l; j1 = n_o; k1 = m;
        i2 = l; j2 = n + nwy; k2 = m;
    }

    // Precursor buffers are 2-D row-major with runtime row length pre_nm
    int pidx = (l - n) * pre_nm + (m - n);

    $REAL incl = c1 * inc_l_0[pidx] + c2 * inc_l_1[pidx];
    int mat_l = ID[IDX4D_ID(lookup_id, i1, j1, k1)];
    field[IDX3D_FIELDS(i1, j1, k1)] += """
            + coeffs
            + """[IDX2D_MAT(mat_l, co)] * incl * sign_l;

    $REAL incu = c1 * inc_u_0[pidx] + c2 * inc_u_1[pidx];
    int mat_u = ID[IDX4D_ID(lookup_id, i2, j2, k2)];
    field[IDX3D_FIELDS(i2, j2, k2)] += """
            + coeffs
            + """[IDX2D_MAT(mat_u, co)] * incu * sign_u;
    """
        ),
    }


update_is_e = _update_is("e", "updatecoeffsE")
update_is_h = _update_is("h", "updatecoeffsH")


def _update_os(kind):
    """update_electric_os / update_magnetic_os write the main grid (SHSG:
    the IS ring) from collocated sub-grid samples. Compiled into the
    main-grid module: field/ID macros and __constant__ coefficients are
    the main grid's own (le/lm loss rows included). The sub-grid
    inc_field is foreign - its y/z dimensions arrive as runtime args.
    One thread per (l, m) node of the main-grid face slice; writes the
    near- and far-face nodes."""
    if kind == "electric":
        coeffs = "updatecoeffsE"
        # Surface normal fine indices of the subgrid near/far face H nodes
        n_s_l = "nb - s * r - r + r / 2"
        n_s_r = "nb + nwn + s * r + r / 2"
    else:
        coeffs = "updatecoeffsH"
        # Fine indices of the subgrid near/far face E nodes
        n_s_l = "nb - r * s"
        n_s_r = "nb + nwn + s * r"
    return {
        "args_cuda": Template(
            """
                __global__ void update_"""
            + kind
            + """_os(int face,
                                int l_l,
                                int l_u,
                                int m_l,
                                int m_u,
                                int n_l,
                                int n_u,
                                int nwn,
                                const unsigned int* __restrict__ ID,
                                $REAL *field,
                                const $REAL* __restrict__ inc_field,
                                int inc_ny,
                                int inc_nz,
                                int lookup_id,
                                int co,
                                int sign_n,
                                int sign_f,
                                int mid,
                                int r,
                                int s,
                                int nb)
                    """
        ),
        "func": Template(
            """
    // Main-grid ring correction (Cython update_"""
            + kind
            + """_os port).

    $CUDA_IDX

    int ml_len = m_u - m_l;
    int total = (l_u - l_l) * ml_len;
    if (i >= total) return;

    int l = l_l + i / ml_len;
    int m = m_l + i % ml_len;

    int n_s_l = """
            + n_s_l
            + """;
    int n_s_r = """
            + n_s_r
            + """;
    // OS at the near face (fine index)
    int os = nb - r * s;

    int l_s = os + (l - l_l) * r + (mid == 1 ? r / 2 : 0);
    int m_s = os + (m - m_l) * r + (mid == 1 ? 0 : r / 2);

    int i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;
    if (face == 2) {
        // Left and right: main grid index / equivalent subgrid index
        i0 = n_l; j0 = l; k0 = m;
        i1 = n_s_l; j1 = l_s; k1 = m_s;
        i2 = n_u; j2 = l; k2 = m;
        i3 = n_s_r; j3 = l_s; k3 = m_s;
    } else if (face == 3) {
        // Front and back
        i0 = l; j0 = n_l; k0 = m;
        i1 = l_s; j1 = n_s_l; k1 = m_s;
        i2 = l; j2 = n_u; k2 = m;
        i3 = l_s; j3 = n_s_r; k3 = m_s;
    } else {
        // Bottom and top
        i0 = l; j0 = m; k0 = n_l;
        i1 = l_s; j1 = m_s; k1 = n_s_l;
        i2 = l; j2 = m; k2 = n_u;
        i3 = l_s; j3 = m_s; k3 = n_s_r;
    }

    // Near face
    int mat_n = ID[IDX4D_ID(lookup_id, i0, j0, k0)];
    $REAL inc_n = inc_field[((size_t)(i1) * inc_ny + (j1)) * inc_nz + (k1)] * sign_n;
    field[IDX3D_FIELDS(i0, j0, k0)] += """
            + coeffs
            + """[IDX2D_MAT(mat_n, co)] * inc_n;

    // Far face
    int mat_f = ID[IDX4D_ID(lookup_id, i2, j2, k2)];
    $REAL inc_f = inc_field[((size_t)(i3) * inc_ny + (j3)) * inc_nz + (k3)] * sign_f;
    field[IDX3D_FIELDS(i2, j2, k2)] += """
            + coeffs
            + """[IDX2D_MAT(mat_f, co)] * inc_f;
    """
        ),
    }


update_electric_os = _update_os("electric")
update_magnetic_os = _update_os("magnetic")


def _update_is_faces(suffix, coeffs):
    """Fused all-faces variant of update_is: one launch covers the six
    per-face calls of an SHSG ring update. Per-row int32 spec (12 wide):
    face, nwl, nwm, field_sel(0-2), inc_l_off, inc_u_off, pre_nm,
    lookup_id, sign_l, sign_u, co, out_off (ascending thread partition).
    The _0/_1 fine buffers arrive as base pointers (they pointer-swap
    each snapshot); per-face offsets are baked into the spec."""
    return {
        "args_cuda": Template(
            """
                __global__ void update_is_faces_"""
            + suffix
            + """(int nwx,
                                int nwy,
                                int nwz,
                                int n,
                                int offset,
                                int n_specs,
                                int total,
                                const int* __restrict__ specs,
                                const unsigned int* __restrict__ ID,
                                $REAL *F0,
                                $REAL *F1,
                                $REAL *F2,
                                const $REAL* __restrict__ fine_0,
                                const $REAL* __restrict__ fine_1,
                                $REAL c1,
                                $REAL c2)
                    """
        ),
        "func": Template(
            """
    // Fused sub-grid ring correction: all six faces in one launch.

    $CUDA_IDX

    if (i >= total) return;

    int sidx = 0;
    while (sidx + 1 < n_specs && specs[(sidx + 1) * 12 + 11] <= i) sidx++;
    const int* sp = specs + sidx * 12;

    int local = i - sp[11];
    int face = sp[0];
    int nwm = sp[2];
    int l = n + local / nwm;
    int m = n + local % nwm;
    int n_o = n + offset;

    int i1, j1, k1, i2, j2, k2;
    if (face == 1) {
        i1 = l; j1 = m; k1 = n_o;
        i2 = l; j2 = m; k2 = n + nwz;
    } else if (face == 2) {
        i1 = n_o; j1 = l; k1 = m;
        i2 = n + nwx; j2 = l; k2 = m;
    } else {
        i1 = l; j1 = n_o; k1 = m;
        i2 = l; j2 = n + nwy; k2 = m;
    }

    $REAL *field = (sp[3] == 0) ? F0 : (sp[3] == 1) ? F1 : F2;
    int pidx = (l - n) * sp[6] + (m - n);
    int lookup_id = sp[7];
    int co = sp[10];

    $REAL incl = c1 * fine_0[sp[4] + pidx] + c2 * fine_1[sp[4] + pidx];
    int mat_l = ID[IDX4D_ID(lookup_id, i1, j1, k1)];
    field[IDX3D_FIELDS(i1, j1, k1)] += """
            + coeffs
            + """[IDX2D_MAT(mat_l, co)] * incl * sp[8];

    $REAL incu = c1 * fine_0[sp[5] + pidx] + c2 * fine_1[sp[5] + pidx];
    int mat_u = ID[IDX4D_ID(lookup_id, i2, j2, k2)];
    field[IDX3D_FIELDS(i2, j2, k2)] += """
            + coeffs
            + """[IDX2D_MAT(mat_u, co)] * incu * sp[9];
    """
        ),
    }


update_is_faces_e = _update_is_faces("e", "updatecoeffsE")
update_is_faces_h = _update_is_faces("h", "updatecoeffsH")


def _update_os_faces(kind):
    """Fused all-faces variant of update_*_os: one launch covers the six
    per-face IS-ring corrections. Per-row int32 spec (16 wide): face,
    l_l, l_u, m_l, m_u, n_l, n_u, nwn, lookup_id, field_sel(0-2),
    inc_sel(0-2), co, sign_n, sign_f, mid, out_off."""
    if kind == "electric":
        coeffs = "updatecoeffsE"
        n_s_l = "nb - s * r - r + r / 2"
        n_s_r = "nb + nwn + s * r + r / 2"
    else:
        coeffs = "updatecoeffsH"
        n_s_l = "nb - r * s"
        n_s_r = "nb + nwn + s * r"
    return {
        "args_cuda": Template(
            """
                __global__ void update_os_faces_"""
            + kind
            + """(int n_specs,
                                int total,
                                const int* __restrict__ specs,
                                const unsigned int* __restrict__ ID,
                                $REAL *F0,
                                $REAL *F1,
                                $REAL *F2,
                                const $REAL* __restrict__ I0,
                                const $REAL* __restrict__ I1,
                                const $REAL* __restrict__ I2,
                                int inc_ny,
                                int inc_nz,
                                int r,
                                int s,
                                int nb)
                    """
        ),
        "func": Template(
            """
    // Fused main-grid ring correction: all six faces in one launch.

    $CUDA_IDX

    if (i >= total) return;

    int sidx = 0;
    while (sidx + 1 < n_specs && specs[(sidx + 1) * 16 + 15] <= i) sidx++;
    const int* sp = specs + sidx * 16;

    int local = i - sp[15];
    int face = sp[0];
    int l_l = sp[1];
    int m_l = sp[3];
    int ml_len = sp[4] - m_l;
    int l = l_l + local / ml_len;
    int m = m_l + local % ml_len;
    int n_l = sp[5];
    int n_u = sp[6];
    int nwn = sp[7];
    int mid = sp[14];

    int n_s_l = """
            + n_s_l
            + """;
    int n_s_r = """
            + n_s_r
            + """;
    int os = nb - r * s;

    int l_s = os + (l - l_l) * r + (mid == 1 ? r / 2 : 0);
    int m_s = os + (m - m_l) * r + (mid == 1 ? 0 : r / 2);

    int i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;
    if (face == 2) {
        i0 = n_l; j0 = l; k0 = m;
        i1 = n_s_l; j1 = l_s; k1 = m_s;
        i2 = n_u; j2 = l; k2 = m;
        i3 = n_s_r; j3 = l_s; k3 = m_s;
    } else if (face == 3) {
        i0 = l; j0 = n_l; k0 = m;
        i1 = l_s; j1 = n_s_l; k1 = m_s;
        i2 = l; j2 = n_u; k2 = m;
        i3 = l_s; j3 = n_s_r; k3 = m_s;
    } else {
        i0 = l; j0 = m; k0 = n_l;
        i1 = l_s; j1 = m_s; k1 = n_s_l;
        i2 = l; j2 = m; k2 = n_u;
        i3 = l_s; j3 = m_s; k3 = n_s_r;
    }

    $REAL *field = (sp[9] == 0) ? F0 : (sp[9] == 1) ? F1 : F2;
    const $REAL* inc = (sp[10] == 0) ? I0 : (sp[10] == 1) ? I1 : I2;
    int lookup_id = sp[8];
    int co = sp[11];

    int mat_n = ID[IDX4D_ID(lookup_id, i0, j0, k0)];
    $REAL inc_n = inc[((size_t)(i1) * inc_ny + (j1)) * inc_nz + (k1)] * sp[12];
    field[IDX3D_FIELDS(i0, j0, k0)] += """
            + coeffs
            + """[IDX2D_MAT(mat_n, co)] * inc_n;

    int mat_f = ID[IDX4D_ID(lookup_id, i2, j2, k2)];
    $REAL inc_f = inc[((size_t)(i3) * inc_ny + (j3)) * inc_nz + (k3)] * sp[13];
    field[IDX3D_FIELDS(i2, j2, k2)] += """
            + coeffs
            + """[IDX2D_MAT(mat_f, co)] * inc_f;
    """
        ),
    }


update_os_faces_electric = _update_os_faces("electric")
update_os_faces_magnetic = _update_os_faces("magnetic")


# Gathers an arbitrary set of axis-aligned boxes ("planes") of the six
# main-grid field arrays into one contiguous buffer, enabling a single
# packed device-to-host transfer per precursor snapshot. Compiled into
# the main-grid module. specs is int32 (n_specs x 8): fieldid (0-5 for
# Ex..Hz), i0, j0, k0, ni, nj, nk, out_offset; entries sorted by
# ascending out_offset with no gaps.
pack_planes = {
    "args_cuda": Template(
        """
                __global__ void pack_planes(int n_specs,
                                int total,
                                const int* __restrict__ specs,
                                $REAL *packed,
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
    // Plane gather for packed precursor snapshots.

    $CUDA_IDX

    if (i >= total) return;

    // Find this element's spec (specs are offset-sorted; <=48 entries)
    int sidx = 0;
    while (sidx + 1 < n_specs && specs[(sidx + 1) * 8 + 7] <= i) sidx++;
    const int* sp = specs + sidx * 8;

    int local = i - sp[7];
    int nj = sp[5];
    int nk = sp[6];
    int di = local / (nj * nk);
    int rem = local % (nj * nk);
    int dj = rem / nk;
    int dk = rem % nk;

    int x = sp[1] + di;
    int y = sp[2] + dj;
    int z = sp[3] + dk;

    const $REAL* f = (sp[0] == 0) ? Ex
                   : (sp[0] == 1) ? Ey
                   : (sp[0] == 2) ? Ez
                   : (sp[0] == 3) ? Hx
                   : (sp[0] == 4) ? Hy
                   : Hz;
    packed[i] = f[IDX3D_FIELDS(x, y, z)];
    """
    ),
}
