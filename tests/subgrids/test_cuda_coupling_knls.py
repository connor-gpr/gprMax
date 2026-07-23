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

"""Unit tests for the CUDA sub-grid coupling kernels against their Cython
twins (the fp64 oracle), on randomised fields over the full argument
matrix: all faces, both offsets/mid values, r in {3, 5}, non-cubic
domains, and both the HSG-style (s=3) and SHSG-style (s=0) argument
sets. Requires a CUDA device + nvcc; skipped otherwise."""

import numpy as np
import pytest

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.compiler
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    from jinja2 import Environment, PackageLoader

    _mod_cache = {}

    def _have_cuda():
        try:
            pycuda.compiler.SourceModule("__global__ void t() {}")
            return True
        except Exception:
            return False

    HAVE_CUDA = _have_cuda()
except ImportError:
    HAVE_CUDA = False

pytestmark = pytest.mark.skipif(not HAVE_CUDA, reason="CUDA device/toolchain unavailable")

from gprMax.cuda_opencl import knl_subgrid_coupling
from gprMax.cython.fields_updates_hsg import (
    update_electric_os,
    update_is,
    update_magnetic_os,
)

NY_MAT = 5
N_MATS = 8


def build_module(field_shape, n_id=6):
    """Compiles the coupling kernels for a grid whose field arrays have
    field_shape and whose ID array is (n_id,) + field_shape."""
    key = (field_shape, n_id)
    if key in _mod_cache:
        return _mod_cache[key]
    env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
    common = env.get_template("knl_common_cuda.tmpl").render(
        REAL="double",
        N_updatecoeffsE=N_MATS * NY_MAT,
        N_updatecoeffsH=N_MATS * NY_MAT,
        NY_MATCOEFFS=NY_MAT,
        NY_MATDISPCOEFFS=1,
        NX_FIELDS=field_shape[0],
        NY_FIELDS=field_shape[1],
        NZ_FIELDS=field_shape[2],
        NX_ID=field_shape[0],
        NY_ID=field_shape[1],
        NZ_ID=field_shape[2],
        NX_T=1,
        NY_T=1,
        NZ_T=1,
        NY_RXCOORDS=3,
        NX_RXS=6,
        NY_RXS=1,
        NZ_RXS=1,
        NY_SRCINFO=4,
        NY_SRCWAVES=1,
        NX_SNAPS=1,
        NY_SNAPS=1,
        NZ_SNAPS=1,
    )
    subs_args = {"REAL": "double", "COMPLEX": "pycuda::complex<double>"}
    subs_func = {
        "REAL": "double",
        "CUDA_IDX": "int i = blockIdx.x * blockDim.x + threadIdx.x;",
    }
    parts = [common]
    for kf in (
        knl_subgrid_coupling.update_is_e,
        knl_subgrid_coupling.update_is_h,
        knl_subgrid_coupling.update_electric_os,
        knl_subgrid_coupling.update_magnetic_os,
        knl_subgrid_coupling.pack_planes,
    ):
        parts.append(
            kf["args_cuda"].substitute(subs_args)
            + "{"
            + kf["func"].substitute(subs_func)
            + "}"
        )
    mod = pycuda.compiler.SourceModule("\n".join(parts), options=["-w"])
    _mod_cache[key] = mod
    return mod


def rand_coeffs(rng):
    return rng.standard_normal((N_MATS, NY_MAT))


def upload_coeffs(mod, coeffsE, coeffsH):
    drv.memcpy_htod(mod.get_global("updatecoeffsE")[0], np.ascontiguousarray(coeffsE))
    drv.memcpy_htod(mod.get_global("updatecoeffsH")[0], np.ascontiguousarray(coeffsH))


def bpg(total):
    return (int(np.ceil(total / 128)), 1, 1)


@pytest.mark.parametrize("r", [3, 5])
@pytest.mark.parametrize("face", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, -1])
@pytest.mark.parametrize("variant", ["e", "h"])
def test_update_is_matches_cython(r, face, offset, variant):
    rng = np.random.default_rng(42 + r + face * 10 + offset + (variant == "h"))
    # Non-cubic SHSG-like fine domain: nb = s*r + 1, ow* = nw* + 2*s*r
    s = 1
    nb = s * r + 1
    nwx, nwy, nwz = 2 * r, 3 * r, 4 * r
    owx, owy, owz = nwx + 2 * s * r, nwy + 2 * s * r, nwz + 2 * s * r
    n = nb - s * r  # os_f
    shape = (2 * nb + nwx + 1, 2 * nb + nwy + 1, 2 * nb + nwz + 1)

    # nwl/nwm per face mirror the SHSG call tables (first call per face)
    nwl, nwm = {1: (owx, owy + 1), 2: (owy, owz + 1), 3: (owx, owz + 1)}[face]

    field = rng.standard_normal(shape)
    ID = rng.integers(0, N_MATS, size=(6,) + shape).astype(np.uint32)
    coeffsE = rand_coeffs(rng)
    coeffsH = rand_coeffs(rng)
    inc_l_0 = rng.standard_normal((nwl, nwm))
    inc_l_1 = rng.standard_normal((nwl, nwm))
    inc_u_0 = rng.standard_normal((nwl, nwm))
    inc_u_1 = rng.standard_normal((nwl, nwm))
    c1, c2 = 2.0 / 3.0, 1.0 / 3.0
    lookup_id, sign_l, sign_u, co = 2, 1, -1, 3

    # CPU oracle: blend on host, then Cython
    inc_l = c1 * inc_l_0 + c2 * inc_l_1
    inc_u = c1 * inc_u_0 + c2 * inc_u_1
    field_cpu = field.copy()
    coeffs = coeffsE if variant == "e" else coeffsH
    update_is(
        owx, owy, owz, coeffs, ID, n, offset, nwl, nwm, 0, face,
        field_cpu, inc_l, inc_u, lookup_id, sign_l, sign_u, co, 1,
    )

    mod = build_module(shape)
    upload_coeffs(mod, coeffsE, coeffsH)
    knl = mod.get_function(f"update_is_{variant}")
    field_dev = gpuarray.to_gpu(field)
    knl(
        np.int32(owx), np.int32(owy), np.int32(owz), np.int32(n),
        np.int32(offset), np.int32(nwl), np.int32(nwm), np.int32(face),
        gpuarray.to_gpu(ID).gpudata, field_dev.gpudata,
        gpuarray.to_gpu(inc_l_0).gpudata, gpuarray.to_gpu(inc_l_1).gpudata,
        gpuarray.to_gpu(inc_u_0).gpudata, gpuarray.to_gpu(inc_u_1).gpudata,
        np.float64(c1), np.float64(c2), np.int32(nwm),
        np.int32(lookup_id), np.int32(sign_l), np.int32(sign_u), np.int32(co),
        block=(128, 1, 1), grid=bpg(nwl * nwm),
    )
    np.testing.assert_allclose(field_dev.get(), field_cpu, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("r", [3, 5])
@pytest.mark.parametrize("face", [1, 2, 3])
@pytest.mark.parametrize("s", [0, 3])
@pytest.mark.parametrize("mid", [0, 1])
@pytest.mark.parametrize("variant", ["electric", "magnetic"])
def test_update_os_matches_cython(r, face, s, mid, variant):
    rng = np.random.default_rng(7 + r + face * 10 + s * 100 + mid + len(variant))
    # Main grid box (IS box for s=0, OS-derived for s=3) and fine subgrid
    nw = {1: 4, 2: 3, 3: 5}  # coarse working cells per axis (non-cubic)
    i0, j0, k0 = 6, 7, 8
    i1, j1, k1 = i0 + nw[1], j0 + nw[2], k0 + nw[3]
    main_shape = (i1 + s + 4, j1 + s + 4, k1 + s + 4)
    nb = s * r + r // 2 + 4  # generic halo big enough for both styles
    nwx, nwy, nwz = nw[1] * r, nw[2] * r, nw[3] * r
    sub_shape = (2 * nb + nwx + 1, 2 * nb + nwy + 1, 2 * nb + nwz + 1)

    # First call per face of the SHSG/HSG tables (electric variant), and
    # its magnetic twin: near/far normal indices n_l/n_u
    if face == 3:
        l_l, l_u, m_l, m_u = i0, i1, k0, k1 + 1
        n_l, n_u, nwn = j0, j1, nwy
    elif face == 2:
        l_l, l_u, m_l, m_u = j0, j1, k0, k1 + 1
        n_l, n_u, nwn = i0, i1, nwx
    else:
        l_l, l_u, m_l, m_u = i0, i1, j0, j1 + 1
        n_l, n_u, nwn = k0, k1, nwz
    if variant == "magnetic":
        n_l = n_l - 1

    field = rng.standard_normal(main_shape)
    ID = rng.integers(0, N_MATS, size=(6,) + main_shape).astype(np.uint32)
    inc_field = rng.standard_normal(sub_shape)
    coeffsE = rand_coeffs(rng)
    coeffsH = rand_coeffs(rng)
    lookup_id, co, sign_n, sign_f = 1, 2, 1, -1

    field_cpu = field.copy()
    if variant == "electric":
        update_electric_os(
            coeffsE, ID, face, l_l, l_u, m_l, m_u, n_l, n_u, nwn,
            lookup_id, field_cpu, inc_field, co, sign_n, sign_f, mid, r, s, nb, 1,
        )
    else:
        update_magnetic_os(
            coeffsH, ID, face, l_l, l_u, m_l, m_u, n_l, n_u, nwn,
            lookup_id, field_cpu, inc_field, co, sign_n, sign_f, mid, r, s, nb, 1,
        )

    mod = build_module(main_shape)
    upload_coeffs(mod, coeffsE, coeffsH)
    knl = mod.get_function(f"update_{variant}_os")
    field_dev = gpuarray.to_gpu(field)
    knl(
        np.int32(face), np.int32(l_l), np.int32(l_u), np.int32(m_l),
        np.int32(m_u), np.int32(n_l), np.int32(n_u), np.int32(nwn),
        gpuarray.to_gpu(ID).gpudata, field_dev.gpudata,
        gpuarray.to_gpu(inc_field).gpudata,
        np.int32(sub_shape[1]), np.int32(sub_shape[2]),
        np.int32(lookup_id), np.int32(co), np.int32(sign_n), np.int32(sign_f),
        np.int32(mid), np.int32(r), np.int32(s), np.int32(nb),
        block=(128, 1, 1), grid=bpg((l_u - l_l) * (m_u - m_l)),
    )
    np.testing.assert_allclose(field_dev.get(), field_cpu, rtol=1e-14, atol=1e-15)


def test_pack_planes_matches_numpy():
    rng = np.random.default_rng(3)
    shape = (14, 12, 10)
    fields = [rng.standard_normal(shape) for _ in range(6)]

    boxes = [
        (0, (2, slice(1, 9), slice(2, 8))),
        (4, (slice(3, 11), 5, slice(0, 9))),
        (5, (slice(0, 13), slice(2, 10), 7)),
        (2, (slice(1, 3), slice(1, 3), slice(1, 3))),
    ]
    specs = []
    expected = []
    offset = 0
    for fid, slc in boxes:
        starts, counts = [], []
        for ax in slc:
            if isinstance(ax, slice):
                starts.append(ax.start)
                counts.append(ax.stop - ax.start)
            else:
                starts.append(ax)
                counts.append(1)
        specs.append([fid] + starts + counts + [offset])
        expected.append(fields[fid][slc].ravel())
        offset += counts[0] * counts[1] * counts[2]
    expected = np.concatenate(expected)

    mod = build_module(shape)
    knl = mod.get_function("pack_planes")
    packed_dev = gpuarray.zeros(offset, dtype=np.float64)
    field_devs = [gpuarray.to_gpu(f) for f in fields]
    knl(
        np.int32(len(specs)), np.int32(offset),
        gpuarray.to_gpu(np.array(specs, dtype=np.int32)).gpudata,
        packed_dev.gpudata,
        *[f.gpudata for f in field_devs],
        block=(128, 1, 1), grid=bpg(offset),
    )
    np.testing.assert_array_equal(packed_dev.get(), expected)
