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

"""Phase 2 gate: the device-resident precursor pipeline (weighted plane
gather + separable weight-matrix interpolation) must reproduce the CPU
PrecursorNodes[Filtered] fine _1 arrays on randomised main-grid fields,
for interpolation in {1, 2, 3} x filter on/off. Requires CUDA; skipped
otherwise."""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.compiler
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray

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

from jinja2 import Environment, PackageLoader

from gprMax.cuda_opencl import knl_precursors
from gprMax.subgrids.cuda_precursors import (
    CUDAPrecursorNodes,
    CUDAPrecursorNodesFiltered,
)
from gprMax.subgrids.precursor_nodes import PrecursorNodes, PrecursorNodesFiltered

_mod_cache = {}


def build_parent(shape):
    """Minimal stand-in for CUDASubgridUpdates: main-grid device arrays and
    the two precursor kernels compiled for the given field shape."""
    key = shape
    if key not in _mod_cache:
        env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
        common = env.get_template("knl_common_cuda.tmpl").render(
            REAL="double",
            N_updatecoeffsE=5,
            N_updatecoeffsH=5,
            NY_MATCOEFFS=5,
            NY_MATDISPCOEFFS=1,
            NX_FIELDS=shape[0],
            NY_FIELDS=shape[1],
            NZ_FIELDS=shape[2],
            NX_ID=shape[0],
            NY_ID=shape[1],
            NZ_ID=shape[2],
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
            knl_precursors.gather_weighted_planes,
            knl_precursors.interp_stage1,
            knl_precursors.interp_stage2,
        ):
            parts.append(
                kf["args_cuda"].substitute(subs_args)
                + "{"
                + kf["func"].substitute(subs_func)
                + "}"
            )
        _mod_cache[key] = pycuda.compiler.SourceModule("\n".join(parts), options=["-w"])
    mod = _mod_cache[key]

    rng = np.random.default_rng(11)
    fields = {n: rng.standard_normal(shape) for n in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    grid = SimpleNamespace(
        gpuarray=gpuarray,
        **fields,
        **{f"{n}_dev": gpuarray.to_gpu(fields[n]) for n in fields},
    )
    return SimpleNamespace(
        drv=drv,
        grid=grid,
        gather_weighted_planes_dev=mod.get_function("gather_weighted_planes"),
        interp_stage1_dev=mod.get_function("interp_stage1"),
        interp_stage2_dev=mod.get_function("interp_stage2"),
    )


@pytest.fixture(autouse=True)
def fake_sim_config(monkeypatch):
    import gprMax.config as config

    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float64}),
        raising=False,
    )


@pytest.mark.parametrize("interpolation", [1, 2, 3])
@pytest.mark.parametrize("filt", [False, True])
def test_device_precursors_match_cpu(interpolation, filt):
    ratio = 3
    surface = SimpleNamespace(
        i0=6, j0=7, k0=8, i1=14, j1=16, k1=18,
        nwx=8 * ratio, nwy=9 * ratio, nwz=10 * ratio,
        ratio=ratio, interpolation=interpolation,
    )
    shape = (24, 26, 28)
    parent = build_parent(shape)
    fdtd_grid = parent.grid  # host arrays live on the same namespace

    cpu_cls = PrecursorNodesFiltered if filt else PrecursorNodes
    dev_cls = CUDAPrecursorNodesFiltered if filt else CUDAPrecursorNodes
    cpu = cpu_cls(fdtd_grid, surface)
    dev = dev_cls(fdtd_grid, surface)
    dev.attach_device(parent)

    cpu.update_electric()
    cpu.update_magnetic()
    dev.update_electric()
    dev.update_magnetic()

    for kind, names in (("e", dev.fn_e), ("h", dev.fn_m)):
        fine = dev._fine[kind]["dev_1"].get()
        for name in names:
            off, shape2 = dev._fine[kind]["offsets"][name]
            got = fine[off : off + shape2[0] * shape2[1]].reshape(shape2)
            want = getattr(cpu, f"{name}_1")
            scale = max(np.max(np.abs(want)), 1e-30)
            np.testing.assert_allclose(
                got, want, rtol=0, atol=1e-12 * scale,
                err_msg=f"{name} interpolation={interpolation} filter={filt}",
            )
