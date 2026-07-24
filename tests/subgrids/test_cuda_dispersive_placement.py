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

"""Per-grid dispersive dispatch gate: a small SHSG model with the Debye
water target placed in the main grid, the sub-grid, or both, is solved
on CPU and CUDA and the rx traces compared. The CUDA run additionally
asserts that ONLY the grids that contain the dispersive material
allocate T arrays / take the dispersive kernel path (the model-level
maxpoles is non-zero in every case, so pre-fix code would have put every
grid on the dispersive path). Each solve runs in a subprocess so the two
solvers get clean config/context state. Requires CUDA; skipped
otherwise."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CASES = {
    # case -> grids expected to hold device dispersive arrays
    "main": {"main_grid"},
    "sg": {"sg"},
    "both": {"main_grid", "sg"},
}


def build_scene(case):
    import gprMax

    scene = gprMax.Scene()
    dl = 6e-3
    scene.add(gprMax.Title(name=f"disp_placement_{case}"))
    scene.add(gprMax.Domain(p1=(0.24, 0.24, 0.24)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1.2e-9))
    scene.add(gprMax.Waveform(wave_type="gaussiandot", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(polarisation="z", p1=(0.072, 0.12, 0.12), waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.18, 0.126, 0.12)))

    sg = gprMax.SubGridSHSG(
        p1=(0.102, 0.102, 0.102), p2=(0.15, 0.15, 0.15), ratio=3, id="sg"
    )
    scene.add(sg)
    sg.add(gprMax.Rx(p1=(0.126, 0.126, 0.126)))

    def add_debye(container, mid, p1, p2):
        # Mild single-pole Debye (moist-soil-like): dispersive enough to
        # exercise the dispersive path, mild enough that the main grid's
        # 6 mm cells pass the numerical-dispersion check
        container.add(gprMax.Material(er=5, se=0.01, mr=1, sm=0, id=mid))
        container.add(
            gprMax.AddDebyeDispersion(
                poles=1, er_delta=[2.0], tau=[9.4e-12], material_ids=[mid]
            )
        )
        container.add(gprMax.Box(p1=p1, p2=p2, material_id=mid))

    if case in ("main", "both"):
        # Outside the sub-grid OS and clear of the PML
        add_debye(scene, "debye_m", (0.066, 0.066, 0.066), (0.09, 0.09, 0.09))
    if case in ("sg", "both"):
        add_debye(sg, "debye_sg", (0.114, 0.114, 0.114), (0.138, 0.138, 0.138))
    return scene


def run_case(case, solver, outdir):
    import gprMax

    out = Path(outdir) / f"disp_{case}_{solver}"
    kwargs = dict(
        scenes=[build_scene(case)], n=1, outputfile=out, subgrid=True, autotranslate=True
    )
    if solver == "cpu":
        gprMax.run(**kwargs)
        return

    from gprMax.grid.cuda_grid import CUDAArrayMixin

    seen = {}
    orig = CUDAArrayMixin.htod_geometry_arrays

    def spy(self):
        seen[self.name] = self
        orig(self)

    CUDAArrayMixin.htod_geometry_arrays = spy
    kwargs.update(gpu=[0], subgrid_gpu=True)
    gprMax.run(**kwargs)

    expected = CASES[case]
    dispersive = {n for n, g in seen.items() if hasattr(g, "Tx_dev")}
    assert dispersive == expected, (
        f"device dispersive arrays on {sorted(dispersive)}, expected {sorted(expected)}"
    )
    for n, g in seen.items():
        if n not in expected:
            assert not hasattr(g, "Tx"), f"[{n}] allocated host dispersive arrays"
    print(f"dispersive grids on device: {sorted(dispersive)} == {sorted(expected)}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    run_case(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0)


# ----------------------------------------------------------------------
# pytest wrapper
# ----------------------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.compiler

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


def _traces(path):
    import h5py

    out = {}
    with h5py.File(path, "r") as f:

        def cb(name, obj):
            if isinstance(obj, h5py.Dataset):
                arr = np.asarray(obj[...])
                if arr.size and np.issubdtype(arr.dtype, np.floating):
                    out[name] = arr.astype(np.float64)

        f.visititems(cb)
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_dispersive_placement_matches_cpu(case, tmp_path):
    for solver in ("cpu", "gpu"):
        r = subprocess.run(
            [sys.executable, __file__, case, solver, str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            cwd=str(REPO),
        )
        assert r.returncode == 0, (
            f"{solver} run failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )

    cpu = _traces(tmp_path / f"disp_{case}_cpu.h5")
    gpu = _traces(tmp_path / f"disp_{case}_gpu.h5")
    assert cpu.keys() == gpu.keys()

    # Group-scale normalisation: near-zero (symmetry) components are
    # judged against their rx group's dominant amplitude, not their own
    # noise-level range
    gscale = {}
    for k, v in cpu.items():
        g = k.rsplit("/", 1)[0]
        gscale[g] = max(gscale.get(g, 0.0), np.abs(v).max())

    worst = 0.0
    for k in cpu:
        scale = gscale[k.rsplit("/", 1)[0]]
        if scale == 0.0:
            assert not gpu[k].any(), f"{k}: CPU silent but GPU non-zero"
            continue
        nrmse = np.sqrt(np.mean((cpu[k] - gpu[k]) ** 2)) / scale
        worst = max(worst, nrmse)
    assert worst < 1e-9, f"worst group-scale NRMSE {worst:.3e} (case {case})"
