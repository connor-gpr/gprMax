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

"""Gating tests for GPU solvers: subgrid_gpu flag, precision override, and
the transmission-line / discrete-plane-wave feature gates (Phase 0 of the
CUDA sub-gridding work). None of these need a physical GPU - device
detection is monkeypatched."""

import argparse
from types import SimpleNamespace

import pytest

import gprMax.config as config
from gprMax.gprMax import args_defaults


def make_args(**overrides):
    args = dict(args_defaults)
    # SimulationConfig derives file paths from these; give it something
    # harmless (nothing is written - the config is never run).
    args["outputfile"] = "gate_test"
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.fixture
def fake_devices(monkeypatch):
    monkeypatch.setattr(config, "detect_cuda_gpus", lambda: {0: object()})
    monkeypatch.setattr(config, "detect_opencl", lambda: {0: object()})
    monkeypatch.setattr(config, "detect_metal", lambda: {0: object()})


class TestSubgridSolverGates:
    def test_subgrid_cuda_without_flag_raises(self, fake_devices):
        with pytest.raises(ValueError):
            config.SimulationConfig(make_args(gpu=[0], subgrid=True))

    def test_subgrid_cuda_with_flag_allowed_and_double(self, fake_devices):
        sc = config.SimulationConfig(make_args(gpu=[0], subgrid=True, subgrid_gpu=True))
        assert sc.general["subgrid"] is True
        assert sc.general["subgrid_gpu"] is True
        assert sc.general["precision"] == "double"

    def test_subgrid_opencl_raises_with_flag(self, fake_devices):
        with pytest.raises(ValueError):
            config.SimulationConfig(make_args(opencl=[0], subgrid=True, subgrid_gpu=True))

    def test_subgrid_metal_raises_with_flag(self, fake_devices):
        with pytest.raises(ValueError):
            config.SimulationConfig(make_args(metal=[0], subgrid=True, subgrid_gpu=True))

    def test_subgrid_cpu_unaffected(self):
        sc = config.SimulationConfig(make_args(subgrid=True))
        assert sc.general["precision"] == "double"
        assert sc.general["subgrid_gpu"] is False


class TestPrecisionOverride:
    def test_default_no_override(self, fake_devices):
        sc = config.SimulationConfig(make_args(gpu=[0]))
        assert sc.general["precision"] == "single"

    def test_gpu_double_override(self, fake_devices):
        sc = config.SimulationConfig(make_args(gpu=[0], precision="double"))
        assert sc.general["precision"] == "double"

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError):
            config.SimulationConfig(make_args(precision="half"))

    def test_subgrid_single_experimental_allowed(self):
        sc = config.SimulationConfig(make_args(subgrid=True, precision="single"))
        assert sc.general["precision"] == "single"


def make_fake_model(main_tls=0, main_pws=0, sg_tls=0, sg_pws=0, n_subgrids=0):
    def grid(tls, pws):
        return SimpleNamespace(
            transmissionlines=[object()] * tls,
            discreteplanewaves=[object()] * pws,
        )

    subgrids = [grid(sg_tls, sg_pws) for _ in range(n_subgrids)]
    return SimpleNamespace(G=grid(main_tls, main_pws), subgrids=subgrids)


class TestSourceFeatureGates:
    """create_solver must reject sources the device solvers never step."""

    @pytest.fixture
    def solver_config(self, monkeypatch):
        def set_solver(name, subgrid=False):
            monkeypatch.setattr(
                config,
                "sim_config",
                SimpleNamespace(general={"solver": name, "subgrid": subgrid}),
            )

        return set_solver

    def test_tl_on_gpu_raises(self, solver_config):
        from gprMax.solvers import create_solver

        solver_config("cuda")
        with pytest.raises(ValueError):
            create_solver(make_fake_model(main_tls=1))

    def test_subgrid_tl_on_gpu_raises(self, solver_config):
        from gprMax.solvers import create_solver

        solver_config("cuda")
        with pytest.raises(ValueError):
            create_solver(make_fake_model(sg_tls=1, n_subgrids=1))

    def test_plane_wave_on_gpu_raises(self, solver_config):
        from gprMax.solvers import create_solver

        for name in ("cuda", "opencl", "metal"):
            solver_config(name)
            with pytest.raises(ValueError):
                create_solver(make_fake_model(main_pws=1))

    def test_plane_wave_in_subgrid_raises_on_cpu(self, solver_config):
        from gprMax.solvers import create_solver

        solver_config("cpu")
        with pytest.raises(ValueError):
            create_solver(make_fake_model(sg_pws=1, n_subgrids=1))

    def test_tl_on_cpu_not_gated(self, solver_config):
        from gprMax.solvers import create_solver

        solver_config("cpu")
        # Gate must not fire; create_solver then fails later for other
        # reasons (fake model has no real grid), which is fine - we only
        # assert the gate's specific behaviour by getting past it.
        try:
            create_solver(make_fake_model(main_tls=1))
        except ValueError as err:
            assert "Transmission lines" not in str(err)
        except Exception:
            pass
