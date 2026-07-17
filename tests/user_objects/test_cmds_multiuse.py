import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.cmds_multiuse import ExcitationFile


class StubSimConfig:
    dtypes = {"float_or_double": np.float64}


@pytest.fixture
def sim_config(monkeypatch):
    monkeypatch.setattr(config, "sim_config", StubSimConfig())


@pytest.fixture
def grid():
    grid = FDTDGrid()
    grid.dt = 1.0
    grid.timewindow = 2.0
    return grid


def write_excitation_file(tmp_path, text):
    excitationfile = tmp_path / "excitation.txt"
    excitationfile.write_text(text)
    return excitationfile


def test_single_waveform_without_time_column(tmp_path, sim_config, grid):
    excitationfile = write_excitation_file(tmp_path, "mypulse\n0.0\n1.0\n0.5\n")

    ExcitationFile(filepath=excitationfile).build(grid)

    assert [w.ID for w in grid.waveforms] == ["mypulse"]
    assert grid.waveforms[0].type == "user"
    assert_allclose(grid.waveforms[0].userfunc([0.0, 1.0, 2.0]), [0.0, 1.0, 0.5])


def test_single_waveform_with_time_column(tmp_path, sim_config, grid):
    excitationfile = write_excitation_file(
        tmp_path, "time mypulse\n0.0 0.0\n1.0 2.0\n2.0 4.0\n"
    )

    ExcitationFile(filepath=excitationfile).build(grid)

    assert [w.ID for w in grid.waveforms] == ["mypulse"]
    assert_allclose(grid.waveforms[0].userfunc(0.5), 1.0)


def test_multiple_waveforms_select_correct_columns(tmp_path, sim_config, grid):
    excitationfile = write_excitation_file(
        tmp_path, "time wave1 wave2\n0.0 0.0 5.0\n1.0 1.0 6.0\n2.0 2.0 7.0\n"
    )

    ExcitationFile(filepath=excitationfile).build(grid)

    assert [w.ID for w in grid.waveforms] == ["wave1", "wave2"]
    assert_allclose(grid.waveforms[0].userfunc(1.0), 1.0)
    assert_allclose(grid.waveforms[1].userfunc(1.0), 6.0)


def test_numeric_fill_value_used_beyond_file_end(tmp_path, sim_config, grid):
    excitationfile = write_excitation_file(
        tmp_path, "time mypulse\n0.0 0.0\n1.0 2.0\n2.0 4.0\n"
    )

    ExcitationFile(filepath=excitationfile, kind="linear", fill_value=0.0).build(grid)

    # Times beyond the file must return the fill value, not raise - interp1d
    # only defaults bounds_error to False for fill_value="extrapolate"
    assert_allclose(grid.waveforms[0].userfunc([1.5, 2.5, 100.0]), [3.0, 0.0, 0.0])


def test_single_data_row(tmp_path, sim_config, grid):
    excitationfile = write_excitation_file(tmp_path, "wave1 wave2\n1.0 2.0\n")

    ExcitationFile(filepath=excitationfile).build(grid)

    assert [w.ID for w in grid.waveforms] == ["wave1", "wave2"]
    # Values are zero-padded to the length of the simulation time array
    assert_allclose(grid.waveforms[0].userfunc([0.0, 1.0]), [1.0, 0.0])
    assert_allclose(grid.waveforms[1].userfunc([0.0, 1.0]), [2.0, 0.0])
