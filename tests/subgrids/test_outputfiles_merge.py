import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from toolboxes.Utilities.outputfiles_merge import get_output_data, get_rx_locations, merge_files

MAIN_ITERATIONS = 8
MAIN_DT = 3e-12
RATIO = 3
SG_ITERATIONS = MAIN_ITERATIONS * RATIO
SG_DT = MAIN_DT / RATIO


def write_synthetic_outputfile(filename, main_trace, subgrid_trace=None):
    """Writes a minimal gprMax output file with one receiver in the main grid
        and, optionally, one receiver in a subgrid, following the layout
        written by gprMax.fields_outputs.write_hdf5_outputfile.

    Args:
        filename: string or Path of file to write.
        main_trace: array of samples for the main grid receiver.
        subgrid_trace: array of samples for the subgrid receiver.
    """
    with h5py.File(filename, "w") as f:
        f.attrs["gprMax"] = "test"
        f.attrs["Title"] = "test"
        f.attrs["Iterations"] = MAIN_ITERATIONS
        f.attrs["nx_ny_nz"] = (10, 10, 10)
        f.attrs["dx_dy_dz"] = (0.001, 0.001, 0.001)
        f.attrs["dt"] = MAIN_DT
        f.attrs["nsrc"] = 1
        f.attrs["nrx"] = 1
        f.attrs["srcsteps"] = (0, 0, 0)
        f.attrs["rxsteps"] = (0, 0, 0)

        rx = f.create_group("/rxs/rx1")
        rx.attrs["Name"] = "rxmain"
        rx.attrs["Position"] = (0.005, 0.005, 0.005)
        rx["Ez"] = main_trace

        if subgrid_trace is not None:
            sg = f.create_group("/subgrids/subgrid1")
            sg.attrs["nx_ny_nz"] = (30, 30, 30)
            sg.attrs["dx_dy_dz"] = (0.001 / RATIO,) * 3
            sg.attrs["dt"] = SG_DT
            sg.attrs["nsrc"] = 0
            sg.attrs["nrx"] = 1
            sg.attrs["Iterations"] = SG_ITERATIONS
            sg.attrs["ratio"] = RATIO

            sgrx = sg.create_group("rxs/rx1")
            sgrx.attrs["Name"] = "rxbowtie"
            sgrx.attrs["Position"] = (0.003, 0.003, 0.003)
            sgrx["Ez"] = subgrid_trace


@pytest.fixture
def outputfiles(tmp_path):
    """Two synthetic output files (a two trace B-scan), with distinct
    traces so merged columns can be identified."""
    files = []
    traces = []
    for i in range(2):
        main_trace = np.arange(MAIN_ITERATIONS, dtype=np.float32) + 100 * i
        subgrid_trace = np.arange(SG_ITERATIONS, dtype=np.float32) + 1000 * i
        filename = tmp_path / f"test{i + 1}.h5"
        write_synthetic_outputfile(filename, main_trace, subgrid_trace)
        files.append(str(filename))
        traces.append((main_trace, subgrid_trace))
    return files, traces


class TestMergeFiles:
    def test_merges_main_grid_and_subgrid_receivers(self, outputfiles, tmp_path):
        files, traces = outputfiles
        merged = str(tmp_path / "test_merged.h5")

        merge_files(files, merged_outputfile=merged)

        with h5py.File(merged, "r") as f:
            main = f["/rxs/rx1/Ez"][:]
            sg = f["/subgrids/subgrid1/rxs/rx1/Ez"][:]

            assert main.shape == (MAIN_ITERATIONS, 2)
            assert sg.shape == (SG_ITERATIONS, 2)
            for i, (main_trace, subgrid_trace) in enumerate(traces):
                assert_array_equal(main[:, i], main_trace)
                assert_array_equal(sg[:, i], subgrid_trace)

            # Subgrid meta data should be preserved so traces can be
            # plotted against the correct time axis
            sggrp = f["/subgrids/subgrid1"]
            assert sggrp.attrs["Iterations"] == SG_ITERATIONS
            assert sggrp.attrs["dt"] == pytest.approx(SG_DT)
            assert sggrp.attrs["ratio"] == RATIO
            assert f["/subgrids/subgrid1/rxs/rx1"].attrs["Name"] == "rxbowtie"

    def test_decimate_subgrids_matches_main_grid_time_axis(self, outputfiles, tmp_path):
        files, traces = outputfiles
        merged = str(tmp_path / "test_merged.h5")

        merge_files(files, merged_outputfile=merged, decimate_subgrids=True)

        with h5py.File(merged, "r") as f:
            sg = f["/subgrids/subgrid1/rxs/rx1/Ez"][:]

            assert sg.shape == (MAIN_ITERATIONS, 2)
            for i, (_, subgrid_trace) in enumerate(traces):
                assert_array_equal(sg[:, i], subgrid_trace[::RATIO])

            sggrp = f["/subgrids/subgrid1"]
            assert sggrp.attrs["Iterations"] == MAIN_ITERATIONS
            assert sggrp.attrs["dt"] == pytest.approx(MAIN_DT)
            assert sggrp.attrs["decimated"]

    def test_merges_files_without_subgrids(self, tmp_path):
        files = []
        for i in range(2):
            filename = tmp_path / f"test{i + 1}.h5"
            write_synthetic_outputfile(filename, np.arange(MAIN_ITERATIONS, dtype=np.float32))
            files.append(str(filename))
        merged = str(tmp_path / "test_merged.h5")

        merge_files(files, merged_outputfile=merged)

        with h5py.File(merged, "r") as f:
            assert f["/rxs/rx1/Ez"].shape == (MAIN_ITERATIONS, 2)
            assert "subgrids" not in f


class TestGetOutputData:
    def test_receiver_numbering_spans_main_grid_and_subgrids(self, outputfiles, tmp_path):
        files, traces = outputfiles
        merged = str(tmp_path / "test_merged.h5")
        merge_files(files, merged_outputfile=merged)

        with h5py.File(merged, "r") as f:
            assert len(get_rx_locations(f)) == 2

        outputdata, dt = get_output_data(merged, 1, "Ez")
        assert dt == pytest.approx(MAIN_DT)
        assert_array_equal(outputdata[:, 0], traces[0][0])

        outputdata, dt = get_output_data(merged, 2, "Ez")
        assert dt == pytest.approx(SG_DT)
        assert_array_equal(outputdata[:, 0], traces[0][1])

    def test_decimated_subgrid_receiver_has_main_grid_dt(self, outputfiles, tmp_path):
        files, _ = outputfiles
        merged = str(tmp_path / "test_merged.h5")
        merge_files(files, merged_outputfile=merged, decimate_subgrids=True)

        outputdata, dt = get_output_data(merged, 2, "Ez")
        assert dt == pytest.approx(MAIN_DT)
        assert outputdata.shape[0] == MAIN_ITERATIONS

    def test_out_of_range_receiver_raises(self, outputfiles):
        files, _ = outputfiles
        with pytest.raises(ValueError):
            get_output_data(files[0], 3, "Ez")
