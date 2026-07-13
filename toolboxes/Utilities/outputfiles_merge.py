# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import argparse
import glob
import logging
import os

import h5py
import numpy as np

from gprMax._version import __version__
from gprMax.utilities.utilities import natural_keys

logger = logging.getLogger(__name__)


def get_rx_locations(f):
    """Lists locations of all receivers in an output file - receivers in the
        main grid followed by receivers in any subgrids.

    Args:
        f: h5py file object of an open output file.

    Returns:
        rxlocations: list of dicts, one per receiver, each with the HDF5 group
            path of the receiver, and the temporal resolution (dt) and number
            of iterations of the grid holding the receiver.
    """

    rxlocations = []
    for rx in range(1, f.attrs["nrx"] + 1):
        rxlocations.append(
            {
                "path": "/rxs/rx" + str(rx),
                "dt": f.attrs["dt"],
                "iterations": f.attrs["Iterations"],
            }
        )

    if "/subgrids" in f:
        sgnames = sorted(f["/subgrids"].keys(), key=natural_keys)
        for sgname in sgnames:
            sg = f["/subgrids/" + sgname]
            for rx in range(1, sg.attrs["nrx"] + 1):
                rxlocations.append(
                    {
                        "path": "/subgrids/" + sgname + "/rxs/rx" + str(rx),
                        "dt": sg.attrs["dt"],
                        "iterations": sg.attrs["Iterations"],
                    }
                )

    return rxlocations


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename: string of tilename (including path) of output file.
        rxnumber: int of receiver output number. Receivers in the main grid
            are numbered first, followed by receivers in any subgrids.
        rxcomponent: string of receiver output field/current component.

    Returns:
        outputdata: array of A-scans, i.e. B-scan data.
        dt: float of temporal resolution of the grid holding the receiver.
    """

    # Open output file and read some attributes
    with h5py.File(filename, "r") as f:
        rxlocations = get_rx_locations(f)

        # Check there are any receivers
        if not rxlocations:
            logger.exception(f"No receivers found in {filename}")
            raise ValueError

        if rxnumber < 1 or rxnumber > len(rxlocations):
            logger.exception(
                f"Receiver {rxnumber} requested, but {filename} "
                + f"contains {len(rxlocations)} receiver(s)"
            )
            raise ValueError

        rxlocation = rxlocations[rxnumber - 1]
        dt = rxlocation["dt"]
        path = rxlocation["path"] + "/"
        availableoutputs = list(f[path].keys())

        # Check if requested output is in file
        if rxcomponent not in availableoutputs:
            logger.exception(
                f"{rxcomponent} output requested to plot, but the "
                + f"available output for receiver 1 is "
                + f"{', '.join(availableoutputs)}"
            )
            raise ValueError

        outputdata = f[path + "/" + rxcomponent]
        outputdata = np.array(outputdata)

    return outputdata, dt


def merge_files(outputfiles, merged_outputfile=None, removefiles=False, decimate_subgrids=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files. Receivers in the
        main grid and in any subgrids are merged.

    Args:
        outputfiles: list of output files to be merged.
        removefiles: boolean flag to remove individual output files after merge.
        merged_outputfile: string or Path object of location to save the merged
        output. If not specified a default location is used.
        decimate_subgrids: boolean flag to decimate subgrid traces by the
            subgrid ratio, so they share a time axis with main grid traces.
    """

    if merged_outputfile is None:
        merged_outputfile = os.path.commonprefix(outputfiles) + "_merged.h5"

    # Combined output file
    fout = h5py.File(merged_outputfile, "w")

    for i, outputfile in enumerate(outputfiles):
        fin = h5py.File(outputfile, "r")
        nrx = fin.attrs["nrx"]
        sgnames = sorted(fin["/subgrids"].keys(), key=natural_keys) if "/subgrids" in fin else []

        # Write properties for merged file on first iteration
        if i == 0:
            fout.attrs["gprMax"] = __version__
            fout.attrs["Iterations"] = fin.attrs["Iterations"]
            fout.attrs["nx_ny_nz"] = fin.attrs["nx_ny_nz"]
            fout.attrs["dx_dy_dz"] = fin.attrs["dx_dy_dz"]
            fout.attrs["dt"] = fin.attrs["dt"]
            fout.attrs["nsrc"] = fin.attrs["nsrc"]
            fout.attrs["nrx"] = fin.attrs["nrx"]
            fout.attrs["srcsteps"] = fin.attrs["srcsteps"]
            fout.attrs["rxsteps"] = fin.attrs["rxsteps"]

            for rx in range(1, nrx + 1):
                path = "/rxs/rx" + str(rx)
                grp = fout.create_group(path)
                for attr, value in fin[path].attrs.items():
                    grp.attrs[attr] = value
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    grp.create_dataset(
                        output,
                        (fout.attrs["Iterations"], len(outputfiles)),
                        dtype=fin[path + "/" + output].dtype,
                    )

            for sgname in sgnames:
                sgin = fin["/subgrids/" + sgname]
                sggrp = fout.create_group("/subgrids/" + sgname)
                for attr, value in sgin.attrs.items():
                    sggrp.attrs[attr] = value
                sgiterations = sgin.attrs["Iterations"]
                if decimate_subgrids:
                    ratio = sgin.attrs["ratio"]
                    sgiterations = len(range(0, sgiterations, ratio))
                    sggrp.attrs["Iterations"] = sgiterations
                    sggrp.attrs["dt"] = sgin.attrs["dt"] * ratio
                    sggrp.attrs["decimated"] = True

                for rx in range(1, sgin.attrs["nrx"] + 1):
                    path = "/subgrids/" + sgname + "/rxs/rx" + str(rx)
                    grp = fout.create_group(path)
                    for attr, value in fin[path].attrs.items():
                        grp.attrs[attr] = value
                    availableoutputs = list(fin[path].keys())
                    for output in availableoutputs:
                        grp.create_dataset(
                            output,
                            (sgiterations, len(outputfiles)),
                            dtype=fin[path + "/" + output].dtype,
                        )

        # For all receivers in the main grid
        for rx in range(1, nrx + 1):
            path = "/rxs/rx" + str(rx) + "/"
            availableoutputs = list(fin[path].keys())
            # For all receiver outputs
            for output in availableoutputs:
                fout[path + "/" + output][:, i] = fin[path + "/" + output][:]

        # For all receivers in any subgrids
        for sgname in sgnames:
            sgin = fin["/subgrids/" + sgname]
            ratio = sgin.attrs["ratio"]
            for rx in range(1, sgin.attrs["nrx"] + 1):
                path = "/subgrids/" + sgname + "/rxs/rx" + str(rx) + "/"
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    trace = fin[path + "/" + output][:]
                    if decimate_subgrids:
                        trace = trace[::ratio]
                    fout[path + "/" + output][:, i] = trace

        fin.close()
    fout.close()

    if removefiles:
        for outputfile in outputfiles:
            os.remove(outputfile)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Merges traces (A-scans) from multiple "
        + "output files into one new file, then "
        + "optionally removes the series of output files.",
        usage="cd gprMax; python -m tools.outputfiles_merge basefilename",
    )
    parser.add_argument("basefilename", help="base name of output file series including path")
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        type=str,
        required=False,
        help="location to save merged file",
    )
    parser.add_argument(
        "--remove-files",
        action="store_true",
        default=False,
        help="flag to remove individual output files after merge",
    )
    parser.add_argument(
        "--decimate-subgrids",
        action="store_true",
        default=False,
        help="flag to decimate subgrid traces by the subgrid ratio, so they "
        + "share a time axis with main grid traces",
    )
    args = parser.parse_args()

    files = glob.glob(args.basefilename + "*.h5")
    outputfiles = [
        filename for filename in files if "_merged" not in filename and args.output_file != filename
    ]
    outputfiles.sort(key=natural_keys)
    merge_files(
        outputfiles,
        merged_outputfile=args.output_file,
        removefiles=args.remove_files,
        decimate_subgrids=args.decimate_subgrids,
    )
