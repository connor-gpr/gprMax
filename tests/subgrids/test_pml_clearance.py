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

"""Tests for the subgrid Outer Surface to main grid PML clearance check."""

import logging
from types import SimpleNamespace

from gprMax.subgrids.user_objects import os_pml_gaps, warn_if_os_close_to_pml


def make_sg(i0, j0, k0, i1, j1, k1, is_os_sep=3):
    """Stand-in for a subgrid with main grid placement indices set."""
    return SimpleNamespace(i0=i0, j0=j0, k0=k0, i1=i1, j1=j1, k1=k1, is_os_sep=is_os_sep)


def make_grid(nx, ny, nz, thickness=10):
    """Stand-in for a main FDTDGrid with uniform PML thickness."""
    return SimpleNamespace(
        nx=nx,
        ny=ny,
        nz=nz,
        pmls={
            "thickness": {
                "x0": thickness,
                "y0": thickness,
                "z0": thickness,
                "xmax": thickness,
                "ymax": thickness,
                "zmax": thickness,
            }
        },
    )


def test_gaps_all_faces():
    # The measured_2_antennas_gpr_btg subgrid model before its domain was
    # enlarged: 1.2 x 1.2 x 1.602m at 6mm with the Inner Surface at cells
    # 14-186 (x, y) and 215-247 (z). The OS (IS +/- 3) ended up 1 cell from
    # the PML on all four lateral faces, which caused late-time instability.
    sg = make_sg(14, 14, 215, 186, 186, 247)
    grid = make_grid(200, 200, 267)
    gaps = os_pml_gaps(sg, grid)
    assert gaps == {"x0": 1, "y0": 1, "z0": 202, "xmax": 1, "ymax": 1, "zmax": 7}


def test_gaps_negative_when_os_inside_pml():
    sg = make_sg(11, 40, 40, 60, 60, 60)
    grid = make_grid(100, 100, 100)
    gaps = os_pml_gaps(sg, grid)
    assert gaps["x0"] == -2
    assert gaps["y0"] == 27


def test_gaps_respect_per_face_thickness():
    sg = make_sg(40, 40, 40, 60, 60, 60)
    grid = make_grid(100, 100, 100, thickness=10)
    grid.pmls["thickness"]["zmax"] = 20
    gaps = os_pml_gaps(sg, grid)
    assert gaps["zmax"] == (100 - 20) - 63
    assert gaps["xmax"] == (100 - 10) - 63


def test_warns_and_returns_close_faces(caplog):
    sg = make_sg(14, 14, 215, 186, 186, 247)
    grid = make_grid(200, 200, 267)
    with caplog.at_level(logging.WARNING, logger="gprMax.subgrids.user_objects"):
        close = warn_if_os_close_to_pml(sg, grid, "#subgrid_hsg:")
    assert close == {"x0": 1, "y0": 1, "xmax": 1, "ymax": 1, "zmax": 7}
    assert len(caplog.records) == 1
    assert "Outer Surface" in caplog.text
    assert "x0: 1" in caplog.text


def test_no_warning_when_clear(caplog):
    sg = make_sg(40, 40, 40, 60, 60, 60)
    grid = make_grid(100, 100, 100)
    with caplog.at_level(logging.WARNING, logger="gprMax.subgrids.user_objects"):
        close = warn_if_os_close_to_pml(sg, grid, "#subgrid_hsg:")
    assert close == {}
    assert not caplog.records
