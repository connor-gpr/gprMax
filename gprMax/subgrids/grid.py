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

import logging
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from gprMax.grid.fdtd_grid import FDTDGrid

logger = logging.getLogger(__name__)


class SubGridBaseGrid(FDTDGrid, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.ratio = kwargs["ratio"]

        if self.ratio % 2 == 0:
            logger.exception("Subgrid Error: Only odd ratios are supported")
            raise ValueError

        # Name of the grid
        self.name = kwargs["id"]
        self.parent_grid: FDTDGrid
        self.iterations = 0

        self.filter = kwargs["filter"]

        # Number of main grid cells between the IS and OS
        self.is_os_sep = kwargs["is_os_sep"]
        # Number of subgrid grid cells between the IS and OS
        self.s_is_os_sep = self.is_os_sep * self.ratio

        # Distance from OS to PML or the edge of the grid when PML is off
        self.pml_separation = kwargs["pml_separation"]

        self.pmls["thickness"]["x0"] = kwargs["subgrid_pml_thickness"]
        self.pmls["thickness"]["y0"] = kwargs["subgrid_pml_thickness"]
        self.pmls["thickness"]["z0"] = kwargs["subgrid_pml_thickness"]
        self.pmls["thickness"]["xmax"] = kwargs["subgrid_pml_thickness"]
        self.pmls["thickness"]["ymax"] = kwargs["subgrid_pml_thickness"]
        self.pmls["thickness"]["zmax"] = kwargs["subgrid_pml_thickness"]

        # Number of sub cells to extend the sub grid beyond the IS boundary
        d_to_pml = self.s_is_os_sep + self.pml_separation
        # Index of the IS
        self.n_boundary_cells = d_to_pml + self.pmls["thickness"]["x0"]
        self.n_boundary_cells_x = d_to_pml + self.pmls["thickness"]["x0"]
        self.n_boundary_cells_y = d_to_pml + self.pmls["thickness"]["y0"]
        self.n_boundary_cells_z = d_to_pml + self.pmls["thickness"]["z0"]

        self.interpolation = kwargs["interpolation"]

    def local_to_global_coordinate(
        self, coord: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        """Maps a local subgrid cell coordinate to the global coordinate system.

        The returned coordinate is a fine (subgrid resolution) cell index
        measured from the main grid origin, i.e. the inverse of
        SubgridUserInput.translate_to_gap. Multiply by self.dl to obtain
        the physical position in the main grid frame.

        Args:
            coord: x, y, z local cell coordinate of the subgrid.

        Returns:
            global_coord: x, y, z fine cell coordinate relative to the
                main grid origin.
        """
        n_boundary_cells = np.array(
            [self.n_boundary_cells_x, self.n_boundary_cells_y, self.n_boundary_cells_z]
        )
        is_corner = np.array([self.i0, self.j0, self.k0]) * self.ratio
        return coord - n_boundary_cells + is_corner

    @abstractmethod
    def update_magnetic_is(self, precursors):
        pass

    @abstractmethod
    def update_electric_is(self, precursors):
        pass

    @abstractmethod
    def update_electric_os(self, main_grid):
        pass

    @abstractmethod
    def update_magnetic_os(self, main_grid):
        pass

    @abstractmethod
    def print_info(self):
        pass
