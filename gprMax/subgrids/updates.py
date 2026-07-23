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

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid

from ..updates.cpu_updates import CPUUpdates
from ..updates.updates import HSGCapable
from .precursor_nodes import PrecursorNodes, PrecursorNodesFiltered
from .subgrid_hsg import SubGridHSG
from .subgrid_shsg import SubGridSHSG

logger = logging.getLogger(__name__)


class OSSurfaceView:
    """Presents a subgrid to the precursor-node classes with its surface
    indices moved from the Inner Surface to the Outer Surface.

    The SHSG interpolates and filters the main-grid fields at the OS
    (the HSG does so at the IS). The precursor classes only read the
    surface box indices and extents from the subgrid, so a shifted view
    reuses them unchanged: the OS box is the IS box expanded by is_os_sep
    main-grid cells, and the fine extents grow accordingly.
    """

    def __init__(self, sg):
        s = sg.is_os_sep
        self.i0, self.j0, self.k0 = sg.i0 - s, sg.j0 - s, sg.k0 - s
        self.i1, self.j1, self.k1 = sg.i1 + s, sg.j1 + s, sg.k1 + s
        self.nwx = sg.nwx + 2 * s * sg.ratio
        self.nwy = sg.nwy + 2 * s * sg.ratio
        self.nwz = sg.nwz + 2 * s * sg.ratio
        self.ratio = sg.ratio
        self.interpolation = sg.interpolation


def create_updates(model: Model):
    """Return the updates object for the given subgrids, dispatched on the
    configured solver. The CPU path is the default; the CUDA path is only
    reachable with subgrid_gpu=True (enforced in config)."""
    solver = config.sim_config.general["solver"]
    if solver == "cuda":
        from .cuda_updates import create_updates as create_cuda_updates

        return create_cuda_updates(model)
    if solver != "cpu":
        # config already rejects these combinations; guard against drift
        logger.exception(f"Sub-grids are not supported with the {solver} solver")
        raise ValueError

    updaters = []

    for sg in model.subgrids:
        sg_type = type(sg)
        if sg_type == SubGridHSG:
            surface = sg
        elif sg_type == SubGridSHSG:
            surface = OSSurfaceView(sg)
            sg.apply_shsg_loss(model.G)
        else:
            logger.exception(f"{str(sg)} is not a subgrid type")
            raise ValueError

        if sg.filter:
            precursors = PrecursorNodesFiltered(model.G, surface)
        else:
            precursors = PrecursorNodes(model.G, surface)

        sgu = SubgridUpdater(sg, precursors, model.G)
        updaters.append(sgu)

    updates = SubgridUpdates(model.G, updaters)
    return updates


class SubgridUpdates(CPUUpdates, HSGCapable):
    """Updates for subgrids."""

    def __init__(self, G, updaters):
        super().__init__(G)
        self.updaters = updaters

    def hsg_1(self):
        """Updates the subgrids over the first phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Updates the subgrids over the second phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_2()


class SubgridUpdater(CPUUpdates[SubGridBaseGrid]):
    """Handles updating the electric and magnetic fields of an HSG subgrid.
    The IS, OS, subgrid region and the electric/magnetic sources are updated
    using the precursor regions.
    """

    def __init__(self, subgrid: SubGridBaseGrid, precursors: PrecursorNodes, G: FDTDGrid):
        """
        Args:
            subgrid: SubGrid3d instance to be updated.
            precursors (PrecursorNodes): PrecursorNodes instance nodes associated
                                            with the subgrid - contain interpolated
                                            fields.
            G: FDTDGrid class describing a grid in a model.
        """
        super().__init__(subgrid)
        self.precursors = precursors
        self.G = G
        self.iteration = 0

    def store_outputs(self):
        return super().store_outputs(self.iteration)

    def update_electric_sources(self):
        super().update_electric_sources(self.iteration)
        self.iteration += 1

    def update_magnetic_sources(self):
        return super().update_magnetic_sources(self.iteration)

    def hsg_1(self):
        """First half of the subgrid update. Takes the time step up to the main
        grid magnetic update.
        """

        G = self.G
        subgrid = self.grid
        precursors = self.precursors

        # Copy the main grid electric fields at the IS position
        precursors.update_electric()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(int(m + subgrid.ratio / 2 - 0.5))
            subgrid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(m)
            subgrid.update_magnetic_is(precursors)
            self.update_magnetic_sources()

        self.store_outputs()
        self.update_electric_a()
        self.update_electric_pml()
        precursors.calc_exact_magnetic_in_time()
        subgrid.update_electric_is(precursors)
        self.update_electric_sources()
        self.update_electric_b()
        subgrid.update_electric_os(G)

    def hsg_2(self):
        """Second half of the subgrid update. Takes the time step up to the main
        grid electric update.
        """

        G = self.G
        subgrid = self.grid
        precursors = self.precursors

        # Copy the main grid magnetic fields at the IS position
        precursors.update_magnetic()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(int(m + subgrid.ratio / 2 - 0.5))
            subgrid.update_magnetic_is(precursors)
            self.update_magnetic_sources()
            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(m)
            subgrid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()

        self.update_magnetic()
        self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        subgrid.update_magnetic_is(precursors)
        self.update_magnetic_sources()
        subgrid.update_magnetic_os(G)
