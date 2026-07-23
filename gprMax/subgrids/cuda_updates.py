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

"""CUDA (device-resident) solver for SHSG sub-grids.

Phase 0 skeleton: the dispatch plumbing (config subgrid_gpu flag,
create_updates solver dispatch, HSGCapable solver gate) is in place, but
the device subgrid updater lands in Phase 1.
"""

import logging

from gprMax.model import Model

logger = logging.getLogger(__name__)


def create_updates(model: Model):
    """Return the CUDA updates object for the given subgrids."""
    logger.exception(
        "The CUDA sub-grid solver is not implemented yet (subgrid_gpu is an "
        "experimental preview flag). Use the CPU solver for sub-gridded models."
    )
    raise NotImplementedError
