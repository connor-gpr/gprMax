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

"""Device-resident SHSG sub-grid for the CUDA solver.

The grid object is the same SubGridSHSG (geometry, loss painting, CPU
call tables) plus the CUDAArrayMixin device-array management that
CUDAGrid uses. The SHSG has no sub-grid PML, so no CUDAPML machinery is
needed; the CPU update_*_is/os call tables are superseded by kernel
launches in CUDASubgridUpdater (subgrids/cuda_updates.py), which mirror
them argument for argument.
"""

from gprMax.grid.cuda_grid import CUDAArrayMixin

from .subgrid_shsg import SubGridSHSG


class CUDASubGridSHSG(CUDAArrayMixin, SubGridSHSG):
    """SHSG sub-grid solved on a CUDA device."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_cuda_arrays()
