from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gprMax.fields_outputs import output_position
from gprMax.geometry_outputs.geometry_views import Metadata
from gprMax.geometry_outputs.grid_view import GridView
from gprMax.subgrids.subgrid_hsg import SubGridHSG
from gprMax.subgrids.user_objects import SubGridHSG as SubGridHSGUserObject
from gprMax.user_inputs import SubgridUserInput


def make_subgrid(
    ratio=5,
    is_corner_cell=(40, 220, 4),
    working_region_main_cells=(80, 64, 32),
    dl_main=0.0025,
):
    """Helper to create a subgrid without building a full model.

    Args:
        ratio: subgrid ratio.
        is_corner_cell: main grid cell of the Inner Surface lower corner.
        working_region_main_cells: size of the working region in main
            grid cells.
        dl_main: spatial discretisation of the main grid.

    Returns:
        sg: SubGridHSG grid object.
    """
    sg = SubGridHSG(
        ratio=ratio,
        id="test_subgrid",
        filter=True,
        is_os_sep=3,
        pml_separation=ratio // 2 + 2,
        subgrid_pml_thickness=6,
        interpolation=1,
    )

    sg.i0, sg.j0, sg.k0 = is_corner_cell
    sg.dx = sg.dy = sg.dz = dl_main / ratio
    sg.dl = np.array([sg.dx, sg.dy, sg.dz])

    sg.nwx, sg.nwy, sg.nwz = np.array(working_region_main_cells) * ratio
    sg.nx = 2 * sg.n_boundary_cells_x + sg.nwx
    sg.ny = 2 * sg.n_boundary_cells_y + sg.nwy
    sg.nz = 2 * sg.n_boundary_cells_z + sg.nwz

    sg.x1, sg.y1, sg.z1 = np.array(is_corner_cell) * dl_main
    sg.x2, sg.y2, sg.z2 = (np.array(is_corner_cell) + np.array(working_region_main_cells)) * dl_main

    return sg


class TestLocalToGlobalCoordinate:
    def test_inverse_of_translate_to_gap(self):
        sg = make_subgrid()
        uip = SubgridUserInput(sg)

        global_point = np.array([230, 1150, 40], dtype=np.int32)
        local_point = uip.translate_to_gap(global_point)

        assert_array_equal(sg.local_to_global_coordinate(local_point), global_point)

    def test_is_corner_maps_to_main_grid_placement(self):
        sg = make_subgrid()
        n_boundary_cells = np.array(
            [sg.n_boundary_cells_x, sg.n_boundary_cells_y, sg.n_boundary_cells_z]
        )

        is_corner = sg.local_to_global_coordinate(n_boundary_cells)

        assert_array_equal(is_corner, np.array([sg.i0, sg.j0, sg.k0]) * sg.ratio)
        # Physical position of the IS corner should be the subgrid
        # placement in the main grid
        assert np.allclose(is_corner * sg.dl, (sg.x1, sg.y1, sg.z1))


class TestSubgridSrcRxReporting:
    def test_geometry_view_positions_are_global(self):
        sg = make_subgrid()
        grid_view = GridView(sg, 0, 0, 0, sg.nx, sg.ny, sg.nz)
        metadata = Metadata(grid_view, materials_only=True)

        # Source at the IS corner of the subgrid
        coord = np.array([sg.n_boundary_cells_x, sg.n_boundary_cells_y, sg.n_boundary_cells_z])
        src = SimpleNamespace(coord=coord, ID="src")

        names, positions = metadata.srcs_rx_gv_comment([src])

        assert names == ["src"]
        assert np.allclose(positions[0], (sg.x1, sg.y1, sg.z1))

    def test_output_file_position_is_global(self):
        sg = make_subgrid()
        rx = SimpleNamespace(
            xcoord=sg.n_boundary_cells_x,
            ycoord=sg.n_boundary_cells_y,
            zcoord=sg.n_boundary_cells_z,
        )

        position = output_position(rx, sg, is_subgrid=True)

        assert np.allclose(position, (sg.x1, sg.y1, sg.z1))

    def test_output_file_position_main_grid_unchanged(self):
        grid = SimpleNamespace(dx=0.001, dy=0.002, dz=0.003)
        rx = SimpleNamespace(xcoord=10, ycoord=20, zcoord=30)

        position = output_position(rx, grid, is_subgrid=False)

        assert np.allclose(position, (0.01, 0.04, 0.09))


class TestSubgridCheckThickness:
    def test_valid_thickness_in_offset_subgrid(self):
        """A valid extent must not fail because the helper points used by
        check_thickness have placeholder zeros in the unchecked axes
        (which the subgrid translation maps outside the grid)."""
        sg = make_subgrid()
        uip = SubgridUserInput(sg)

        # z extent within the subgrid working region (global coordinates)
        lower_extent = sg.z1 + 0.005
        thickness = 0.002

        within_grid, lower, thick = uip.check_thickness("z", lower_extent, thickness, "#triangle")

        assert within_grid
        # The returned lower extent is in the local coordinates of the
        # subgrid, matching the translated coordinates used to build
        # geometry objects
        assert lower == pytest.approx(uip.round_to_grid((0, 0, lower_extent))[2])
        assert thick == pytest.approx(thickness)

    def test_out_of_bounds_thickness_raises(self):
        sg = make_subgrid()
        uip = SubgridUserInput(sg)

        # z extent beyond the top of the subgrid array
        lower_extent = sg.z2 + 1.0

        with pytest.raises(ValueError, match="z dimension"):
            uip.check_thickness("z", lower_extent, 0.002, "#triangle")


class TestSubgridTimewindow:
    def test_set_timewindow(self):
        sg = make_subgrid()
        sg.dt = 1e-12
        sg.iterations = 1000

        user_object = SubGridHSGUserObject(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), id="sg")
        user_object.set_timewindow(sg)

        assert sg.timewindow == pytest.approx((sg.iterations - 1) * sg.dt)
