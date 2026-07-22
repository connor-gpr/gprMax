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

"""Tests for the Switched Huygens Subgridding (SHSG)."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.subgrids.subgrid_shsg import SubGridSHSG as SubGridSHSGGrid
from gprMax.subgrids.updates import OSSurfaceView
from gprMax.subgrids.user_objects import SubGridSHSG


def shsg_kwargs(**over):
    kw = dict(p1=[0.1, 0.1, 0.1], p2=[0.2, 0.2, 0.2], ratio=3, id="sg")
    kw.update(over)
    return kw


class TestUserObject:
    def test_defaults_no_pml_single_cell_halo(self):
        so = SubGridSHSG(**shsg_kwargs())
        assert so.kwargs["subgrid_pml_thickness"] == 0
        assert so.kwargs["pml_separation"] == 1
        assert so.kwargs["is_os_sep"] == 3
        assert so.kwargs["filter"] is True

    def test_default_loss_factors_are_maximal(self):
        # Paper (Figs 14/16/18): stability monotonic in l, l=1 best and
        # accuracy-neutral
        so = SubGridSHSG(**shsg_kwargs())
        for k in ("le", "lm", "les", "lms"):
            assert so.kwargs[k] == 1.0

    def test_rejects_all_zero_loss(self):
        # SHSG without loss is LESS stable than the HSG - not a valid config
        with pytest.raises(ValueError):
            SubGridSHSG(**shsg_kwargs(le=0, lm=0, les=0, lms=0))


class TestGridGeometry:
    def make_grid(self, ratio=3, is_os_sep=3):
        so = SubGridSHSG(**shsg_kwargs(ratio=ratio, is_os_sep=is_os_sep))
        return SubGridSHSGGrid(**so.kwargs)

    def test_boundary_cells_single_cell_beyond_os(self):
        g = self.make_grid()
        assert g.n_boundary_cells == 3 * 3 + 1
        assert g.os_f == 1

    def test_boundary_cells_scale_with_sep_and_ratio(self):
        g = self.make_grid(ratio=5, is_os_sep=2)
        assert g.n_boundary_cells == 2 * 5 + 1
        assert g.os_f == 1

    def test_no_subgrid_pml(self):
        g = self.make_grid()
        assert all(t == 0 for t in g.pmls["thickness"].values())


class TestMasks:
    # closed box [1, 4] in a 6-node axis; unstaggered nodes inside: 1..4,
    # staggered (pos idx+0.5) inside: 1..3
    def test_inside_mask_unstaggered(self):
        m = SubGridSHSGGrid._inside_mask((6, 6, 6), (1, 1, 1), (4, 4, 4), ())
        assert m[1, 1, 1] and m[4, 4, 4]
        assert not m[0, 2, 2] and not m[5, 2, 2] and not m[2, 0, 2]

    def test_inside_mask_staggered_axis(self):
        m = SubGridSHSGGrid._inside_mask((6, 6, 6), (1, 1, 1), (4, 4, 4), (0,))
        # x staggered: positions 1.5..3.5 inside -> i in 1..3, i=4 outside
        assert m[1, 1, 1] and m[3, 4, 4]
        assert not m[4, 2, 2] and not m[0, 2, 2]

    def test_outside_is_complement(self):
        args = ((5, 5, 5), (1, 1, 1), (3, 3, 3), (1,))
        inside = SubGridSHSGGrid._inside_mask(*args)
        outside = SubGridSHSGGrid._outside_mask(*args)
        assert (inside ^ outside).all()

    def test_paint_covers_only_the_halo_shell(self):
        # With the SHSG halo the non-working region is exactly the
        # outermost node layer(s) beyond the OS at fine index 1
        shape = (12, 12, 12)
        lo, hi = (1, 1, 1), (11, 11, 11)
        # unstaggered component (e.g. Ez along x-y): outside = index 0 only
        # per axis (position 0 < 1) since 11 is on the far plane
        m = SubGridSHSGGrid._outside_mask(shape, lo, hi, ())
        assert m[0].all() and m[:, 0].all() and m[:, :, 0].all()
        assert not m[1:12, 1:12, 1:12].any()


class TestLossMaterial:
    def make_stub_grid(self, dt=1e-12, n=4):
        return SimpleNamespace(
            dt=dt, dx=0.001, dy=0.001, dz=0.001,
            updatecoeffsE=np.zeros((2, 5)), updatecoeffsH=np.zeros((2, 5)),
            materials=[],
        )

    @staticmethod
    def stub_config(monkeypatch):
        import gprMax.config as config

        monkeypatch.setattr(
            config, "get_model_config",
            lambda: SimpleNamespace(materials={"maxpoles": 0}),
        )
        monkeypatch.setattr(
            config, "sim_config",
            SimpleNamespace(em_consts={"e0": config.e0, "m0": config.m0}),
        )

    def test_loss_factor_one_gives_zero_field_persistence(self, monkeypatch):
        self.stub_config(monkeypatch)
        g = self.make_stub_grid()
        numID = SubGridSHSGGrid._add_loss_material(g, 1.0, 0.0, "test_le")
        # l = sigma*dt/(2*eps) = 1 -> CA = (1-l)/(1+l) = 0
        assert numID == 2
        assert g.updatecoeffsE.shape == (3, 5)
        assert g.updatecoeffsE[2, 0] == pytest.approx(0.0, abs=1e-12)
        # magnetic side untouched by an le-only material
        assert g.updatecoeffsH[2, 0] == pytest.approx(1.0)

    def test_loss_factor_value(self, monkeypatch):
        self.stub_config(monkeypatch)
        g = self.make_stub_grid()
        SubGridSHSGGrid._add_loss_material(g, 0.25, 0.0, "test_le")
        # CA = (1-0.25)/(1+0.25) = 0.6
        assert g.updatecoeffsE[2, 0] == pytest.approx(0.6)


class TestOSSurfaceView:
    def test_extents_grow_by_two_seps(self):
        sg = SimpleNamespace(
            i0=10, j0=12, k0=14, i1=20, j1=22, k1=24,
            nwx=30, nwy=30, nwz=30, ratio=3, is_os_sep=3, interpolation=1,
        )
        v = OSSurfaceView(sg)
        assert (v.i0, v.j0, v.k0) == (7, 9, 11)
        assert (v.i1, v.j1, v.k1) == (23, 25, 27)
        assert v.nwx == v.nwy == v.nwz == 30 + 2 * 9
        assert v.ratio == 3 and v.interpolation == 1
