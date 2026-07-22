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

"""1-D numerical reference for the Switched Huygens Subgridding.

A self-contained NumPy implementation of the SHSG coupling using the same
sign/stencil/sequencing conventions as gprMax's 3-D implementation
(gprMax/subgrids/subgrid_shsg.py). It verifies, quickly and without any
gprMax machinery, that those conventions produce:

- transparency of an empty subgrid to a traversing pulse,
- the switched propositions (zero field in both non-working regions,
  physical field in the IS-OS gap),
- insensitivity of the working-region solution to the artificial loss.

The conventions were originally frozen with this model (Phase 0 of the
SHSG implementation); if a refactor of the 3-D code disagrees with it,
the 3-D code is wrong.
"""

import numpy as np

CFL = 0.9
RATIO = 3
SEP = 3


def gaussian(t, t0=60.0, tau=15.0):
    return np.exp(-(((t - t0) / tau) ** 2))


class Yee1D:
    def __init__(self, n):
        self.n = n
        self.Ez = np.zeros(n + 1)
        self.Hy = np.zeros(n + 1)  # Hy[i] at i+1/2
        self.le = np.zeros(n + 1)
        self.lm = np.zeros(n + 1)
        self.c = CFL
        self.pec = np.zeros(n + 1, dtype=bool)

    def update_E(self):
        le = self.le[1:self.n]
        self.Ez[1:self.n] = ((1 - le) / (1 + le)) * self.Ez[1:self.n] + (
            self.c / (1 + le)) * (self.Hy[1:self.n] - self.Hy[0:self.n - 1])
        self.Ez[self.pec] = 0.0

    def update_H(self):
        lm = self.lm[0:self.n]
        self.Hy[0:self.n] = ((1 - lm) / (1 + lm)) * self.Hy[0:self.n] + (
            self.c / (1 + lm)) * (self.Ez[1:self.n + 1] - self.Ez[0:self.n])


class SHSG1D:
    """Main grid A + switched subgrid B (ratio RATIO, IS-OS sep SEP)."""

    def __init__(self, NA, w1, w2, loss=1.0, pec_a=(), pec_b=()):
        r = RATIO
        self.NA, self.w1, self.w2 = NA, w1, w2
        self.os1, self.os2 = w1 - SEP, w2 + SEP
        self.A = Yee1D(NA)
        self.b0_f = self.os1 * r - 1  # single fine cell beyond the OS
        self.nB = (self.os2 - self.os1) * r + 2
        self.B = Yee1D(self.nB)
        self.fB = lambda x: int(round(x * r - self.b0_f))

        # loss: A inside the closed IS box; B beyond the OS
        self.A.le[w1:w2 + 1] = loss
        self.A.lm[w1:w2] = loss
        self.B.le[0] = loss
        self.B.lm[0:1] = loss
        self.B.lm[self.fB(self.os2):] = loss
        self.B.pec[0] = True
        self.B.pec[self.nB] = True
        for p in pec_a:
            self.A.pec[int(round(p))] = True
        for p in pec_b:
            self.B.pec[self.fB(p)] = True

        self.pE0 = np.zeros(2)
        self.pE1 = np.zeros(2)
        self.pH0 = np.zeros(2)
        self.pH1 = np.zeros(2)

    def snap_E(self):
        self.pE0[:] = self.pE1
        self.pE1[:] = (self.A.Ez[self.os1], self.A.Ez[self.os2])

    def snap_H(self):
        r = RATIO
        self.pH0[:] = self.pH1
        wl, wr = (r - r // 2) / r, (r // 2) / r
        hl = wl * self.A.Hy[self.os1 - 1] + wr * self.A.Hy[self.os1]
        hr = wr * self.A.Hy[self.os2 - 1] + wl * self.A.Hy[self.os2]
        self.pH1[:] = (hl, hr)

    def interpE(self, k):
        f = k / RATIO
        return (1 - f) * self.pE0 + f * self.pE1

    def interpH(self, k):
        f = k / RATIO
        return (1 - f) * self.pH0 + f * self.pH1

    # --- the four correction families (signs frozen in Phase 0) -------
    def os_E(self, Hinc):
        il, ir = self.fB(self.os1), self.fB(self.os2)
        self.B.Ez[il] += -self.B.c * Hinc[0]
        self.B.Ez[ir] += +self.B.c * Hinc[1]

    def os_H(self, Einc):
        il, ir = self.fB(self.os1) - 1, self.fB(self.os2)
        self.B.Hy[il] += -(self.B.c / (1 + self.B.lm[il])) * Einc[0]
        self.B.Hy[ir] += +(self.B.c / (1 + self.B.lm[ir])) * Einc[1]

    def is_E(self):
        c = self.A.c
        hl = self.B.Hy[self.fB(self.w1) - (RATIO // 2) - 1]
        hr = self.B.Hy[self.fB(self.w2) + (RATIO // 2)]
        self.A.Ez[self.w1] += +(c / (1 + self.A.le[self.w1])) * hl
        self.A.Ez[self.w2] += -(c / (1 + self.A.le[self.w2])) * hr

    def is_H(self):
        c = self.A.c
        self.A.Hy[self.w1 - 1] += +c * self.B.Ez[self.fB(self.w1)]
        self.A.Hy[self.w2] += -c * self.B.Ez[self.fB(self.w2)]

    def step(self, n, src_i, src_fn):
        r = RATIO
        upper_m = r // 2

        self.A.update_E()
        self.A.Ez[src_i] += src_fn(n)

        # hsg_1
        self.snap_E()
        for m in range(1, upper_m + 1):
            self.B.update_E()
            self.os_E(self.interpH(m + r // 2))
            self.B.update_H()
            self.os_H(self.interpE(m))
        self.B.update_E()
        self.os_E(self.pH1)
        self.is_E()

        self.A.update_H()

        # hsg_2
        self.snap_H()
        for m in range(1, upper_m + 1):
            self.B.update_H()
            self.os_H(self.interpE(m + r // 2))
            self.B.update_E()
            self.os_E(self.interpH(m))
        self.B.update_H()
        self.os_H(self.pE1)
        self.is_H()


def run_coarse_reference(nsteps, NA, src_i, src_fn, probes, pec=()):
    g = Yee1D(NA)
    for p in pec:
        g.pec[int(round(p))] = True
    out = {p: np.zeros(nsteps) for p in probes}
    for n in range(nsteps):
        g.update_E()
        g.Ez[src_i] += src_fn(n)
        g.update_H()
        for p in probes:
            out[p][n] = g.Ez[p]
    return out


NA, W1, W2, SRC = 400, 180, 220, 60
NSTEPS = 500
PROBES = [140, 300]


def run_shsg(loss=1.0, pec_a=(), pec_b=()):
    m = SHSG1D(NA, W1, W2, loss=loss, pec_a=pec_a, pec_b=pec_b)
    out = {p: np.zeros(NSTEPS) for p in PROBES}
    lvl = dict(nonworking=0.0, beyond_os=0.0, gap=0.0)
    for n in range(NSTEPS):
        m.step(n, SRC, gaussian)
        for p in PROBES:
            out[p][n] = m.A.Ez[p]
        lvl["nonworking"] = max(lvl["nonworking"],
                                np.abs(m.A.Ez[W1 + 2:W2 - 1]).max())
        lvl["beyond_os"] = max(lvl["beyond_os"], abs(m.B.Hy[0]),
                               abs(m.B.Hy[m.nB - 1]))
        lvl["gap"] = max(lvl["gap"],
                         abs(m.A.Ez[W1 - 1] - m.B.Ez[m.fB(W1 - 1)]))
    return out, lvl


def nrmse(x, ref):
    rng = ref.max() - ref.min()
    return np.sqrt(np.mean((x - ref) ** 2)) / rng * 100


def test_transparency():
    ref = run_coarse_reference(NSTEPS, NA, SRC, gaussian, PROBES)
    out, _ = run_shsg()
    for p in PROBES:
        assert nrmse(out[p], ref[p]) < 1.0


def test_switched_propositions():
    _, lvl = run_shsg()
    assert lvl["nonworking"] < 1e-3   # main grid inside the IS ~ zero
    assert lvl["beyond_os"] < 1e-2    # subgrid beyond the OS ~ zero
    assert lvl["gap"] < 1e-2          # gap field physical in both grids


def test_pec_in_nonworking_region_has_no_effect():
    out0, _ = run_shsg()
    out1, _ = run_shsg(pec_a=[(W1 + W2) // 2])
    for p in PROBES:
        assert np.abs(out0[p] - out1[p]).max() < 1e-12


def test_loss_does_not_corrupt_solution():
    out_ref, _ = run_shsg(loss=1e-6)
    out_l1, _ = run_shsg(loss=1.0)
    for p in PROBES:
        assert np.abs(out_l1[p] - out_ref[p]).max() < 5e-3
