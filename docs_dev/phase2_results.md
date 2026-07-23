# Phase 2 results — device-resident precursors (2026-07-23)

The precursor pipeline now runs entirely on device; **steady-state PCIe
traffic is zero** (matching the plain main-grid GPU solver). The Phase 1
bridge (pack → D2H → CPU SciPy → H2D) is no longer used by the solver but
stays in-tree (`CUDAPrecursorNodes[Filtered]Bridge`) as a debugging
fallback and reference.

## Mechanism

- `gather_weighted_planes` (one launch per snapshot per field type):
  gathers up to 4 coarse main-grid planes per face-component and combines
  them with per-plane weights computed once at init - the 3-tap FIR
  `[0.25, 0.5, 0.25]` and the transverse H bracketing-pair weights
  collapse into a single weight vector per face-component (filtered H:
  `[0.25c1, 0.5c1+0.25c2, 0.25c1+0.5c2, 0.25c2]`).
- `interp_faces` (one launch): `fine = Wx · coarse · Wyᵀ` per
  face-component with separable weight matrices probed through the exact
  SciPy `RectBivariateSpline` code path at init (unit vectors + partition
  of unity), exact for any `interpolation` degree; an init-time assertion
  verifies `Wx·F·Wyᵀ` reproduces SciPy on random data to 1e-11·scale.
  Output writes directly into the packed fine `_1` buffer used by the
  `update_is` kernels; rollover stays a device pointer swap.
- Per main iteration the precursor cost is now 4 kernel launches total
  (2 per field type), replacing 4 packed PCIe transfers + host SciPy.

## Validation

- `tests/subgrids/test_cuda_precursors.py`: device pipeline vs CPU
  `PrecursorNodes[Filtered]` fine `_1` arrays on randomised main-grid
  fields — interpolation {1,2,3} × filter on/off, atol 1e-12·scale:
  **6/6 pass**.
- Full-model CPU-vs-CUDA matrix (same 6 cases and CPU references as
  Phase 1):

| case | worst NRMSE | result |
|---|---|---|
| filt_r3 | 5.6e-11 | PASS |
| nofilt_r3 | 5.7e-11 | PASS |
| filt_r5 | 3.5e-10 | PASS |
| interp2_r3 | 5.9e-11 | PASS |
| mainsrc | 2.8e-10 | PASS |
| twosg | 2.5e-10 | PASS |

Same NRMSE class as the Phase 1 bridge — the weight-matrix path is
numerically indistinguishable from host SciPy at the trace level, and
the 1e-9 gate is met with two orders of margin.
