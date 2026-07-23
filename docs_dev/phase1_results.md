# Phase 1 results — device-resident SHSG sub-grid core (2026-07-23)

Machine: RTX 3050 Ti Laptop (4 GB, WDDM), CUDA 12.9 toolkit, PyCUDA
2025.1.2, double precision throughout (the sub-grid rule).

## What runs where

- Sub-grid volume E/H updates, sources, receivers: inherited CUDAUpdates
  machinery on the sub-grid's own module set (dims + loss-painted
  coefficients baked; no PML — SHSG has none).
- OS-ring injection (`update_is_e/h`): compiled per sub-grid module;
  fused temporal blending (c1/c2 scalars, `_0`/`_1` device buffers,
  pointer-swap rollover).
- IS-ring main-grid corrections (`update_electric_os`/`update_magnetic_os`)
  and the `pack_planes` gather: compiled once into the main-grid module
  set; sub-grid dims passed as runtime args, so one compilation serves
  all sub-grids.
- Precursor spatial interpolation (FIR, transverse weighting, SciPy
  spline): CPU, byte-identical to the CPU solver — the pack kernel
  gathers the coarse surface planes, one packed D2H per snapshot, and
  the planes are scattered into the host main-grid arrays at their
  exact slice positions so the unmodified PrecursorNodes[Filtered] code
  runs unchanged. Interpolated fine `_1` sets return in one packed
  pinned H2D per snapshot. Steady-state PCIe: 2 D2H + 2 H2D packed
  transfers per main iteration, independent of substep count.

## Kernel unit tests (fp64 oracle: the Cython kernels)

73 tests in `tests/subgrids/test_cuda_coupling_knls.py`: update_is_e/h,
update_electric_os, update_magnetic_os over r ∈ {3, 5} × faces {1,2,3} ×
offsets/mid × HSG-style (s=3) and SHSG-style (s=0) argument sets on
non-cubic randomised domains — all match Cython to rtol 1e-14; the
pack_planes gather is bit-exact. All pass.

## Full-model CPU-vs-CUDA cross-validation

Gate: worst-component rx-trace NRMSE < 1e-9 (plan §6.1 short-run).

Initial single-subgrid model (48³ coarse, ratio 3, filtered, source +
PEC + rx in sub-grid, rx in main grid, 300 main iterations):
- sub-grid rx worst NRMSE 5.2e-11, main-grid rx worst 1.5e-11 — PASS.

Matrix (250 main iterations each): see `shsg_matrix` results below.

| case | config | worst NRMSE | result |
|---|---|---|---|
| filt_r3 | ratio 3, filtered, src+PEC+rx in sg | 5.3e-11 | PASS |
| nofilt_r3 | ratio 3, filter off | 5.9e-11 | PASS |
| filt_r5 | ratio 5, filtered | 3.5e-10 | PASS |
| interp2_r3 | ratio 3, interpolation=2 | 6.3e-11 | PASS |
| mainsrc | source in main grid, passive sub-grid | 2.8e-10 | PASS |
| twosg | two sub-grids coupled through the main grid | 2.9e-10 | PASS |

(12 rx datasets compared per case — all six components, sub-grid and
main-grid receivers, 250 main iterations.)

## Notes / deferred

- The CPU-vs-GPU comparison transitively checks the switched
  propositions (the CPU reference's IS-interior ≈ 0 and
  PEC-in-non-working invariance hold on GPU to the same NRMSE).
- Compile cost per model ≈ 5–7 s cold, mitigated by PyCUDA's source-hash
  disk cache on repeat runs; an in-process module cache for B-scans is
  measured/decided in Phase 3.
- Device-memory accounting for sub-grids (extend `calculate_memory_used`)
  deferred to Phase 3 benchmarks.
- Multi-sub-grid IS-ring writes are correct today because all launches
  serialize on the default stream; per-sub-grid streams (Phase 3) must
  keep the IS-ring launches serialized (see plan §4 Phase 3 item 3).
- Pre-existing observation to verify upstream (not sub-grid-specific):
  `htod_src_arrays` sizes waveform rows `iterations+1` but the baked
  `NY_SRCWAVES` stride is `iterations` — suspect off-by-row indexing for
  the 2nd+ source of a type in any plain-GPU model. To be confirmed
  with a two-source test before flagging upstream.
