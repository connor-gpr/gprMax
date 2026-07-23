# Phase 0 gate results — CUDA sub-gridding groundwork (2026-07-23)

Machine: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB, WDDM), driver CUDA 13.2,
conda-installed CUDA toolkit 12.9 (`cuda-nvcc`, `cuda-cudart-dev` in the
`gprMax-devel` env), PyCUDA 2025.1.2, MSVC 14.4x via VS2022 BuildTools
`vcvars64.bat`. All commands below need the vcvars + `Library\bin` PATH
wrapper for nvcc to be found.

## 1. Context-refactor regression (bit-identical)

`CUDAUpdates` now accepts an optional shared pycuda context (`ctx=` kwarg,
ownership flag, `cleanup()` pops only when owning); `CUDAGrid`'s device-array
management was extracted into `CUDAArrayMixin`. Regression: a 60³-cell,
400-iteration model (Hertzian dipole, lossy dielectric + PEC blocks, 10-cell
PML) run on the CUDA solver in single precision before and after the
refactor is **bit-identical** (max abs diff exactly 0.0 on all rx
components).

## 2. fp64 CUDA enablement gate — PASS

The CUDA fp64 path (never before compiled or executed in gprMax) was
exercised via the new `precision="double"` API override:

- **Literal audit:** no hard-coded `f`-suffixed literals or `float`-typed
  temporaries in any `gprMax/cuda_opencl` kernel template or body; all real
  types flow through `$REAL`/Jinja substitution. `_copy_mat_coeffs`'s
  constant-memory sizing check uses `.nbytes` of the actual arrays, so it is
  correct when coefficient bytes double.
- **CUDA-fp64 vs CPU-fp64, PML disabled:** worst rx-trace NRMSE **4.9e-13**
  (max abs 9.8e-12) over 400 iterations — rounding-order noise only. The
  volume/source/receiver kernel chain is fp64-clean.
- **CUDA-fp64 vs CPU-fp64, with PML:** worst NRMSE 5.6e-9. The delta is
  entirely attributable to the PML kernels' different arithmetic ordering
  (Cython vs CUDA): it is unchanged by `-fmad=false` (FMA is not the cause)
  and vanishes with PML off. For calibration, the *shipping* configuration
  (CPU-single vs CUDA-single, same model) shows NRMSE 3.3e-4 — the fp64
  path is ~5 orders of magnitude tighter than the accepted fp32 behaviour.
- Gate disposition: fp64 CUDA is validated for subgrid work. Full-model
  CPU-vs-GPU comparisons that include a main-grid PML should use the
  PML-attributed tolerance (NRMSE ≲ 1e-8 at 400 iterations, scaling with
  run length) rather than the PML-free 1e-12 class.

## 3. Launch-overhead spike and graph-dependency decision

Measured with a tiny fp64 kernel (4096 threads), 10k launches:

- Host-side async launch cost: **10.7 µs** (WDDM; high end of the plan's
  3–10 µs range).
- Simulated unfused subgrid iteration (~120 launches + sync): **1.28 ms**.
- Simulated fused iteration (~50 launches + sync): **0.53 ms**.

**Decision: fusion-only. `cuda-python` is NOT added as a dependency.**
PyCUDA has no CUDA Graph API; the fused launch count meets the Phase 3
target budget (< 0.5 ms/iteration is within reach of further fusion of the
per-face ring kernels, and datacenter/TCC deployments will sit well below
the WDDM numbers measured here). Revisit only if Phase 3 benchmarks miss
their targets on representative hardware.

## 4. Feature gates added (previously silent wrong results)

- Transmission lines under any non-CPU solver: build-time `ValueError`
  (they are stepped only by `CPUUpdates`).
- Discrete plane waves under any non-CPU solver: build-time `ValueError`.
- Discrete plane waves inside sub-grids: build-time `ValueError` on every
  solver (they were never time-stepped there, silently, even on CPU).
- `subgrid_gpu` flag (API-only, default off): CUDA+subgrid proceeds (double
  precision, SHSG only — enforced in Phase 1); OpenCL/Metal+subgrid raise
  with per-backend messages; without the flag the historical error message
  is unchanged.
- `precision` override (API-only, default None): validation/testing knob;
  warns when loosening the subgrid double-precision rule.

Tests: `tests/test_gpu_gates.py` (no GPU required - device detection
monkeypatched).
