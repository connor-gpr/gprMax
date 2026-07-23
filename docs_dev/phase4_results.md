# Phase 4 results — dispersive sub-grids, docs, remaining scope (2026-07-23)

## Dispersive materials with subgrid_gpu — supported

Removing the global `maxpoles` gate was sufficient: the inherited
CUDAUpdates machinery already compiles the templated dispersive A/B
kernels per grid, uploads `Tx/Ty/Tz` + `updatecoeffsdispersive` (which
`apply_shsg_loss` extends with zero rows), and dispatches
`update_electric_a/b` on `maxpoles` — all of which the sub-grid updater
inherits on its own module set. The CPU ordering quirk (main-grid
`update_electric_b` runs after `hsg_1`, so precursor E snapshots see
pre-B main E) is preserved by construction, matching the CPU solver.

Validation (single-pole Debye water block inside the sub-grid, 250 main
iterations, CPU vs CUDA fp64): dominant-component rx traces agree to
NRMSE ≤ 8e-11; absolute errors are uniform ~5e-10 across all
components (sub-dominant components' own-scale NRMSE up to 1.3e-9,
an artifact of normalising by a 100× smaller amplitude). Pass.

This also restores the dispersive-main-grid + non-dispersive-sub-grid
case that the Phase 1 global gate blocked.

## Docs

`docs/source/examples_advanced.rst` documents the experimental
`subgrid_gpu` flag: SHSG-only, double precision, no per-iteration
transfers; TL/plane-wave gates called out (previously silent drops);
`precision="single"` noted as experimental.

## Explicitly deferred (loud gates in place)

- **Transmission lines under GPU solvers**: build-time error (Phase 0)
  remains; the device TL port (per-TL 1-D recursion kernel) is scoped in
  the plan §4 Phase 4 item 2 and not implemented here. TL-fed models run
  on CPU; the flagship acceptance models are voltage-fed.
- **OpenCL parity**: subgrid+OpenCL raises with its own message. The
  kernel bodies are reusable but need raw `cl.Program` authoring, the
  `cl_khr_fp64` capability check, and the `OpenCLPML` per-slab
  `event.wait()` removal (plan §4 Phase 4 item 3).
- **Metal**: permanently gated (no fp64).
- **In-process module cache** for B-scan-heavy workloads (Phase 3 note).
- **`subgrid_gpu` graduation** to default-on: after platform
  (datacenter GPU) benchmarks and the measured_2 flagship acceptance
  run. The local consumer-GPU evidence: 8.5–16× vs all-cores CPU,
  20k-iteration stability parity, full matrix + dispersive gates green.
