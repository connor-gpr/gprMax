# Phase 3 results — fusion, stability, benchmarks (2026-07-23)

Machine: RTX 3050 Ti Laptop (4 GB, WDDM, fp64 at 1/64 rate), CUDA 12.9,
PyCUDA 2025.1.2. All solver runs double precision unless stated.

## Kernel fusion — and the ring-edge race it exposed

The six per-face ring launches per update fuse into **two** spec-driven
launches (`update_is_faces_e/h`, `update_os_faces_electric/magnetic`),
not one: the per-face calls are *not disjoint* — ring-box **edge nodes
legitimately accumulate `+=` contributions from two faces writing the
same field component**. Sequential launches order those safely; a single
fused launch makes them a concurrent read-modify-write race (first
attempt: worst rx NRMSE ~1 across the whole matrix). The fix splits the
six calls into two conflict-free groups (greedy 2-colouring — each
component appears exactly twice, so it always succeeds), preserving
determinism with no atomics. Post-fix the full 6-case matrix reproduces
the unfused results to 4 significant figures (worst NRMSE 3.5e-10,
gate 1e-9).

This is the same hazard class the plan flagged for multi-sub-grid
IS-ring writes under concurrent streams — measured here intra-sub-grid.
**Multi-sub-grid launches stay serialized on the default stream** (the
twosg matrix case passes); per-sub-grid streams remain a future opt-in
that must keep all IS-ring launches on one stream.

Launch counts per main iteration (r=3, one sub-grid): ring corrections
36 → 12; total subgrid path ~67 → ~31.

## Benchmarks (300 main iterations, source+rx in sub-grid, solve time)

| model | CPU (all cores) | GPU | speedup |
|---|---|---|---|
| nw=12, r=3 | 7.37 s | 0.87 s | **8.5×** |
| nw=20, r=3 | 19.1 s | 1.58 s | **12.1×** |
| nw=12, r=5 | 35.8 s | 2.22 s | **16.1×** |

On a consumer laptop GPU with 1/64-rate fp64 this already clears the
plan's ≥5× consumer target, and the trend with r matches the `nw³·r⁴`
subgrid-work scaling — datacenter-class fp64 should comfortably clear
the ≥10× target. B-scan amortization: traces 2..3 cost ~2.7 s each
total (0.9–1.0 s solve + ~1.7 s rebuild/render with the PyCUDA disk
compile cache hitting) — above the plan's <1 s stretch; an in-process
module cache keyed on (dims, precision, maxpoles, PML) is the known
next step if B-scan-heavy workloads need it.

## Stability soaks (20 000 main iterations, full CFL)

Resonant PEC backplane one coarse cell from the IS inside the sub-grid
(the HSG-unstable configuration class):

| run | late-time envelope growth | verdict |
|---|---|---|
| CPU fp64 | −0.0013/ns | stable |
| GPU fp64 | −0.0013/ns (matches CPU) | stable |
| GPU fp32 | −0.0020/ns | stable |

## Single-precision experiment

An fp32 sub-grid run initially produced NaN at sub-grid iteration ~26 —
root cause: the host precursor arrays are always float64, and the device
buffer dtype was derived from them instead of the configured precision,
so fp32 kernels read float64 bits. After fixing the dtype source
(`config.sim_config.dtypes`), **fp32 is stable over the full soak** with
decaying envelope. Trace accuracy vs fp64 on this deliberately resonant
model: NRMSE 4.1e-3 over 300 iterations — above the 1e-3 aspiration, so
**fp32 stays opt-in** (explicit `precision="single"`), documented as
stable but reduced-accuracy; revisit on non-resonant workloads.

## De-flag decision

`subgrid_gpu` **stays experimental (off by default)**: local numbers are
from one consumer WDDM GPU; graduate after the platform (datacenter)
benchmarks and the measured_2 flagship acceptance run reproduce these
margins. Everything needed to run it is in place.
