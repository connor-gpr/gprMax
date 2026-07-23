# PR #297 salvage review — GSoC 2021 CUDA port of Huygens subgridding

Design note reviewing https://github.com/gprMax/gprMax/pull/297 ("[GSoC 2021]
Implementing GPU Accelerated Sub-gridding") as prior art for a fresh
device-resident CUDA HSG port (see `GPU_SUBGRID_PLAN.md`).

**Sources.** All material was retrievable (2026-07-23):

- PR metadata/description via GitHub API (`/repos/gprMax/gprMax/pulls/297`).
- Full file list + patches via `/pulls/297/files` (12 files, +1569/−125).
- Raw sources at the PR head SHA `0d204849ee791ae55c068408ae408eac9e057ae4`
  from `raw.githubusercontent.com/ThenoobMario/gprMax/<sha>/...`:
  `gprMax/cuda/hsg_field_updates.py` (409 lines, the CUDA kernels),
  `gprMax/subgrids/subgrid_hsg.py` (830 lines, launch code),
  `gprMax/subgrids/base.py`, `gprMax/subgrids/updates.py`, `gprMax/solvers.py`,
  `gprMax/grid.py`.
- The fork branch `ThenoobMario/gprMax:devel` still exists and its tip **is**
  the PR head (`0d204849`, last commit 2021-11-09) — the branch is frozen.
- Cython ground truth: local
  `C:\Users\conno\Desktop\workspaces\gprmax_devel\gprMax\cython\fields_updates_hsg.pyx`
  (289 lines) and `gprMax\subgrids\subgrid_hsg.py` (current devel).

Nothing relevant was unretrievable. Note that there is *no review discussion
to retrieve*: the API reports `comments = 0` and `review_comments = 0`.

---

## (a) Status and history

| Item | Value |
|---|---|
| Title | [GSoC 2021] Implementing GPU Accelerated Sub-gridding |
| Author | `ThenoobMario` (GSoC 2021 student, mentored project) |
| Opened | 2021-08-19 |
| Head | `ThenoobMario:devel` @ `0d204849` (30 commits) |
| Base | `gprMax:devel` (the pre-refactor devel of 2021) |
| Size | 12 files, +1569 / −125 |
| Closed | 2025-08-12, **unmerged** (`merged = false`, `merged_at = null`); the PR page attributes the close to maintainer craig-warren |
| Review | Zero review comments, zero issue comments — it was never formally reviewed |

The PR was effectively abandoned in late 2021 and sat open for ~4 years until
a maintainer housekeeping pass closed it. The author's own description flags
it as incomplete: "Further Work to be Done — GPU kernels for Precursors Nodes
calculations; Refactoring the CUDASubGridHSG class to reduce redunduncy."
I.e. **precursor interpolation stayed on the CPU**, which forces the
host/device transfer architecture criticized below. Key integration changes:
the `config.py` guard that forbade CUDA + subgrids was simply commented out,
and `cmds_multiuse.py` `isinstance(SubGridBase)` checks were widened to
`CPUSubGridBase`/`CUDASubGridBase` (both inherit a plain `SubGridBase` mixin;
`CUDASubGridBase(SubGridBase, CUDAGrid)` in `subgrids/base.py`).

Because the PR targets the 2021 codebase, it predates the current
`cuda_opencl/` kernel organisation (`knl_fields_updates.py` with
CUDA/OpenCL/Metal argument blocks) — none of its plumbing maps onto today's
devel; only the kernel bodies are of interest.

## (b) How PR #297's CUDA kernels index the dual-grid arrays

All kernels live in `gprMax/cuda/hsg_field_updates.py` as two
`string.Template` objects (`kernel_template_os`, `kernel_template_is`) with
**array dimensions baked in at compile time** via `Template.substitute`:

```c
// kernel_template_os macros
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS) + (j)*($NZ_FIELDS) + (k)
#define INDEX3D_SUBFIELDS(i, j, k) (i)*($NY_SUBFIELDS)*($NZ_SUBFIELDS) + (j)*($NZ_SUBFIELDS) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID) + (i)*($NY_ID)*($NZ_ID) + (j)*($NZ_ID) + (k)
```

This is the one genuinely load-bearing dual-grid idea in the PR: the OS
kernels address **two different grids in one kernel** — the main-grid `field`,
`ID` and `updatecoeffs` through `$N*_FIELDS`/`$N*_ID` strides, and the subgrid
incident field through separate `$N*_SUBFIELDS` strides. For OS updates the
substitution uses `main_grid.nx+1 ...` for FIELDS and `self.nx+1 ...`
(subgrid) for SUBFIELDS. The IS template has only one `INDEX3D_FIELDS` macro,
substituted with *subgrid* dims, because IS updates touch only subgrid arrays.

### Thread mapping: full-volume 1D launch masked down to a 2D face

Every kernel maps threads the same way (quoting `hsg_update_electric_os`):

```c
// Current Thread Index
int idx = blockIdx.x * blockDim.x + threadIdx.x;
...
// Linear Index to subscript
// Since we are only concerned for a 2D slice of the grid with no
// circular wrapping, the calculation done below works
l = idx / ($NZ_FIELDS * $NY_FIELDS);
m = idx % ($NZ_FIELDS * $NY_FIELDS);

// If block for front and back face calculation
if(face == 3 && l >= l_l && l < l_u && m >= m_l && m < m_u) { ... }
```

The launch configuration is the grid's *volume* launch reused verbatim
(`grid.py` in the fork):

```python
self.tpb = (128, 1, 1)
self.bpg = (int(np.ceil(((self.nx + 1) * (self.ny + 1) * (self.nz + 1)) / self.tpb[0])), 1, 1)
```

OS kernels launch with `main_grid.tpb / main_grid.bpg`; IS kernels with the
subgrid's `self.tpb / self.bpg`. So one thread is launched **per cell of the
entire 3D volume**, `l` decodes to the slab index (0..NX) and `m` to the
0..NY·NZ offset within the slab; the `if` guard then discards every thread
whose `m` is not in `[m_l, m_u)` — i.e. `m` is compared directly against a
*coordinate* bound of order tens. Each surviving `(l, m)` pair occurs exactly
once, so the mapping is *correct*, but occupancy is face-area/volume — for a
typical model well under 1% of launched threads do any work, and the `face`
branches are compiled into every thread. The PR description sells this as a
2D-slice decode ("Only two indices change during one computation"), but it is
really a full-volume launch with masking; the divisor `$NZ_FIELDS *
$NY_FIELDS` only works because it happens to equal the slab size.

### OS kernels: main-grid vs subgrid subscripts

The dual-grid arithmetic is a faithful transcription of the Cython. Constants
(computed per-thread, redundantly):

```c
// Surface normal index for the subgrid near face h nodes (left i index)
n_s_l = n_boundary_cells - (surface_sep * sub_ratio) - sub_ratio + floor((double) sub_ratio / 2);
// Surface normal index for the subgrid far face h nodes (right i index)
n_s_r = n_boundary_cells + nwn + (surface_sep * sub_ratio) + floor((double) sub_ratio / 2);
// OS at the left face
os = n_boundary_cells - (sub_ratio * surface_sep);
```

Per-face transverse mapping main→sub (`mid` selects which of the two
transverse axes gets the half-cell offset):

```c
if(mid == 1) {
    l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
    m_s = os + (m - m_l) * sub_ratio;
}
else {
    l_s = os + (l - l_l) * sub_ratio;
    m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
}
```

and the face branch assigns main-grid (`i0/j0/k0` near, `i2/j2/k2` far) and
subgrid (`i1/j1/k1`, `i3/j3/k3`) subscripts, e.g. face 2 (left/right):

```c
// Main grid Index
i0 = n_l; j0 = l; k0 = m;
// Sub-grid Index
i1 = n_s_l; j1 = l_s; k1 = m_s;
i2 = n_u; j2 = l; k2 = m;
i3 = n_s_r; j3 = l_s; k3 = m_s;
```

followed by the update, one thread doing *both* near and far faces:

```c
int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;

int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;
field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
```

`hsg_update_magnetic_os` is identical except `n_s_l = n_s_r-style` E-node
normals (`n_s_l = n_boundary_cells - sub_ratio * surface_sep;`,
`n_s_r = n_boundary_cells + nwn + sub_ratio * surface_sep;`) and
`updatecoeffsH`. No write races: within a launch each `(l,m)` is one thread,
near/far writes are distinct nodes, and the six per-face launches serialize on
the default stream.

### IS kernel: subgrid-only, 2D precursor arrays flattened with a runtime stride

```c
// For inner Faces H nodes are 1 cell before n_boundary_cell
n_o = n + offset;

l = idx / ($NZ_FIELDS * $NY_FIELDS);
m = idx % ($NZ_FIELDS * $NY_FIELDS);

if(l >= n && l < (nwl + n) && m >= n && m < (nwm + n)) {
    if(face == 1) {
        i1 = l; j1 = m; k1 = n_o;
        i2 = l; j2 = m; k2 = n + nwz;
    }
    else if(face == 2) {
        i1 = n_o; j1 = l; k1 = m;
        i2 = n + nwx; j2 = l; k2 = m;
    }
    else {
        i1 = l; j1 = n_o; k1 = m;
        i2 = l; j2 = n + nwy; k2 = m;
    }

    inc_i = l - n;
    inc_j = m - n;

    // Precursor Field index
    int pre_index = (inc_i * pre_coeff) + inc_j;

    field_material_l = ID[INDEX4D_ID(lookup_id, i1, j1, k1)];
    inc_l = inc_field_l[pre_index];
    f_l = updatecoeffs[INDEX2D_MAT(field_material_l, co)] * inc_l * sign_l;
    field[INDEX3D_FIELDS(i1, j1, k1)] += f_l;
    ...
```

`pre_coeff` is passed at launch as `np.int32(precursors.ex_bottom.shape[1])` —
the 2D precursor array's row stride supplied as a **runtime kernel argument**.
This is the one place the PR already does what the fresh port intends to do
everywhere.

### Launch-site architecture (the fatal part)

Each of `update_electric_is/magnetic_is/electric_os/magnetic_os` in
`CUDASubGridHSG` (`subgrids/subgrid_hsg.py` in the fork), *on every call*:

1. re-`Template.substitute`s and re-`SourceModule`-compiles the kernel
   (`update_*_is` is called `ratio` times per half-timestep from
   `SubgridUpdater.hsg_1/hsg_2`);
2. `gpuarray.to_gpu(...)` uploads its inputs — for the OS updates that is the
   **full main-grid `ID` (4D uint32 volume), coefficient tables and three full
   field volumes**, and for the IS updates 12 host-computed precursor arrays;
3. launches the six per-face kernels;
4. `.get()`s the full field volumes back to host
   (`main_grid.Ex = main_grid.Ex_gpu.get()`, `self.Hx = self.Hx_gpu.get()`, ...).

Precursor interpolation (`PrecursorNodes`) is untouched CPU code operating on
*host* main-grid arrays, while the main grid itself runs under `CUDAUpdates`
with device-resident fields — so mid-iteration the host copies the OS uploads
read are stale, and re-binding `main_grid.Ex_gpu = to_gpu(host_array)` can
clobber the live device fields. The scheme is per-call offload, not a GPU
port, and it is not coherent even as that.

## (c) Comparison with the current Cython kernels

Ground truth `gprMax/cython/fields_updates_hsg.pyx` (current devel). The three
kernels use an outer OpenMP `prange` over `l` and an inner loop over `m`, then
the same face-branch subscript assignment. `update_electric_os`:

```cython
n_s_l = nb - s * r - r + r // 2
n_s_r = nb + nwn + s * r + r // 2
os = nb - r * s

for l in prange(l_l, l_u, nogil=True, schedule='static', num_threads=nthreads):
    if mid == 1:
        l_s = os + (l - l_l) * r + r // 2
    else:
        l_s = os + (l - l_l) * r
    for m in range(m_l, m_u):
        if mid == 1:
            m_s = os + (m - m_l) * r
        else:
            m_s = os + (m - m_l) * r + r // 2
        if face == 2:
            i0, j0, k0 = n_l, l, m
            i1, j1, k1 = n_s_l, l_s, m_s
            i2, j2, k2 = n_u, l, m
            i3, j3, k3 = n_s_r, l_s, m_s
        ...
        material_e_l = ID[lookup_id, i0, j0, k0]
        inc_n = inc_field[i1, j1, k1] * sign_n
        field[i0, j0, k0] += updatecoeffsE[material_e_l, co] * inc_n
```

`update_magnetic_os` likewise with `n_s_l = nb - r * s`,
`n_s_r = nb + nwn + s * r`. `update_is`:

```cython
cdef int n_o = n + offset
for l in prange(n, nwl + n, nogil=True, schedule='static', num_threads=nthreads):
    for m in range(n, nwm + n):
        if face == 1:
            i1, j1, k1 = l, m, n_o
            i2, j2, k2 = l, m, n + nwz
        ...
        inc_i = l - n
        inc_j = m - n
        field_material_l = ID[lookup_id, i1, j1, k1]
        inc_l = inc_field_l[inc_i, inc_j]
        f_l = updatecoeffsE[field_material_l, co] * inc_l * sign_l
        field[i1, j1, k1] += f_l
```

Point-by-point:

| Aspect | Cython (current) | PR #297 CUDA |
|---|---|---|
| Iteration domain | `prange(l_l, l_u)` × `range(m_l, m_u)` — exactly the face | 1D launch over full grid volume; guard `l>=l_l && l<l_u && m>=m_l && m<m_u` masks to the face |
| `(l,m)` decode | loop indices | `l = idx / (NZ*NY); m = idx % (NZ*NY)` — slab decode, `m` is a raw slab offset that only survives the guard for `m < m_u` |
| Face dispatch | three `if face ==` blocks inside the loop body | identical three `if` blocks, but replicated per kernel with the whole update body inlined in each (3× code) |
| Sub-index arithmetic | `os + (l - l_l) * r [+ r // 2]` | identical, with `floor((double) sub_ratio / 2)` for `r // 2` (equal for positive `r`) |
| Dual-grid subscripts | memoryview `field[i,j,k]` (main) vs `inc_field[i,j,k]` (sub) — strides carried by the memoryviews | manual macros `INDEX3D_FIELDS` (main strides) vs `INDEX3D_SUBFIELDS` (sub strides), strides baked at template-substitution time |
| Precursor arrays (IS) | 2D memoryview `inc_field_l[inc_i, inc_j]` | flat pointer + runtime `pre_coeff` row stride |
| Datatype | `np.float64_t` memoryviews | `$REAL` arrays but `double inc_n, inc_f` temporaries (forces fp64 math even in fp32 builds) |
| Near/far faces | both updated per iteration | both updated per thread (same) |
| Parameter names | `r`, `s`, `nb` | `sub_ratio`, `surface_sep`, `n_boundary_cells` (same meanings; `nwn` kept for OS, the Cython `update_is` `nwn` arg is unused in both) |

The index *arithmetic* is a line-for-line match; every divergence is in
thread mapping, memory residency and the launch plumbing.

## (d) Known-wrong logic and flagged issues

No reviewer ever flagged anything (zero comments). The following defects come
from direct comparison of the PR head against its own CPU path and current
devel:

1. **`CUDASubGridHSG.__htod_precursor_fields` is missing `self`** —
   `def __htod_precursor_fields(precursors, offset):` is an instance method
   with no `self` parameter, called as
   `self.__htod_precursor_fields(precursors, -1)`. Every IS update raises
   `TypeError` (3 positional args into a 2-parameter function). The head
   commit cannot have been run as committed.

2. **Wrong field array in `update_electric_is`, front/back Ez launch** — the
   CPU path (both PR and current devel) does
   `update_is(..., self.Ez, precursors.hx_front, precursors.hx_back, IDlookup['Ez'], 1, -1, 2, ...)`,
   but the CUDA launch passes `self.Ex_gpu.gpudata` with `IDlookup['Ez']` and
   the `hx_front/hx_back` precursors: Ez never receives its front/back IS
   correction and Ex gets corrupted with Ez-node data instead.

3. **Swapped signs in `update_magnetic_is`, front/back pair** — CPU ground
   truth (PR's own file line 64–65, identical in current devel):
   `(Hz, ex_front/ex_back): sign_l=-1, sign_u=+1` and
   `(Hx, ez_front/ez_back): sign_l=+1, sign_u=-1`. The CUDA launches pass
   `(1, -1)` for the Hz/ex pair and `(-1, 1)` for the Hx/ez pair — the two
   sign pairs are exchanged. Bottom/top and left/right pairs match; only
   front/back is flipped. This alone would make any 3D run numerically wrong.

4. **Recompilation per call** — `SourceModule(kernel_template_*.substitute(...))`
   inside every update method, several times per timestep (pycuda's disk cache
   softens the nvcc cost but not the Python/module overhead).

5. **Per-call PCIe round-tripping and stale-host reads** — see (b). Full
   main-grid `ID`/fields uploaded and downloaded around every OS call, all
   precursors uploaded per IS call, full subgrid fields downloaded per IS
   call, and host-side precursor code reads main-grid host arrays that are
   stale while `CUDAUpdates` owns the device copies. Re-binding
   `main_grid.*_gpu` attributes mid-iteration can clobber live device state.

6. **~0.1% thread utilization** — full-volume launches masked to a 2D face
   (and a dead locally computed `bpg = ... / 128` in `update_magnetic_os` that
   is then ignored in favour of `main_grid.bpg`).

7. **fp64 temporaries in `$REAL` kernels** (`double inc_n, inc_f;`), and
   `NY_MATCOEFFS` substituted from `updatecoeffsH.shape[1]` in the *electric*
   IS path (harmless only because E/H coefficient tables share a shape).

8. **Author-acknowledged gaps**: no GPU precursor kernels (the whole
   time/space interpolation stage is CPU), and the admitted redundancy in
   `CUDASubGridHSG` (six near-identical launch blocks per method). No tests;
   the only validation artifact is a demo notebook
   (`user_models/sub-gridding/subgrid_basic_gpu.ipynb`).

9. **Guard removal without replacement**: `config.py`'s "CUDA + subgrid not
   supported" check was commented out rather than replaced with capability
   detection.

## (e) What to reuse / what to avoid for a fresh CUDA port

**Reuse**

- The **face-branch subscript tables and dual-grid index arithmetic** (the
  `n_s_l/n_s_r/os`, `l_s/m_s` expressions and the `(i0..k3)` assignments).
  They transcribe the Cython exactly and were re-verified here against
  `fields_updates_hsg.pyx`; they are the correct starting point for kernel
  bodies. (Equivalently: re-derive them from the Cython directly — the PR adds
  nothing beyond confirming the transcription is mechanical.)
- The **two-stride-set idea** (`INDEX3D_FIELDS` vs `INDEX3D_SUBFIELDS`): an OS
  kernel inherently addresses two grids, so it needs two independent
  (ny·nz, nz) stride pairs plus the 4D `ID` strides and the coefficient row
  stride. In the fresh port make all of these **runtime `int` arguments**
  (as current devel already does for `NX/NY/NZ` in
  `cuda_opencl/knl_fields_updates.py`) so one compiled kernel serves any
  main-grid/subgrid pairing and multiple subgrids without re-templating.
- The **`pre_coeff` runtime-stride pattern** from `hsg_update_is` — it is
  precisely the "dims as kernel arguments" convention, already proven in the
  PR for the precursor arrays.
- The per-face/per-component **launch argument tables** (face, `co`,
  `sign_n/sign_f` or `sign_l/sign_u`, `mid`, lookup ids, bounds) — but lift
  them from *current devel's* `subgrids/subgrid_hsg.py` CPU calls, not from
  the PR (defects 2 and 3 above live exactly in that transcription).

**Avoid**

- The full-volume masked launch. Launch a 1D (or 2D) grid over the actual face
  domain: `N_l = l_u - l_l`, `N_m = m_u - m_l`,
  `l = l_l + idx / N_m; m = m_l + idx % N_m`, with `N_m` a runtime argument —
  ~1000× fewer threads and no `face` branching if face is fixed per
  specialized kernel or resolved via precomputed axis-permutation arguments.
- Any per-call `SourceModule` compilation, `to_gpu`/`.get()` round trips, or
  rebinding of `*_gpu` attributes. The fresh plan's device-resident design
  (fields, `ID`, coefficients and precursors all living on the GPU; IS/OS
  corrections as kernels between the standard update kernels) is the fix; the
  PR is the cautionary demonstration of why halfway offload cannot work —
  correctness (stale hosts copies), not just speed.
- Copying the PR's launch argument lists verbatim (sign swap + wrong-array
  bugs), the `double` temporaries, and the missing-`self`/staticmethod
  pattern.
- Its class plumbing (`CPUSubGridBase`/`CUDASubGridBase`, `solvers.py` hooks):
  written against the 2021 codebase, superseded by today's
  `cuda_opencl`/`Updates` structure.

**Bottom line.** PR #297 is unreviewed, unfinished, frozen since Nov 2021 and
numerically wrong as committed (defects 1–3). Its value is narrow but real: it
confirms the Cython→CUDA transcription of the HSG index arithmetic is
mechanical, demonstrates the two-stride-set requirement for cross-grid
kernels, and already contains the runtime-stride-argument pattern the fresh
port generalizes. Everything at and above the launch layer should be written
from scratch against the current Cython ground truth.
