# Corpus Parity Plan

Goal: 100% corpus parity with gfortran and lfortran on **compile and runtime
correctness**, plus tracked compile/runtime performance, across the lfortran
`integration_tests`, `gfortran.dg`, and FortFront example corpora. Standard
Fortran is prioritized over lazy extensions. Fixes are authorized across ffc,
FortFront, LIRIC, and fo.

Live scoreboard: issue #299 (pass rates + performance, updated each session).
This document is the durable overview of every open issue and the phased plan
to close them. Update it when the issue set or phase boundaries change.

## Status snapshot

| Suite | PASS | FAIL | XFAIL / remaining | TOTAL |
|-------|-----:|-----:|------------------:|------:|
| fortfront-f90 | 324 | 0 | 102 | 426 |
| fortfront-lf  | 196 | 0 | 67  | 263 |
| lfortran      | 751 | 1 | 3519 | 4280 |
| gfortran-dg   | 1143 | 256 | 2159 (+2371 excluded) | 5938 |

Distance to 100% (files still failing or expected-fail, the real backlog):
lfortran ~3520, gfortran-dg ~2415 conformant, fortfront 102 f90 + 67 lf.
`gfortran.dg` is not a 100% target: 2371 files are error-detection, deprecated,
or vendor-extension tests excluded per `docs/CONFORMANCE.md`; the goal is its
runnable conformant subset plus correct rejection of its negative tests.

Two correctness debts tracked separately from feature coverage:
- **Negative-test acceptance (#291):** ffc accepts ~240 `gfortran.dg` invalid
  programs it should reject. This grows as features land and must be reversed
  by rejection diagnostics.
- **Silent miscompiles (#280):** a full scan finds ~186 xfailed lfortran files
  that compile and run but print output differing from gfortran (many print
  nothing where gfortran prints a result). These are the highest-severity bugs.

## Open-issue inventory

### ffc (this repo)

Umbrella and epics (the compliance spine, #272):
- **#272** path to F95/2003 + 100% corpus (umbrella).
- **#262** E1 lower FortFront-inferred (lazy/implicit) declarations.
- **#263** E2 module system and separate compilation.
- **#264** E3 array lowering coverage.
- **#265** E4 derived-type coverage.
- **#266** E5 procedure results, dummies, argument passing.
- **#267** E6 scalar type coverage.
- **#268** E7 Fortran runtime library (`ffc/runtime/`) and I/O.
- **#269** E8 numeric precision parity and intrinsic breadth.
- **#270** E9 control flow and remaining statement nodes.
- **#271** E10 drive corpora to 100% and gate in CI.

Feature gaps (each a bucket of xfailed files):
- **#273** runtime polymorphism: SELECT TYPE, class(T), class allocatable
  (needs vtables - hard frontier).
- **#274** separate-file modules: emit/consume `.fmod`, resolve external USE.
- **#275** top-level external procedures as program units.
- **#276** remaining array forms: char/derived element arrays, rank-2 sections,
  vector subscripts, multi-dim/runtime allocatable.
- **#277** remaining control flow: computed goto, EQUIVALENCE, directives.
- **#278** remaining I/O: E/EN descriptors, formatted read, scratch, complex(4).
- **#279** remaining intrinsics: uint, string, array intrinsics.
- **#281** lazy-Fortran support: inference, walrus, global scope, monomorphization.
- **#284** separate compilation: duplicate runtime-helper symbols.
- **#286** file:line:col diagnostics on stderr plus `--json`.
- **#294** mixed-kind generic dispatch per call-site argument kinds.
- **#303** generic dispatch rank tracking: same-kind specifics distinguished by
  argument rank (scalar vs array), resolved in PR #312.
- **#297** separate-file submodules from `.fmod`.

Correctness bugs:
- **#257** real(8) function dummy arguments not lowered as real.
- **#258** nested real intrinsic argument (is_f32 misses intrinsic results).
- **#260** libm float32/64 precision mismatch.
- **#280** compiles-but-mismatches-gfortran (the ~186 silent miscompiles).
- **#290** ICE "allocate already allocated: context" (31 dg files).
- **#291** negative-test acceptance (~240 dg files, missing rejections).
- **#295** 13 dg compile regressions on pre-baseline files.
- **#296** bind(c) module scalars break host association.
- **#298** latent module-export memory corruption (needs asan).
- **#300** real scalar reduction in a DO loop kept only the last term - FIXED
  this session; keep as a template for the #280 silent-miscompile class.

Process/triage: **#261** conformance drift detection, **#282** harden prior
work + re-land lazy-procedure cluster, **#283** lfortran corpus triage,
**#299** parity dashboard (living).

### FortFront (parser/AST gaps that block ffc corpus files)

Each blocks whole clusters and must be fixed upstream (route root-cause work to
a strong model in a FortFront worktree):
- **#2848** old-style DIMENSION statement parsed as a bare identifier.
- **#2849** DO CONCURRENT rejects the F2018 type-spec index form.
- **#2850** multi-unit split duplicates a program_node.
- **#2851** implicit-typing synthesis skips procedure bodies.
- **#2852** COMMON block named "block" loses its name.
- **#2853** BLOCK DATA not terminated by a bare END.
- **#2854** uppercase INTERFACE/MODULE PROCEDURE hide a generic interface.
- **#2855** EXTERNAL statement dropped from a program body.
- **#2856** function-statement integer kind dropped.
- **#2857** interface-body-form specifics dropped.
- **#2858** double-precision-function parses to a placeholder node.
- Infra: **#2840-2842** arena deep-copy/clone, **#2844** LSP reparse budget,
  **#2846** formatter drops inline comments.

### LIRIC (backend)

- **#520** backend C-API coordination for the compliance path.
- **#521** x86-64 ICMP-to-memory miscompile (ffc works around via select_value;
  removing the workaround depends on this).
- **#522** LLVM session backend is JIT-only + emits a duplicate `snprintf`
  declaration - blocks the LLVM-backend leg of the performance comparison.

### fo (build tool)

Parity-relevant: **#88** transient parallel-build race (occasionally inflates
gauntlet FAIL). Adjacent: **#55-57** fortfront link / LSP / `.lf` support,
**#59-62** deep lint/fmt and fortrun retirement. Not in parity scope:
**#1-7** the store/capsule/HPC backlog.

## Gap analysis (what the remaining backlog actually is)

From per-file triage (re-run `scratchpad/triage.py <suite>` after each wave;
first-error clustering is a coarse proxy - probe files for the real gap):

- **scalar-type bucket (~250):** heterogeneous - derived types with
  allocatable/pointer components, class(T), complex arrays. Needs feature
  stacking; single features land as foundations without clearing files.
- **function-call bucket (~365):** a grab-bag - calls to module/top-level/
  statement functions, array-valued results, transfer, elemental dispatch.
- **array bucket (~150):** assumed-shape runtime bounds, runtime-sized locals,
  vector subscripts, rank-2 sections.
- **character bucket (~200):** deferred/assumed-length, char arrays as dummies.
- **derived bucket (~130):** allocatable-array + pointer components, arrays of
  derived, deferred-length char components.
- **separate compilation (~31):** `separate_compilation_*b` files need #274.
- **silent miscompiles (~186, #280)** and **negative-accept (~240, #291).**

## Multi-week phased plan

Delivery is by cost-tiered agent **waves** (see Execution mechanics). Each
phase has a measurable exit target. Phases overlap: correctness (C) runs
alongside features (B) every wave.

**Phase A - measurement + infrastructure. DONE.**
Baseline all suites; `--backend {isel,copy-patch,llvm}` flag; `perf_compare.py`
+ benchmarks; dashboard #299; per-file triage tooling. Exit: baseline recorded,
perf table published, tracker live. (Complete.)

**Phase B - feature waves (weeks 1-3).** Close corpus xfail buckets by probe-
first waves, biggest verified levers first: derived allocatable-array/pointer
components, function-call breadth, runtime/assumed-shape arrays, character
breadth, array/scalar intrinsics. Interleave FortFront parser fixes
(#2848-2858), which unblock whole clusters. Exit target: lfortran ≥ 1500 PASS,
fortfront-f90 ≥ 360, FortFront parser blockers closed.
Milestones: +50 lfortran files/wave early, tapering as the deep frontier is
reached; feature-stacking waves (combine landed primitives) to clear
multi-feature files.

**Phase C - correctness hardening (weeks 1-4, continuous).**
- **#280 silent miscompiles:** drive the ~186 wrong-output files to zero, one
  root-cause cluster per wave (the #300 real-loop and bare-`print` fixes are the
  template). Highest severity; front-loaded.
- **#291 negative-test rejection:** add rejection diagnostics class by class
  (type/kind/rank mismatch, bad kinds, duplicate declarations), gated on zero
  valid-program regression, until dg negative-accept → 0.
- **#295, #296, #298, #290, #260, #257, #258** individual bug fixes.
Exit target: #280 wrong-output = 0, dg negative-accept < 20, all listed bugs
closed or root-caused upstream.

**Phase D - hard frontiers (weeks 3-5).**
- **#273** polymorphism/vtables (SELECT TYPE, class(T)).
- **#274/#297** separate-file modules and submodules (`.fmod`).
- Runtime-bound/automatic arrays, vector subscripts (#276).
- **#294** mixed-kind generic dispatch.
- **LIRIC #521/#522** removed workaround + working LLVM backend.
Exit target: separate compilation and basic polymorphism land; LLVM perf leg
measured.

**Phase E - convergence + CI gate (week 5-6).**
Drive each gated suite to its 100% target, wire the gate in CI (#271), close
the epic umbrellas (#272 and E1-E10), and keep the perf table green. Exit:
CI-gated 100% conformant subset, umbrellas closed.

**Phase F - performance parity (continuous).** Keep ffc(native) and, once
LIRIC #522 lands, ffc(llvm) compile-time and runtime competitive with
gfortran/lfortran; file LIRIC perf issues as they surface. Current: ffc
compiles ~10x faster than gfortran, runs at gfortran -O0 class.

## Execution mechanics

- **Waves:** a Workflow spawns N worktree-isolated agents, one per cluster,
  cost-tiered (opus for hard/ambiguous, sonnet for mechanical, fable/gpt for
  deep FortFront root-cause). Each probes ~8 sample files, implements the
  dominant sub-gap, adds a behavioral test, keeps `fo test` green, and commits
  a `wave<N>/<key>` branch without pushing.
- **Harvest:** cherry-pick each branch onto main; resolve `SUPPORT_CONTRACT.md`
  row conflicts with `scratchpad/rowmerge.py` (hand-merge README prose);
  `fo clean` rebuild (delete `*lowering*.o` - `.inc` content is not hashed);
  `fo test`; run the four gauntlets with no concurrent builds; verify the
  stable dg positive-FAIL set (16); promote XPASS by trimming the manifest;
  commit and push per wave.
- **Gates:** never weaken/skip a test; a new lowering path must decline
  gracefully, never hard-error a case a prior path accepted; every rejection
  must re-verify valid near-misses still compile.
- **Disk:** ~3.5 GB per worktree build tree; cap ~6-8 concurrent worktrees and
  `git worktree remove --force` + `git worktree prune` promptly.

## Known-hard / blocked

- Polymorphism (#273) needs a vtable ABI decision (coordinate via LIRIC #520).
- Separate-file modules (#274/#297) need `.fmod` to record dummy names and a
  mutable dummy-binding path (arena is currently `intent(in)` through lowering).
- #298 needs asan to fix safely (layout-sensitive OOB write on module_exports).
- Some lfortran files are blocked on the FortFront parser gaps above; ffc waves
  cannot fix them until the upstream fix lands.
