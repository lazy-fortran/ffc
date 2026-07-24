# ffc Development Plan

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style
source. The active pipeline is:

```
Fortran / Lazy Fortran source
  -> FortFront typed AST + diagnostics
  -> ffc lowering + runtime ABI
  -> LIRIC C API (via ISO_C_BINDING)
  -> object file / executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI decisions,
runtime calls, LIRIC bindings, and object/exe emission.

## Architecture decisions

- New compiler work goes through LIRIC's session C API.
- `ffc` does not add LLVM or MLIR bindings.
- A need for private FortFront AST layout is filed as a FortFront issue,
  not as an ffc workaround.

The retired MLIR/HLFIR experiment lives only in git history. Reference it
by commit hash if you need to look back, but do not revive it without an
explicit decision.

## Path to standard Fortran conformance

The target is the Fortran standard through F2023, minus the parallel and vendor
features excluded below. The standard defines what must work; the corpora only
measure how far along we are. A corpus file is evidence, never the goal, so a
case that tests another compiler's extension surface is excluded rather than
chased.

Excluded, and never counted in any denominator:

- coarrays, images, teams, events, and collective subroutines
- OpenMP, OpenACC, and MPI
- GPU and device backends
- vendor extensions outside the standard
- features deleted from the standard
- compiler-option and DejaGNU-harness behavior the runner does not model

Current pass rates, from the checked-in snapshot in
`test/conformance/parity_dashboard.tsv`: `fortfront-f90` 341/441,
`fortfront-lf` 205/264, `lfortran` integration 848/4280, `gfortran.dg`
1175/5938. Read the snapshot rather than this paragraph; regenerate it with
`scripts/generate_parity_dashboard.sh` whenever corpus state changes.

Those denominators are raw file counts, not the conformance denominator. 241
cases are still `NOREF`, meaning undefined output, missing linkage, or a
harness contract the runner does not model. Until #430 classifies them, the
number that 100% is 100% *of* is not yet known. Finish that classification
before quoting a conformance percentage.

The `E1` through `E10` epics (#262 through #271), the `#272` compliance
umbrella, and the LIRIC coordination issue `krystophny/liric#520` are all
closed. They were split into the atomic issues that now carry the work; do not
cite them as the live plan.

The live work order is the chunk sequence in the workspace roadmap <!-- slop-ok: names a real document -->: freeze the
public compiler graph, centralize typed lowering, stabilize the
descriptor/runtime/backend ABIs, make module artifacts authoritative, route
arrays and I/O through shared engines, then close corpus breadth. Each chunk
names its own atomic issues.

Neither external corpus is a 100% target as a whole. `gfortran.dg` contains
error-detection, deprecated, and vendor-extension tests; the `lfortran`
integration suite exercises that compiler's own extension surface. Gate only
the runnable, standard-conforming subset of each and document the exclusions in
`docs/CONFORMANCE.md`. The two FortFront corpora are maintained in-tree and are
100% targets once their `NOREF` cases are classified.

F2023 is part of the target, and the delta it adds over F2018 is unscoped. The
`[ffc-f2023-*]` trackers (#243 through #255) were closed after being split into
the current issue set, but that split covered F95-through-F2018 language
coverage; the syntax and intrinsics F2023 itself introduced were never
enumerated. #473 owns auditing that delta so "newest standard" does not quietly
mean F2018.

## Shipped baseline: direct LIRIC session backend

Covered features and the public claim live in
`docs/SUPPORT_CONTRACT.md`. Roughly:

- main program, scalar integer / real / logical, fixed-length and
  deferred-length character;
- arithmetic, comparisons, logical conditions;
- block `if` with PHI merges; counted `do` with literal positive and
  negative step; `SELECT CASE` with single- and multi-arm terminating
  bodies (including multi-label arms) and `case default`;
- contained integer / real / logical functions and subroutines, including
  early `return`;
- fixed-size 1-D integer arrays and rank-2 integer arrays with scalar
  element access, array sections with compile-time integer bounds as
  rvalues, whole-array copy, elemental `+`/`-`/`*`, and the array
  intrinsics `size`, `shape`, `sum`, `product`, `maxval`, and `minval`; simple
  derived types with scalar integer components;
- minimal `print *, expr`, compound formatted `print fmt, items` with literal
  `I`, `X`, and `F` descriptors, `stop <expr>`, `abs` / `min` / `max` / `mod`
  and integer-to-real `real()`;
- compile-time `//` folding for character literal chains;
- CLI: `-o`, `-c`, `-I <dir>` accepted (`-I` not yet consumed).

This surface is the baseline the compliance epics build on.

## Runtime, ABI, and conformance

The current ABI is documented in `docs/RUNTIME_ABI.md`. Conformance against
external corpora is documented in `docs/CONFORMANCE.md`. Each new feature must
update both documents and add executable tests in the same change.

The Fortran I/O and intrinsics runtime lives in `ffc/runtime/` (the local
`libgfortran` equivalent), linked through LIRIC's `lr_session_set_runtime_archive`.
It carries a stable C ABI so it can split into its own repo once it stabilizes.
LIRIC stays a backend-neutral codegen layer; no Fortran-language semantics land
there. This work is E7 (#268) and E8 (#269).

## FortFront boundary

`ffc` reaches into FortFront's arena (`select type (node => ...)`) for most
lowering. Each new lowering function should prefer a public FortFront query
over reaching into arena internals. A need for private FortFront AST layout is
filed as a FortFront issue, not an ffc workaround.

## Verification

```bash
export LIBRARY_PATH=<liric-build>   # so the LIRIC static library is linkable
fo                                  # static analysis, build, tests, lint, fmt
```

Use `fo` for every build and test loop; call `fpm` directly only to isolate one
named test or to diagnose a `fo` failure. CI runs the same workflow on every
push and pull request. Run `fo` before pushing.
