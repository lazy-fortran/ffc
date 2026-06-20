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

## Path to Fortran 95 / 2003 compliance

The full path is tracked in #272, measured as 100% corpus pass with output
matching `gfortran -w` and `lfortran`. Current pass rates: `fortfront-f90`
164/400, `fortfront-lf` 28/255, `lfortran` integration ~123/4257.

Two root-cause blockers dominate every corpus and come first:

- E1 inferred-declaration lowering (lazy typing): #262.
- E2 module system and separate compilation: #263.

Then type coverage (E3 arrays #264, E4 derived types #265, E5 procedure
results/dummies/args #266, E6 scalars #267), the runtime (E7 `ffc/runtime/`
I/O library #268, E8 precision and intrinsics #269), control flow (E9 #270),
and convergence (E10 corpora to 100% plus CI gate #271). Backend coordination
with LIRIC is krystophny/liric#520; no concrete backend gap is open.

`gfortran.dg` is not a 100% target: it contains error-detection, deprecated,
and vendor-extension tests. Gate only its runnable, conformant subset and
document the exclusions in `docs/CONFORMANCE.md`.

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
LIBRARY_PATH=<liric-build> fpm build
LIBRARY_PATH=<liric-build> fpm test
```

CI runs the same workflow on every push and pull request. Run these
commands before pushing.
