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

## Phase A: direct LIRIC session backend (in progress)

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
  element access, whole-array copy, elemental `+`/`-`/`*`, and the array
  intrinsics `size`, `shape`, `sum`, `maxval`, and `minval`; simple
  derived types with scalar integer components;
- minimal `print *, expr`, compound formatted `print fmt, items` with literal
  `I`, `X`, and `F` descriptors, `stop <expr>`, `abs` / `min` / `max` / `mod`
  and integer-to-real `real()`;
- compile-time `//` folding for character literal chains;
- CLI: `-o`, `-c`, `-I <dir>` accepted (`-I` not yet consumed).

The detailed slices are tracked as GitHub issues. The self-hosting
dependency map is in #167.

## Phase B: runtime, ABI, and conformance (continuous)

The current ABI is documented in `docs/RUNTIME_ABI.md`. Conformance
against external corpora is documented in `docs/CONFORMANCE.md`. Each
new feature must update both documents and add executable tests in the
same change. Major open categories:

- non-integer scalar procedure ABI: #50, #164.
- character value + length representation beyond the current MVP: #51.
- a Fortran-aware scalar I/O runtime replacing the `printf` shim: #55.
- array descriptors and allocatable lifecycle: #53, slices #184-#186.
- general control-flow value merging across SELECT CASE / IF: #56, #175,
  #176, #180.

## Phase C: FortFront boundary

`ffc` reaches into FortFront's arena (`select type (node => ...)`) in
~50 sites. Replacing those with named compiler queries is tracked by #58
and #173. Each new lowering function should prefer a public FortFront
query over reaching into arena internals.

## Verification

```bash
LIBRARY_PATH=<liric-build> fpm build
LIBRARY_PATH=<liric-build> fpm test
```

No CI is configured for this repository. Run these commands before
pushing.
