# ffc Backlog

This backlog tracks the active FortFront-to-LIRIC compiler path. The
retired MLIR/HLFIR experiment lives only in git history; do not revive it
without an explicit decision.

## P0: Keep main honest

- Keep `fpm.toml` pointing at `src/`.
- Keep executable behavioural tests for every claimed compiler feature.
- Keep CI green; never weaken or skip a failing test to do so.

## P1: Direct LIRIC session lowering

The direct LIRIC session lowerer covers the subset documented in
`docs/SUPPORT_CONTRACT.md`. The currently supported surface includes:

- empty `program main`, integer/real/logical scalars, character literals
  and fixed-length character variables;
- deferred-length `character(len=:), allocatable` including self-aliasing
  and three-way `//` concatenation;
- block `if` with PHI merge for scalars; array-element writes inside `if`
  branches; counted `do` with literal positive and negative step;
- contained integer/real/logical functions and subroutines with scalar
  args; early `return` inside contained functions and subroutines;
- fixed-size 1-D integer arrays and rank-2 integer arrays with scalar
  element access plus array sections with compile-time integer bounds as
  rvalues, whole-array copy, elemental `+`/`-`/`*`, and the array
  intrinsics `size`, `shape`, `sum`, `maxval`, and `minval`, simple
  derived types with scalar integer components;
- single-arm and multi-arm `SELECT CASE` (incl. multi-label and
  `case default`), each arm terminating with `stop`/`return`;
- compile-time `//` folding for character literal chains;
- minimal `print *, expr`, `stop <expr>`, scalar `abs`/`min`/`max`/`mod`
  and integer-to-real `real()` conversion;
- CLI: `-o`, `-c`, `-I <dir>` (`-I` stored, not yet consumed).

This surface is the baseline. The path to Fortran 95 / 2003 compliance and
100% corpus pass is tracked in #272.

## P1: Compliance path

#272 measures the path as 100% corpus pass with output matching `gfortran -w`
and `lfortran`. Phase-ordered epics:

- Phase 0, unblocks the most: E1 inferred-declaration lowering #262, E2 module
  system and separate compilation #263.
- Phase 1, type coverage: E3 arrays #264, E4 derived types #265, E5 procedure
  results/dummies/args #266, E6 scalars #267.
- Phase 2, runtime: E7 `ffc/runtime/` I/O library #268, E8 precision and
  intrinsics #269.
- Phase 3: E9 control flow and statement nodes #270.
- Phase 4: E10 drive corpora to 100% and gate in CI #271.

Open bugs #257-#261 fold into E5/E8/E10.

## P2: FortFront API boundary

`ffc` still walks the FortFront AST arena with `select type` for most lowering.
New compiler-facing FortFront queries should replace those reach-ins one slice
at a time.

## P3: Runtime and ABI

The current direct-session ABI is in `docs/RUNTIME_ABI.md`. Each new feature
must update that document and grow executable tests in the same change. The
Fortran I/O and intrinsics runtime lives in `ffc/runtime/`, linked through
LIRIC's `lr_session_set_runtime_archive`; LIRIC stays backend-neutral.
Tracked by E7 #268 and E8 #269.

## Verification

```bash
LIBRARY_PATH=<liric-build> fpm build
LIBRARY_PATH=<liric-build> fpm test
```

CI runs the same on every push and pull request.
