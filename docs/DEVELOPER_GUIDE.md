# ffc Developer Guide

## Architecture

`ffc` connects FortFront to LIRIC:

```text
source -> FortFront typed AST -> ffc lowering and ABI -> LIRIC C API -> executable
```

FortFront stays backend-neutral. `ffc` owns lowering, runtime calls, ABI
decisions, and behavioural executable tests.

## Build

The default fpm library source is `src/`. The retired MLIR/HLFIR
experiment lives only in git history.

Use `fo` revision `32ef96d` or newer so `SUBMODULE` parent identifiers become
real build-graph edges. No filename ordering or `_order.f90` helper modules are
part of the source contract.

```bash
LIBRARY_PATH=/path/to/liric/build fo build
LIBRARY_PATH=/path/to/liric/build fo test
```

## Development rules

- Add new executable behaviour to the direct LIRIC session path; keep the
  CLI on `session_program_lowering`.
- Keep compiler clients on the two-procedure `session_program_lowering`
  facade. Lowering implementation units belong under
  `session_program_lowering_impl` and expose only the procedures their
  descendants require.
- Extract growing implementation units into modules or submodules with
  explicit interfaces. Let `fo` derive submodule ancestry from the source; do
  not add ordering shims or production `include` fragments.
- Keep constant-initialization validation in
  `session_program_lowering_reject_const_init.f90`. It is an immutable-AST
  descendant service with explicit interfaces; changes require both accepted
  and rejected executable/compiler oracles in
  `test_session_reject_const_01_compiler`.
- Add focused behavioural tests under `test/`. Each file is a standalone
  `program test_*` picked up by fpm auto-discovery.
- Treat a need for private FortFront AST layout as a FortFront API issue
  (see #58 / #173) rather than an `ffc` workaround.
- Before claiming support for a feature, update
  `docs/SUPPORT_CONTRACT.md`.
- If a feature changes calling convention, storage, or runtime calls,
  update `docs/RUNTIME_ABI.md` in the same change.
- Never `git add .` or `git add -A`. Stage paths explicitly.

## Feature order

The supported surface is in `docs/SUPPORT_CONTRACT.md`. Broadly the order
in which features have been added (and the order new slices should
follow) is:

1. Empty programs and integer `stop`.
2. Scalar integer declarations and assignments.
3. Integer arithmetic and comparisons.
4. Minimal `print *, expr`.
5. Block `if`, fallthrough integer merges, counted `do`.
6. Real and logical scalars; minimal character output.
7. Contained integer / real / logical functions and subroutines.
8. Fixed-size 1-D integer arrays.
9. Simple derived types with scalar integer components.
10. Deferred-length character (assignment, concatenation, self-aliasing).
11. `SELECT CASE` with terminating arms (single, multi, multi-label).
12. Early `return` inside contained subroutines and functions.
13. CLI `-I <dir>` accepted (storage only; consumption is future work).

The remaining work is the self-hosting tracker (#167) plus open
issues for individual slices.

## Verification

Use the repo-declared `fo` targets. Do not invoke build-tree binaries directly.

```bash
LIBRARY_PATH=/path/to/liric/build fo test          # full suite
LIBRARY_PATH=/path/to/liric/build fo test test_session_empty_program_compiler
```

For CLI checks:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=/path/to/liric/build fo exec ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
LIBRARY_PATH=/path/to/liric/build fo exec ffc -- /tmp/empty.f90 -c -o /tmp/empty.o
```
