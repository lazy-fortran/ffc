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

```bash
LIBRARY_PATH=/path/to/liric/build fpm build
LIBRARY_PATH=/path/to/liric/build fpm test
```

## Development rules

- Add new executable behaviour to the direct LIRIC session path; keep the
  CLI on `session_program_lowering`.
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

The remaining roadmap is the self-hosting tracker (#167) plus open
issues for individual slices.

## Verification

Use repo-declared fpm targets. Do not invent build commands.

```bash
LIBRARY_PATH=/path/to/liric/build fpm test          # full suite
LIBRARY_PATH=/path/to/liric/build fpm test test_session_empty_program_compiler
```

For CLI checks:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=/path/to/liric/build fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
LIBRARY_PATH=/path/to/liric/build fpm run ffc -- /tmp/empty.f90 -c -o /tmp/empty.o
```

CI runs the same workflow on every push and pull request.
