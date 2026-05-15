# ffc Developer Guide

## Architecture

`ffc` connects FortFront to LIRIC:

```text
source -> FortFront typed AST -> ffc lowering and ABI -> LIRIC C API -> executable
```

FortFront stays backend-neutral. `ffc` owns lowering, runtime calls, ABI
decisions, and executable tests.

## Default Build

The default fpm library source is `src_mvp/`. The old experimental source tree
under `src/` is not part of the default build.

Build with LIRIC on the linker search path:

```bash
LIBRARY_PATH=/path/to/liric/build fpm build
```

## Development Rules

- Add new executable behavior to the direct LIRIC session path.
- Keep the CLI on `session_program_lowering`.
- Add focused executable tests in `test_mvp/`.
- Keep bootstrap reference tests only where they compare behavior not yet
  available through the direct session path.
- Treat a need for private FortFront AST layout as a FortFront API issue.

## Feature Order

1. Empty programs and integer `stop`.
2. Scalar integer declarations, assignments, arithmetic, and comparisons.
3. Minimal `print *, expr`.
4. Non-unrolled counted loops.
5. Procedures and explicit ABI tests.
6. Character representation and a fuller runtime surface.
7. Arrays, modules, derived types, allocatables, and generics.

## Verification

Use repo-declared fpm targets. Do not invent build commands.

```bash
LIBRARY_PATH=/path/to/liric/build fpm test test_session_empty_program_compiler
LIBRARY_PATH=/path/to/liric/build fpm test test_session_integer_variable_compiler
LIBRARY_PATH=/path/to/liric/build fpm test test_session_block_if_compiler
LIBRARY_PATH=/path/to/liric/build fpm test test_session_if_merge_compiler
```

For CLI checks:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=/path/to/liric/build fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
```
