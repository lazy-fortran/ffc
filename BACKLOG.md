# ffc Backlog

This backlog tracks the active FortFront to LIRIC compiler path. The old
HLFIR/MLIR backlog is obsolete and must not drive new work.

## P0: Keep Main Honest

- Keep `fpm.toml` defaulting to `src_mvp/`.
- Keep MLIR/HLFIR code outside the default build.
- Keep docs explicit that `.ll` text emission is a bootstrap path, not the
  target architecture.
- Keep executable tests for every claimed compiler feature.

## P1: Direct LIRIC Session Lowering

Goal: replace `.ll` text lowering with direct `lr_session_*` emission.

- Done: bind `lr_session_create/destroy`.
- Done: bind scalar `i32` type access.
- Done: bind function begin/end, block creation, block selection, instruction
  emission, and executable emission.
- Done: prove direct session executable emission with `main` returning zero.
- Done: create a session lowerer for empty `program main`.
- Done: lower integer literal and binary arithmetic expressions to LIRIC vregs.
- Done: lower integer declarations and assignments to direct-session values.
- Done: lower terminating integer comparison `if` blocks to LIRIC blocks.
- Next: lower mutable integer storage once direct executable alloca/load/store
  behavior is covered.
- Next: lower non-terminating `if` with merge values.
- Next: lower counted `do` loops to LIRIC blocks, not unrolled text.
- Done: move the CLI default from bootstrap `.ll` emission to direct session
  lowering for the currently supported direct-session subset.
- Next: close direct-session feature gaps until the old bootstrap path can be
  removed.

Verification:

```bash
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_liric_session_bindings
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_empty_program_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_stop_code_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_block_if_compiler
```

## P2: Bootstrap Feature Parity

The bootstrap `.ll` path currently proves these executable slices:

- empty `program main`
- integer scalar declarations, assignment, arithmetic, and `print`
- one-line and block integer `if`
- literal-bound counted `do` by unrolling
- real literals and real scalar variables/arithmetic
- character literal printing
- logical literal printing

Keep this path stable while porting behavior to the direct LIRIC session
lowerer. Do not add broad new features only to the bootstrap path.

## P3: FortFront API Needs

`ffc` still reaches into arena/node representation for lowering. FortFront
should expose narrower compiler-facing queries so `ffc` can lower without
depending on private layout details.

Needed API surface:

- node kind and child/body queries
- typed expression queries
- declaration and symbol lookup queries
- literal value queries
- procedure signature and call-site queries
- source location and diagnostic mapping

## P4: Runtime And ABI

Document and test the ABI before implementing procedures and arrays.

- program entry and exit convention
- scalar storage and calling convention
- pass-by-reference rules
- function result variables
- logical representation
- character value plus length representation
- print/runtime call surface
- array descriptor representation

## Deferred

- full Fortran module compilation
- derived types
- allocatable arrays and pointers
- generics and monomorphization
- cross-module inference orchestration
- optimization beyond basic lowering correctness
