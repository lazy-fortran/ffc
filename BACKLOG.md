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
- Done: bind direct session object emission and expose `ffc -c`.
- Done: create a session lowerer for empty `program main`.
- Done: lower integer literal and binary arithmetic expressions to LIRIC vregs.
- Done: lower integer declarations and assignments to direct-session values.
- Done: lower terminating integer comparison `if` blocks to LIRIC blocks.
- Done: lower fallthrough integer comparison `if` blocks that merge assigned
  integer values before a later `stop`.
- Done: lower counted `do` loops with runtime-computed integer bounds through
  direct LIRIC session blocks and PHI backedges.
- Next: lower mutable integer storage once direct executable alloca/load/store
  behavior is covered.
- Next: generalize non-terminating `if` merges beyond the current integer
  assignment subset.
- Done: move the CLI default from bootstrap `.ll` emission to direct session
  lowering for the currently supported direct-session subset.
- Done: lower minimal scalar `print` for integers, real values, character
  literals, and logical literals through direct-session `printf` calls.
- Done: lower scalar real declarations, assignments, arithmetic, and printing.
- Done: lower scalar logical declarations, assignments, `if (flag)`, and
  printed logical variables.
- Done: lower simple contained integer functions and integer call expressions.
- Done: lower simple contained integer subroutines and explicit `CALL`
  statements.
- Done: close krystophny/liric#519 with executable PHI-loop regression
  coverage and consume the validated direct-session loop shape in `ffc`.
- Done: split contained-procedure lowering out of the main direct-session
  lowerer before it reached the module size limit.
- Next: close direct-session feature gaps until the old bootstrap path can be
  removed.

Verification:

```bash
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_liric_session_bindings
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_empty_program_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_empty_program_object_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_stop_code_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_block_if_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_if_merge_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_counted_do_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_scalar_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_real_literal_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_character_literal_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_logical_literal_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_logical_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_real_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_function_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_subroutine_compiler
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

`ffc` still reaches into arena/node representation for much of lowering.
FortFront now exposes explicit subroutine call queries; more compiler-facing
queries should follow so `ffc` can lower without depending on private layout
details.

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
