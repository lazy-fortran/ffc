# ffc Backlog

This backlog tracks the active FortFront to LIRIC compiler path. The old
HLFIR/MLIR backlog is obsolete and must not drive new work.

## P0: Keep Main Honest

- Keep `fpm.toml` defaulting to `src_mvp/`.
- Keep MLIR/HLFIR code outside the default build.
- Keep executable tests for every claimed compiler feature.

## P1: Direct LIRIC Session Lowering

Goal: keep the compiler path on direct `lr_session_*` emission.

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
- Tracked by #56: generalize non-terminating `if` merges beyond the current
  integer assignment subset.
- Done: move the CLI default to direct session lowering.
- Done: lower minimal scalar `print` for integers, real values, character
  literals, and logical literals through direct-session `printf` calls.
- Done: lower scalar real declarations, assignments, arithmetic, and printing.
- Done: lower scalar logical declarations, assignments, `if (flag)`, and
  printed logical variables.
- Done: lower simple contained integer functions and integer call expressions.
- Done: lower simple contained integer subroutines and explicit `CALL`
  statements.
- Done: lower integer procedure arguments through LIRIC pointer parameters with
  copy-back for variable actual arguments.
- Done: close krystophny/liric#519 with executable PHI-loop regression
  coverage and consume the validated direct-session loop shape in `ffc`.
- Done: split contained-procedure lowering out of the main direct-session
  lowerer before it reached the module size limit.
- Done: remove the old bootstrap `.ll` reference path from the default MVP
  source and test set.

Verification:

```bash
LIBRARY_PATH=<liric-build> fpm test test_liric_session_bindings
LIBRARY_PATH=<liric-build> fpm test test_session_empty_program_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_empty_program_object_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_stop_code_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_integer_variable_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_block_if_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_if_merge_compiler
LIBRARY_PATH=<liric-build> fpm test test_counted_do_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_scalar_print_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_real_literal_print_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_character_literal_print_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_logical_literal_print_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_logical_variable_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_real_variable_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_integer_function_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_integer_subroutine_compiler
```

## P2: FortFront API Needs

Tracked by #58. `ffc` still reaches into arena/node representation for much of
lowering. FortFront now exposes explicit subroutine call queries; more
compiler-facing queries are required so `ffc` can lower without depending on
private layout details.

Needed API surface:

- node kind and child/body queries
- typed expression queries
- declaration and symbol lookup queries
- literal value queries
- procedure signature and call-site queries
- source location and diagnostic mapping

## P3: Runtime And ABI

Document and test the ABI before claiming support for each feature.

- Done: program entry and exit convention.
- Done for current subset: integer scalar procedure storage and calling
  convention.
- #50: non-integer pass-by-reference rules and function results.
- Done for current subset: logical `i32` representation.
- #51: character value plus length representation.
- #55: print/runtime call surface.
- #52 and #53: array descriptor representation.

## Issue Map

- #50: Direct session: support non-integer scalar procedure ABI.
- #51: Direct session: implement scalar character variables and ABI.
- #52: Direct session: support fixed-size one-dimensional arrays.
- #53: Runtime ABI: design descriptor-backed allocatable arrays.
- #54: Compiler: support modules, separate compilation, and name mangling.
- #55: Runtime: replace `printf` shim with Fortran-aware scalar I/O.
- #56: Direct session: generalize control-flow value merging.
- #57: Diagnostics: add source locations and targeted unsupported-feature
  errors.
- #58: Frontend boundary: replace arena internals with FortFront compiler
  queries.
- #59: Conformance: add standard/Infer-mode end-to-end coverage.
- #60: Direct session: implement simple derived types and component access.
- #61: Direct session: lower first scalar intrinsic functions.

Features outside this map are not part of the current implementation plan
until a new issue names the feature, contract, tests, and dependencies.
