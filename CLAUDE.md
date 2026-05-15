# ffc Project Guide

## Mission

`ffc` is the Lazy Fortran compiler driver. The active architecture is:

```text
Fortran / Lazy Fortran source
    -> FortFront typed AST and diagnostics
    -> ffc lowering and runtime ABI
    -> LIRIC C API through ISO_C_BINDING
    -> object file or executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI decisions, runtime
calls, LIRIC bindings, object emission, and executable emission.

## Backend Direction

- Use direct LIRIC `lr_session_*` APIs for new compiler work.
- Keep the `.ll` text path only as temporary bootstrap/reference coverage while
  direct session lowering catches up.
- Do not add LLVM bindings or revive the MLIR/HLFIR path unless the backend
  decision is explicitly reopened.
- Keep legacy MLIR/HLFIR code outside the default `fpm` build.

## Current Build

- Default `fpm` sources are under `src_mvp/`.
- The CLI parses through FortFront's compiler-facing API.
- The CLI emits executables through direct LIRIC session lowering for the
  currently supported subset.

## Supported Direct-Session Slice

- Empty `program main`.
- Integer literal and arithmetic `stop` codes.
- Integer declarations and assignments consumed by `stop`.
- Integer comparison `if` blocks where both branches terminate with `stop`.
- Fallthrough integer `if` blocks that merge assigned integer values before
  `stop`.
- Literal-bound counted `do` loops expanded through direct LIRIC scalar
  operations.
- Minimal `print *, expr` for integer expressions, real literals, character
  literals, and logical literals through direct-session external `printf`
  calls.
- Real declarations, assignments, arithmetic expressions, and printed real
  variables.
- Simple contained integer functions with integer parameters, assignment to the
  function result name, and integer call expressions.

## Immediate Work

- Add narrow FortFront compiler query APIs so lowering does not depend on raw
  arena internals.
- Lower mutable storage through direct LIRIC once alloca/load/store executable
  behavior is covered.
- Generalize non-terminating control flow with merge values.
- Add runtime counted `do` loops through LIRIC blocks with backedge PHI values
  after krystophny/liric#519.
- Broaden runtime/ABI calls beyond integer `print`.
- Split the direct-session lowerer and binding modules before they cross the
  hard 1000-line module limit.

## Commands

Build:

```bash
LIBRARY_PATH=/home/ert/code/liric/build fpm build
```

Focused MVP tests:

```bash
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_liric_session_bindings
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_empty_program_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_stop_code_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_block_if_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_if_merge_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_counted_do_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_scalar_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_real_literal_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_character_literal_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_logical_literal_print_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_real_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_function_compiler
```

## Documentation

- `README.md`: current user-facing status.
- `ROADMAP.md`: active implementation phases.
- `BACKLOG.md`: current prioritized work.
- `docs/`: legacy MLIR/HLFIR references unless the document explicitly says
  otherwise.
