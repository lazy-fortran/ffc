# ffc

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style source.

The active build is the FortFront + direct LIRIC session compiler path. The
older MLIR/HLFIR experiment remains in `src/` as legacy reference code, but it
is not part of the default fpm build.

## Current Status

- The package builds the MVP sources in `src_mvp/`.
- The CLI parses files through FortFront's compiler-facing frontend API.
- The CLI lowers through the direct LIRIC session API and emits a runnable
  executable without `.ll` text.
- The direct LIRIC session lowerer can compile empty `program main`, integer
  arithmetic `stop` codes, and integer declarations/assignments when the
  assigned value is consumed by `stop`.
- The direct LIRIC session lowerer can compile integer comparison `if` blocks
  when both branches terminate with `stop` or both branches assign integer
  values that merge before a later `stop`.
- The direct LIRIC session lowerer can compile counted `do` loops with
  runtime-computed integer bounds through LIRIC blocks and PHI backedges.
- The direct LIRIC session lowerer can compile minimal integer
  `print *, expr`, real literal `print`, character literal `print`, and
  logical literal `print` through external `printf` calls.
- The direct LIRIC session lowerer can compile scalar `real` declarations,
  assignments, arithmetic, and printing of real variables.
- The direct LIRIC session lowerer can compile scalar `logical` declarations,
  assignments, `if (flag)` conditions, and printing of logical variables.
- The direct LIRIC session lowerer can compile simple contained integer
  functions and subroutines with integer parameters and integer call
  expressions/statements.
- The direct LIRIC session path emits native executables and object files.
- `ffc empty.f90 -o empty` emits a native executable; `ffc empty.f90 -c -o
  empty.o` emits an object file.
- The bootstrap LIRIC compiler API path still has broader scalar coverage
  through generated text IR. It is kept as temporary executable reference
  coverage while the direct session path catches up.
- Broader runtime calls, richer procedure signatures, arrays, modules, and
  richer I/O are still pending.

## Target Architecture

```
Fortran / Lazy Fortran source
        |
        v
FortFront typed AST + diagnostics
        |
        v
ffc lowering and runtime ABI
        |
        v
LIRIC C API
        |
        v
object file / executable
```

FortFront should remain backend-neutral. `ffc` owns lowering, ABI decisions,
runtime calls, backend selection, object emission, and executable emission.

## Backend Direction

The preferred backend path is LIRIC through ISO C bindings to its C API.

Two implementation levels are expected:

1. **Direct path**: lower typed AST to LIRIC `lr_session_*` calls. This is the
   CLI path and target architecture; new lowering work should go here.
2. **Bootstrap reference path**: lower a small typed-AST subset to textual
   low-level IR accepted by LIRIC's compiler API. This is temporary
   compatibility coverage, not the target architecture.

The current MLIR binding work should not be expanded unless the project
explicitly changes direction back to a Flang-style backend.

## MVP Scope

The first useful compiler should support:

- `program main`
- scalar `integer` literals
- scalar integer declarations and assignments
- integer arithmetic
- minimal `print *, expr`
- one-line `if` with integer comparisons
- counted `do` loops with integer bounds and literal integer step
- scalar `real` literals in `print`
- simple `character` literals in `print`
- scalar `logical` literals in `print`
- real variables/arithmetic
- block `if`
- simple functions and subroutines
- object/executable emission through LIRIC

Arrays, allocatables, modules, derived types, full I/O, generics, and
cross-module inference come after that subset is executable and tested.

## Build

Builds need the LIRIC static library on the linker search path:

```bash
LIBRARY_PATH=/home/ert/code/liric/build fpm build
```

Run the MVP tests:

```bash
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_liric_bindings
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
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_empty_program_compiler
```

Compile the smallest supported program:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=/home/ert/code/liric/build fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
```

## Related Repositories

- [fortfront](https://github.com/lazy-fortran/fortfront): frontend,
  transformation, typed AST work.
- [standard](https://github.com/lazy-fortran/standard): intended LFortran
  Standard and Infer behavior.
- [liric](https://github.com/lazy-fortran/liric): backend C API target.

## Status Source

See [ROADMAP.md](ROADMAP.md) for the active plan. The `docs/` directory now
describes the FortFront-to-LIRIC path.

The current direct-session MVP ABI is documented in
[docs/RUNTIME_ABI.md](docs/RUNTIME_ABI.md).
