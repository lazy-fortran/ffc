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
- The direct LIRIC session lowerer can compile literal-bound counted `do`
  loops by expanding their scalar body through direct LIRIC operations.
- The direct LIRIC session lowerer can compile minimal integer
  `print *, expr` through an external `printf` call.
- `ffc empty.f90 -o empty` emits a native executable for:
  `program main; end program main`.
- The bootstrap LIRIC compiler API path still has broader scalar coverage
  through generated text IR. It is kept as temporary executable reference
  coverage while the direct session path catches up.
- Runtime counted `do` blocks with PHI backedges, broader runtime calls,
  procedures, and richer I/O are still pending.

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
- counted `do` loops with literal integer bounds and step
- scalar `real` literals in `print`
- simple `character` literals in `print`
- scalar `logical` literals in `print`
- real variables/arithmetic
- block `if`
- dynamic counted `do`
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
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_stop_code_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_integer_variable_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_block_if_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_if_merge_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_counted_do_compiler
LIBRARY_PATH=/home/ert/code/liric/build fpm test test_session_scalar_print_compiler
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
