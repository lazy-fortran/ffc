# ffc

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style source.

The active build is the FortFront + LIRIC bootstrap compiler path. The older
MLIR/HLFIR experiment remains in `src/` and `test/` as legacy reference code,
but it is not part of the default fpm build.

## Current Status

- The package builds the MVP sources in `src_mvp/`.
- The CLI parses files through FortFront's compiler-facing frontend API.
- The bootstrap backend lowers a small integer scalar subset to LLVM IR text and
  feeds it to LIRIC through ISO C bindings.
- `ffc empty.f90 -o empty` emits a native executable for:
  `program main; end program main`.
- Integer declarations, assignment, `+ - * /` arithmetic, and `print *, expr`
  are implemented for straight-line programs.
- One-line integer comparison `if` statements are implemented.
- Real/logical/character lowering, block control flow, counted loops,
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

1. **Bootstrap path**: lower a small typed-AST subset to LLVM IR text and feed
   it to `lr_compiler_feed_ll()`. This avoids LLVM bindings while proving
   executable output quickly.
2. **Direct path**: lower typed AST to LIRIC `lr_session_*` calls for better
   performance and less text generation.

The current MLIR/LLVM binding work should not be expanded unless the project
explicitly changes direction back to a Flang/MLIR backend.

## MVP Scope

The first useful compiler should support:

- `program main`
- scalar `integer` literals
- scalar integer declarations and assignments
- integer arithmetic
- minimal `print *, expr`
- one-line `if` with integer comparisons
- scalar `real`, `logical`, and simple `character` literals
- block `if` and counted `do`
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

See [ROADMAP.md](ROADMAP.md) for the active plan. The MLIR API documents under
`docs/` are legacy references until the backend direction changes.
