# ffc

`ffc` is the planned compiler driver for Lazy Fortran and LFortran Infer-style
source. It is not currently a working compiler.

The existing tree contains an unfinished MLIR/HLFIR experiment. That code is
useful as historical scaffolding, but the command path does not yet parse input
through FortFront and the object/executable path is not implemented.

## Current Status

- The package builds against `fortfront` and contains MLIR/LLVM-oriented
  modules.
- The CLI still has a source-file compilation stub.
- MLIR lowering/object/executable emission is explicitly unfinished.
- Several end-to-end MLIR tests are disabled.
- The active direction is to make `ffc` a small, real compiler driver using
  FortFront for the frontend and LIRIC for backend code generation.

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
- scalar `integer`, `real`, `logical`, and simple `character` literals
- scalar declarations and assignments
- arithmetic and comparisons
- `if` and counted `do`
- simple functions and subroutines
- `print` through a tiny runtime shim
- object/executable emission through LIRIC

Arrays, allocatables, modules, derived types, full I/O, generics, and
cross-module inference come after that subset is executable and tested.

## Build

```bash
fpm build
```

The current manifest still links MLIR/LLVM libraries because the old backend is
present. The LIRIC backend work should remove that requirement from the default
build path.

## Related Repositories

- [fortfront](https://github.com/lazy-fortran/fortfront): frontend,
  transformation, typed AST work.
- [standard](https://github.com/lazy-fortran/standard): intended LFortran
  Standard and Infer behavior.
- [liric](https://github.com/lazy-fortran/liric): backend C API target.

## Status Source

See [ROADMAP.md](ROADMAP.md) for the active plan. The MLIR API documents under
`docs/` are legacy references until the backend direction changes.
