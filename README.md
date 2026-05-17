# ffc

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style source.

The active build is the FortFront + direct LIRIC session compiler path. The
older MLIR/HLFIR experiment remains in `src/` as legacy reference code, but it
is not part of the default fpm build.

## Current Status

- The package builds the active compiler sources in `src_mvp/`.
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
  `print *, expr`, real literal `print`, character literal `print`,
  character variable `print`, and logical literal `print` through external
  `printf` calls.
- The direct LIRIC session lowerer can compile scalar `real` declarations,
  assignments, arithmetic, and printing of real variables.
- The direct LIRIC session lowerer can compile scalar `logical` declarations,
  assignments, `if (flag)` conditions, and printing of logical variables.
- The direct LIRIC session lowerer can compile simple contained integer, real,
  and logical functions and subroutines with scalar parameters and scalar call
  expressions/statements. Procedure arguments use pointer parameters with
  copy-back for variable actual arguments.
- The direct LIRIC session lowerer can compile scalar `abs`, `min`, and
  `max` intrinsics for integer and real values, plus integer-to-real `real()`
  conversion, inline through LIRIC scalar operations, blocks, and PHI values.
- The direct LIRIC session lowerer can compile fixed-size one-dimensional
  integer arrays with compile-time integer bounds, including scalar integer
  parameters and explicit lower:upper bounds. Element assignment, element
  reads, `print`, `stop`, and counted-loop subscripts are supported. Runtime
  bounds checks are not emitted; out-of-bounds subscripts have backend-level
  behavior until #53 defines array descriptors and checks.
- The direct LIRIC session path emits native executables and object files.
- `ffc empty.f90 -o empty` emits a native executable; `ffc empty.f90 -c -o
  empty.o` emits an object file.
- Allocatable arrays, multidimensional arrays, non-integer arrays, and modules
  are unsupported. Broader runtime calls and richer I/O are unsupported. The
  tracked work is listed in
  [docs/SUPPORT_CONTRACT.md](docs/SUPPORT_CONTRACT.md).

## Support Contract

A program is supported only when every construct it uses appears in
[docs/SUPPORT_CONTRACT.md](docs/SUPPORT_CONTRACT.md). Anything else must fail
with a diagnostic instead of being partially lowered.

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

FortFront remains backend-neutral. `ffc` owns lowering, ABI decisions,
runtime calls, backend selection, object emission, and executable emission.

## Backend Direction

The preferred backend path is LIRIC through ISO C bindings to its C API.

Lower typed AST to LIRIC `lr_session_*` calls. This is the CLI path and target
architecture; new lowering work goes here.

Do not expand the current MLIR binding work unless the project explicitly
changes direction back to a Flang-style backend.

## Current MVP Scope

The current MVP support claim is:

- `program main`
- scalar `integer` literals
- scalar integer declarations and assignments
- integer arithmetic
- minimal `print *, expr`
- one-line `if` with integer comparisons
- counted `do` loops with integer bounds and literal integer step
- scalar `real` literals in `print`
- simple `character` literals in `print`
- scalar `character(len=N)` variables assigned from literals and printed
- scalar `logical` literals in `print`
- real variables/arithmetic
- block `if`
- simple contained integer, real, and logical functions and subroutines
- integer and real scalar `abs`, `min`, and `max` intrinsics
- integer-to-real `real()` conversion
- fixed-size one-dimensional integer arrays with compile-time integer bounds
- object/executable emission through LIRIC

Allocatable arrays, multidimensional arrays, non-integer arrays, modules,
derived types, full I/O, generics, cross-module inference, and character
procedure arguments are unsupported today. See the issue map in
[docs/SUPPORT_CONTRACT.md](docs/SUPPORT_CONTRACT.md).

## Build

Builds need the LIRIC static library on the linker search path:

```bash
LIBRARY_PATH=<liric-build> fpm build
```

Run the MVP tests:

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
LIBRARY_PATH=<liric-build> fpm test test_session_non_integer_procedure_compiler
LIBRARY_PATH=<liric-build> fpm test test_session_integer_intrinsic_compiler
```

Compile the smallest supported program:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=<liric-build> fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
```

## Related Repositories

- [fortfront](https://github.com/lazy-fortran/fortfront): frontend,
  transformation, typed AST work.
- [standard](https://github.com/lazy-fortran/standard): intended LFortran
  Standard and Infer behavior.
- [liric](https://github.com/lazy-fortran/liric): backend C API target.

## Status Source

The repository plan file tracks active work. The `docs/` directory now describes
the FortFront-to-LIRIC path.

The current direct-session MVP ABI is documented in
[docs/RUNTIME_ABI.md](docs/RUNTIME_ABI.md).
