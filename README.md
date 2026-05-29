# ffc

`ffc` is the compiler driver for Lazy Fortran and LFortran Infer-style
source. It compiles supported Fortran programs to native object files and
executables via FortFront's typed AST and LIRIC's session C API.

```
Fortran / Lazy Fortran source
  -> FortFront typed AST + diagnostics
  -> ffc lowering + runtime ABI
  -> LIRIC C API (via ISO_C_BINDING)
  -> object file / executable
```

FortFront stays backend-neutral. `ffc` owns lowering, ABI, runtime calls,
LIRIC bindings, and object/exe emission. The retired MLIR/HLFIR
experiment lives only in git history.

## Current support claim

The public contract is `docs/SUPPORT_CONTRACT.md`. Today's surface:

- `program main`, scalar `integer` / `real` / `logical`, character
  literals, fixed-length `character(len=N)` variables;
- deferred-length `character(len=:), allocatable` including
  self-aliasing and three-way `//` concatenation; compile-time `//`
  folding for character literal chains;
- arithmetic, comparisons, scalar logical conditions;
- block `if` with PHI merge for scalars; array-element assignment inside
  `if` branches; counted `do` with integer bounds and literal positive
  or negative step;
- single-arm and multi-arm `SELECT CASE` with terminating or
  non-terminating (merge-at-end) arms
  (including multi-label `case (a, b)`) and `case default`;
- contained integer / real / logical functions and subroutines with
  scalar parameters; early `return` inside contained subroutines and
  functions; integer procedure arguments use pointer parameters with
  copy-back for variable actuals;
- fixed-size 1-D integer arrays with compile-time bounds; element
  assignment, element reads, `print`, `stop`, counted-loop
  subscripts, and whole-array assignment from an array constructor
  (`a = [e1, e2, ...]`);
- simple derived types with scalar integer components; component
  assignment, component reads, `print`, and `stop`;
- `print *, expr` (list-directed, gfortran-exact bytes) for integers,
  reals, logicals, characters, plus formatted `print '(I0)'/'(Iw)'/'(A)',
  expr` with a single edit descriptor; `stop <integer expression>` returns
  its argument
  as the process exit status; `abs`, `min`, `max`, `mod`, `iand`, `ior`,
  `ieor`, `not`, `ishft`, `ishftc`, `sign`, integer-to-real `real()`,
  real `**` via libm `pow`, real `sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/
  `atan`/`atan2` via libm, real-to-integer `int`/`nint`/`floor`/`ceiling`,
  and `command_argument_count`/`get_command_argument` in the main program;
- CLI: `-o <file>`, `-c`, `-I <dir>` (`-I` accepted and stored, not yet
  consumed by lowering).

Allocatable arrays, multidimensional arrays, modules and separate
compilation, polymorphism, full runtime I/O, generics, and most
intrinsics are unsupported today and tracked as GitHub issues. The
self-hosting dependency map is in
[#167](https://github.com/lazy-fortran/ffc/issues/167).

## Build

The LIRIC static library must be on the linker path:

```bash
cd ../liric && cmake -S . -B build -G Ninja && cmake --build build
cd ../ffc
export LIBRARY_PATH=../liric/build
fpm build
fpm test
```

`fpm build` produces the `ffc` binary; `fpm test` runs the behavioural
test suite (currently 40 programs). CI runs the same workflow on every
push and pull request.

Compile a minimal program:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=../liric/build fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
echo $?
```

## Layout

- `app/ffc.f90` — CLI entry.
- `src/` — lowering, LIRIC bindings, CLI options (fpm auto-discovers).
- `test/` — behavioural tests; each file is a standalone `program test_*`
  picked up by fpm auto-discovery.
- `docs/` — `SUPPORT_CONTRACT.md`, `RUNTIME_ABI.md`, `DEVELOPER_GUIDE.md`,
  `API_REFERENCE.md`, `C_API_USAGE.md`, `MIGRATION_GUIDE.md`.
- `ROADMAP.md`, `BACKLOG.md`, `DESIGN.md` — high-level direction.

## Conventions

- Free-form Fortran 2003+; no implicit typing; declarations at scope top.
- Modules under 500 lines (hard cap 1000); functions under 50 lines
  (hard cap 100). Split into `*.inc` files (already used heavily) when
  the lowerer grows.
- Symbols `snake_case`, derived types end in `_t`.
- Each new supported construct lands as a code change in
  `src/session_program_lowering_*` plus a behavioural test under `test/`.
  Update `README.md` and `docs/SUPPORT_CONTRACT.md` in the same commit.

## Related repositories

- [fortfront](https://github.com/lazy-fortran/fortfront) — frontend,
  transformation, typed AST.
- [liric](https://github.com/lazy-fortran/liric) — backend C API target.
- [standard](https://github.com/lazy-fortran/standard) — intended
  LFortran Standard and Infer behaviour.
