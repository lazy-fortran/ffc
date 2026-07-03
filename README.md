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

## Supported features

The public contract is `docs/SUPPORT_CONTRACT.md`. It lists every
supported construct, its ABI, and every tracked gap with issue links.
Refer to that document instead of this README for the feature list.
Current slices include compound formatted `print` with literal `I`, `X`,
`F`, and `A` descriptors on stdout, including a bare array among other print
items; fixed-size arrays of rank 1 through 7 with
`integer`, `real`, `real(8)`, and `logical` elements (rank 3 and above cover
declaration, `a(i, j, k, ...)` element read/write, scalar broadcast,
whole-array copy, elemental `+`/`-`/`*`, whole-array `print`, `lbound`,
`ubound`, `size(a[, dim])`, and `sum`; sections, `matmul`, `transpose`, and
`reshape` stay rank-1/rank-2); fixed-size rank-1 and
rank-2 `character(len=N)` arrays (character-literal element assignment, element
`print`, and whole-array `print`); array constructors as
whole-array assignment right-hand sides, plain (`[a, b, c]`), typed
(`[integer :: 1, 2]`, real-to-integer truncation and integer-to-real
promotion), and integer/real implied-do (`[(i*i, i=1, n)]`); scalar element
access, array
sections, whole-array copy, elemental arithmetic, and the array intrinsics
`size`, `shape`, `sum`, `product`, `maxval`, `minval`, `dot_product`,
`matmul`, `transpose`, `reshape`, `lbound`, `ubound`, `count`, `any`, `all`,
and rank-1 scalar `maxloc`/`minloc` (optional `dim=1` and `mask`); scalar
element read and write on an allocated 1-D integer, real, or
logical allocatable (`a(i)`); whole-array assignment from an array constructor
to a 1-D allocatable with auto-reallocation (`a = [e1, e2, ...]`); and
whole-array `print` of a 1-D allocatable whose extent is a compile-time
constant; and assumed-shape dummies (`a(:)`, `a(:,:)`) bound to the actual's
base address, with their extent taken from a whole-array actual of
compile-time size, so `size`, element access, `sum`, and whole-array `print`
work in the callee. Scalar
integer `pointer`/`target` with `p => t`, read/write through `p`,
`associated(p)`, and `nullify(p)` is supported, as are constant-folded
`selected_int_kind` and `selected_real_kind`. Compile-time integer folding
in array-bound and `parameter` initializer positions also covers `kind`,
`size`, `len`, `min`, `max`, `int`, `huge`, `bit_size`, `precision`, `range`,
`selected_char_kind`, and a bare `iso_c_binding` kind name used as a value
(`c_bool`, `c_int`, `c_long`, ...). Derived types take scalar
integer, real, logical, and `c_ptr` components, fixed-size rank-1 integer,
real, and logical array components (`real :: r(N)`, accessed as `x%r(i)`),
and scalar nested derived components (`type(inner) :: c`, accessed
as `x%c%field` to any depth), and support single inheritance
(`type, extends(parent) :: child`) with parent-first component layout. A
whole-derived scalar assignment (`y = x`) copies one instance into another,
and a scalar structure constructor over integer/real/logical components
(`x = t(1, 2.5, .true.)`, omitted components keeping their defaults) stores
its positional arguments into the target. A
contained function may return a fixed-size rank-1 array: the result lowers
through the sret ABI (the caller passes the destination buffer as a hidden
result pointer), so `r = vec_fn(...)` and `print *, vec_fn(...)` write the
result straight into the destination. A module procedure may `contains` internal procedures, lowered as
flat functions. Module-level integer variables persist as globals and are
visible across `use`. The `associate` construct binds scalar selectors. The
`where` construct masks elementwise assignment over rank-1 integer and real
arrays, including a final `elsewhere`. The `forall` construct, single-statement
and block form, lowers to a sequential loop nest over its index set, with an
optional scalar-comparison mask guarding the body. The
scalar numeric intrinsics `mod`, `modulo`, `sign`, `dim`, `int`, `nint`,
`floor`, `ceiling`, `real`, `dble` are supported, as are the bit intrinsics
`iand`, `ior`, `ieor`, `not`, `ishft`, `ishftc`, `ibits`, `btest`, and
`mvbits`. The character intrinsics `len`, `len_trim`, `trim`, `adjustl`,
`adjustr`, `index`, `scan`, `verify`, `repeat`, `achar`, and `iachar` are
supported. A `//` concatenation of character variables, literals, and these
intrinsics assigns into a fixed-length scalar, truncating or blank-padding to
the declared length. The real transcendental intrinsics lower to libm: `sqrt`, `exp`,
`log`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `gamma`,
`log_gamma`, and `hypot`.

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
test suite.

Compile a minimal program:

```bash
printf 'program main\nend program main\n' > /tmp/empty.f90
LIBRARY_PATH=../liric/build fpm run ffc -- /tmp/empty.f90 -o /tmp/empty
/tmp/empty
echo $?
```

## Conformance

`docs/CONFORMANCE.md` documents the conformance gauntlet runner that
drives external Fortran test corpora through the full `ffc` pipeline.
It produces an xfail-style report with JSONL output.

## Layout

- `app/ffc.f90` - CLI entry.
- `src/` - lowering, LIRIC bindings, CLI options (fpm auto-discovers).
- `test/` - behavioural tests; each file is a standalone `program test_*`
  picked up by fpm auto-discovery.
- `docs/` - `SUPPORT_CONTRACT.md`, `RUNTIME_ABI.md`, `DEVELOPER_GUIDE.md`,
  `API_REFERENCE.md`, `C_API_USAGE.md`, `MIGRATION_GUIDE.md`,
  `CONFORMANCE.md`.
- `BACKLOG.md`, `DESIGN.md` - planning docs.

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

- [fortfront](https://github.com/lazy-fortran/fortfront) - frontend,
  transformation, typed AST.
- [liric](https://github.com/lazy-fortran/liric) - backend C API target.
- [standard](https://github.com/lazy-fortran/standard) - intended
  LFortran Standard and Infer behaviour.
