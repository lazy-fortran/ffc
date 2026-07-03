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
items and an inline array constructor as a print item
(`print *, [e1, e2, ...]`, each explicit numeric element printed like a scalar);
fixed-size arrays of rank 1 through 7 with
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
sections, whole-array copy, elemental arithmetic (`+`, `-`, `*`, `/`, `**`,
unary minus, an array constructor as an rvalue operand, and general scalar
expressions broadcasting to every element), and the array intrinsics
`size`, `shape`, `sum`, `product`, `maxval`, `minval`, `dot_product`,
`matmul`, `transpose`, `reshape`, `lbound`, `ubound`, `count`, `any`, `all`,
and rank-1 scalar `maxloc`/`minloc` (optional `dim=1` and `mask`); scalar
element read and write on an allocated 1-D integer, real, or
logical allocatable (`a(i)`); whole-array assignment from an array constructor
to a 1-D allocatable with auto-reallocation (`a = [e1, e2, ...]`); whole-array
assignment from a general elementwise expression to a 1-D allocatable already
allocated to a compile-time-constant extent (`a = b + c`, `a = a * 3`); and
whole-array `print` of a 1-D allocatable whose extent is a compile-time
constant; and assumed-shape dummies (`a(:)`, `a(:,:)`) bound to the actual's
base address, with their extent taken from a whole-array actual of
compile-time size (including a `dimension(n)` bound naming a caller-scope
`parameter`), so element read/write, `size(a)`, `size(a, dim)`,
`lbound`/`ubound`, `sum`, and whole-array `print` work in the callee, in both
program-contained and module procedures. A rank-1 assumed-shape dummy of a
subroutine also accepts a rank-1 allocatable actual of runtime-only extent:
the extent travels as a hidden argument, so `size(a)`, `ubound(a, 1)`,
element read/write, a `do` loop bound by `size(a)`, and integer `sum(a)`
all work against the caller's runtime allocation. A scalar
`integer`/`real`/`logical, allocatable` variable
supports `allocate`/`deallocate` and `allocate(x, source=<expr>)`/`mold=<expr>`
with any source expression; a rank-1/rank-2 allocatable array dummy argument
aliases the caller's own descriptor, so `allocate`/`deallocate`/element writes
inside the callee are visible to the caller. Scalar
integer `pointer`/`target` with `p => t`, read/write through `p`,
`associated(p)`, and `nullify(p)` is supported, as are constant-folded
`selected_int_kind` and `selected_real_kind`. Compile-time integer folding
in array-bound and `parameter` initializer positions also covers `kind`,
`size`, `len`, `min`, `max`, `int`, `huge`, `bit_size`, `precision`, `range`,
`selected_char_kind`, and a bare `iso_c_binding` kind name used as a value
(`c_bool`, `c_int`, `c_long`, ...). Non-default integer kinds
`integer(1)`/`(2)`/`(8)` (and their `iso_c_binding` C-interop kind names, incl.
`c_size_t`/`c_intptr_t`/`c_ptrdiff_t`/`c_intmax_t`) support arithmetic,
comparison, and `print`, and fixed-size rank-1/rank-2 arrays of these kinds
support element assignment, element reads, scalar broadcast, and whole-array
`print`; `real(8)` recognizes the `dp`/`wp` kind alias
convention and resolves a literal kind suffix naming any other declared
`integer, parameter` kind constant to its folded value. `real()`/`dble()`
applied to a BOZ-literal-constant argument reinterpret its bit pattern as
the result kind rather than converting the magnitude. Scalar `complex`/`complex(8)` support `+`/`-`/`*`/`/` arithmetic,
including a mixed real/complex operand, and fixed-size rank-1/rank-2 complex
arrays support element assignment, element reads, elemental `+`/`-`/`*`/`/`
between array elements, and single-element `print`. An `if` condition accepts any
scalar logical expression: `.not.`/`.and.`/`.or.`/`.eqv.`/`.neqv.` trees, a
logical array element, a derived-type logical component, `allocated(a)`, and
a contained logical function's result, including the one-line
`if (cond) stmt` form without `then`. Derived types take scalar
integer, real, logical, and `c_ptr` components, fixed-size rank-1 integer,
real, and logical array components (`real :: r(N)`, accessed as `x%r(i)`),
and scalar nested derived components (`type(inner) :: c`, accessed
as `x%c%field` to any depth), and support single inheritance
(`type, extends(parent) :: child`) with parent-first component layout. A
whole-derived scalar assignment (`y = x`) copies one instance into another,
and a scalar structure constructor over integer/real/logical components
(`x = t(1, 2.5, .true.)`, omitted components keeping their defaults) stores
its positional arguments into the target, whether written as an executable
assignment or as a scalar variable initializer (`type(t) :: v = t(1, 2.5)`).
Integer, default-real (f32), and logical component default initialisers
materialise on default-initialised instances. A
contained function may return a fixed-size rank-1 array: the result lowers
through the sret ABI (the caller passes the destination buffer as a hidden
result pointer), so `r = vec_fn(...)` and `print *, vec_fn(...)` write the
result straight into the destination. A module procedure may `contains` internal procedures, lowered as
flat functions. A `logical`-valued function call (contained or module) prints
directly as `T`/`F`, and a module function with a deferred-length
(`character(len=:)`) or runtime-length (`character(len=len(arg))`) character
result is callable and printable from a program in the same file. Module-level
integer, real, and logical scalar variables persist as globals and are visible
across `use`, in the same file or across separate compilation. A module
subroutine or integer function with integer, real, or logical scalar arguments
is callable from a separately compiled program: its signature round-trips
through the `.fmod` and the two objects link, each keeping its own string and
format literals (#284). A module-scope
scalar variable of a registered derived type is likewise a flat slot global, its
compile-time component defaults folded into the static bytes and read through
`use` (including a `use ..., alias => var` rename). A file whose
top-level units are one or more modules with no main program is a valid
translation unit: it lowers to a no-op main, so it compiles to an object with
`-c` (each module's procedures under their mangled symbols) and links to an
empty executable, matching gfortran's own module-only object. A single-file A single-file
`submodule (m) s` implements the module procedures its parent module `m`
declares through interface bodies; both the restated signature form and the
separate `module procedure` form lower under the parent's mangled symbol, so a
`use m` call resolves regardless of which submodule holds the body. A parent
generic interface whose specific is a module-procedure interface body dispatches
a call through the generic name to the submodule body implementing that specific.
The `associate` construct binds scalar selectors. The
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
the declared length, and so does assigning a plain character variable or
intrinsic result to a fixed-length target. `==`, `/=`, `<`, `<=`, `>`, `>=`
between character operands use Fortran's blank-padded lexical ordering, and a
character `SELECT CASE` accepts a lexical range label (`case ('a':'j')`). A
fixed-length dummy (`character(len=N), intent(in)`) keeps its own declared
width rather than the caller's runtime length. A character `parameter` named
constant may declare a fixed length, padded or truncated from its folded
initializer, and that initializer may concatenate an earlier character named
constant. The real transcendental intrinsics lower to libm: `sqrt`, `exp`,
`log`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `gamma`,
`log_gamma`, and `hypot`. `open`/`close`/`rewind` map to `fopen`/`fclose`/
`rewind`, preserving an existing file's content when `status=` is omitted;
a file unit's list-directed and numeric-edit-descriptor `read` covers
integer, real, and fixed-length character scalars. Internal `read (buf, *)
value` (list-directed) and `write (buf, fmt) value` with a compound literal
format (`I`/`A` descriptors) are supported. `inquire` covers `exist=`,
`opened=`, and `iostat=` on `file=` and `unit=`. Invalid programs are
rejected during lowering: an integer `SELECT CASE` with overlapping
integer-literal CASE labels, and a character-valued I/O specifier
(`STATUS=`, `ACCESS=`, `ADVANCE=`, `IOMSG=`, ...) handed a numeric or
logical literal (`status=1`, `advance=5.`), both fail with a diagnostic.

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
