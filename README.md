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
`print`, whole-array `print`, element comparison in `if` conditions such as
`if (arr(i) /= "A")`, and compile-time initializers from a literal array
constructor `character(len=4) :: s(4) = ["sngl", "dble", ...]` or a scalar
broadcast `character(len=3) :: p(3) = "xy"`, including `parameter` arrays); array constructors as
whole-array assignment right-hand sides, plain (`[a, b, c]`), typed
(`[integer :: 1, 2]`, real-to-integer truncation and integer-to-real
promotion), and integer/real implied-do (`[(i*i, i=1, n)]`); scalar element
access, array
sections, whole-array copy, elemental arithmetic (`+`, `-`, `*`, `/`, `**`,
unary minus, an array constructor as an rvalue operand, general scalar
expressions broadcasting to every element, elemental `max`/`min` of two
conforming arrays, a relational comparison between two conforming
arrays assigned to a logical array, e.g. `mask = a > b`, and whole-array
`.not.` of a logical array or logical-array expression, e.g.
`a = .not. (b > c)`), and the array intrinsics
`size`, `shape`, `sum`, `product`, `maxval`, `minval`, `dot_product`,
`matmul`, `transpose`, `reshape` (also as a declaration initializer, with
integer-source-to-real conversion, and with a `shape(X)` shape argument or
keyword `source=`/`shape=` form), `lbound`, `ubound`, `count`, `any`, `all`
(including a whole-array relational comparison of two conforming integer, real,
or allocatable arrays, an array against a scalar, or an array against an array
constructor, e.g. `if (any(a /= b))`, `all(a == [1, 2, 3])`, plus dim-wise
`sum`/`product`/`count`/`any`/`all` of a rank-2 source into a rank-1 target
with a compile-time `dim`, e.g. `s = sum(a, 1)`, `m = any(a == b, 2)`),
and rank-1 scalar `maxloc`/`minloc` (optional `dim=1` and `mask`); scalar
element read and write on an allocated 1-D or 2-D integer, real, or
logical allocatable (`a(i)`, `a(i,j)`); whole-array assignment from an array
constructor to a 1-D allocatable with auto-reallocation (`a = [e1, e2, ...]`);
whole-array assignment from a general elementwise expression to a 1-D
allocatable already allocated to a compile-time-constant extent
(`a = b + c`, `a = a * 3`); whole-array `print`, `size(a)`, and integer
`sum(a)` of a 1-D allocatable whether its extent is a compile-time constant or
a runtime value (`allocate(a(n))` with `n` a variable), reading a runtime extent
back from the descriptor, including `T`/`F` formatting for a logical
allocatable element or whole array; a fixed-length rank-1
`character(len=N), allocatable :: a(:)` with `allocate(a(N))`
blank-filling every slot, element write/read (`a(i) = "text"`), and
whole-array `print`; a deferred-length rank-1
`character(len=:), allocatable :: c(:)` whose element length is fixed by a
compile-time `allocate(character(len=N) :: c(M))` and then behaves like the
fixed-length allocatable array (element write/read, comparison, `len`, `size`,
whole-array `print`, `allocated`, `deallocate`); and assumed-shape dummies (`a(:)`, `a(:,:)`) bound to the actual's
base address, with their extent taken from a whole-array actual of
compile-time size (including a `dimension(n)` bound naming a caller-scope
`parameter`), so element read/write, `size(a)`, `size(a, dim)`,
`lbound`/`ubound`, `sum`, and whole-array `print` work in the callee, in both
program-contained and module procedures. The actual may also be a contiguous
rank-1 array section with compile-time bounds -- a stride-1 slice `a(2:4)` or a
whole column `m(:,j)` (integer, `real`, `real(8)`) -- whose extent folds from
the section and whose first-element address binds the dummy in place. A rank-1
assumed-shape dummy of a
subroutine also accepts a rank-1 allocatable actual of runtime-only extent:
the extent travels as a hidden argument, so `size(a)`, `ubound(a, 1)`,
element read/write, a `do` loop bound by `size(a)`, and integer `sum(a)`
all work against the caller's runtime allocation. An assumed-size dummy
(`a(*)`, `a(n1, ..., *)`, dummy arguments only) folds its leading dimensions
at compile time and binds to the actual's base address, so element read/write
and `lbound(a, dim)` work; the trailing dimension carries no extent, so
`ubound`/`size` on it and whole-array operations are not supported. A named
constant declared `a(*) = [...]` (single dimension, a non-implied-do array
constructor initializer) instead takes its extent from the initializer's
element count. A scalar
`integer`/`real`/`logical, allocatable` variable
supports `allocate`/`deallocate` and `allocate(x, source=<expr>)`/`mold=<expr>`
with any source expression; a rank-1/rank-2 allocatable array dummy argument
aliases the caller's own descriptor, so `allocate`/`deallocate`/element writes
inside the callee are visible to the caller. A single `allocate` or
`deallocate` may list several targets at once
(`allocate(a(N), b(N), c(M,K), stat=ierr)`, `deallocate(a, b, c)`); each target
is sized from its own subscripts and an optional `stat=` integer is set to 0 on
success. A rank-2 allocatable also supports whole-array scalar broadcast
(`m = 9`) once allocated to a compile-time-constant shape. Scalar
integer `pointer`/`target` with `p => t`, read/write through `p`,
`associated(p)`, and `nullify(p)` is supported. Rank-1 and rank-2 fixed-size
integer, real, logical, and complex `pointer`/`target` arrays support whole-
array `p => t` aliasing, so element read/write, `lbound`/`ubound`, and
`print` through `p` reach `t`'s storage, as are constant-folded
`selected_int_kind` and `selected_real_kind`. Compile-time integer folding
in array-bound and `parameter` initializer positions also covers `kind`,
`size`, `len`, `min`, `max`, `int`, `huge`, `bit_size`, `precision`, `range`,
`digits`, `radix`, `minexponent`, `maxexponent`, `selected_logical_kind`,
`selected_char_kind`, `mod`, `modulo`, `sign`, `dim`, `abs`, the bit
intrinsics `iand`/`ior`/`ieor`/`xor`/`not`/`ishft`/`ishftc`/`ibits`/`ibset`/
`ibclr`, comparison and `.and.`/`.or.` operators folding to 1/0, `merge` on
a folded mask, `product`/`sum`/`maxval`/`minval`/`dot_product` over an array
constructor or a named integer `parameter` array (itself indexable by a
compile-time constant), and a bare `iso_c_binding` kind name used as a value
(`c_bool`, `c_int`, `c_long`, ...). Non-default integer kinds
`integer(1)`/`(2)`/`(8)` (and their `iso_c_binding` C-interop kind names, incl.
`c_size_t`/`c_intptr_t`/`c_ptrdiff_t`/`c_intmax_t`) support arithmetic,
comparison, and `print`, and fixed-size rank-1/rank-2 arrays of these kinds
support element assignment, element reads, scalar broadcast, and whole-array
`print`; `real(8)` recognizes the `dp`/`wp` kind alias
convention and resolves a literal kind suffix naming any other declared
`integer, parameter` kind constant to its folded value. Declaration-side
`real(prec)`/`integer(prec)`/`complex(prec)` kind specs resolve the same
declared-parameter names, not just `dp`/`wp`. `real()`/`dble()`
applied to a BOZ-literal-constant argument reinterpret its bit pattern as
the result kind rather than converting the magnitude. Scalar `complex`/`complex(8)` support `+`/`-`/`*`/`/` arithmetic,
including a mixed real/complex operand, `cmplx()` (single-argument or with a
keyword/positional kind selector), `dcmplx()`, `real()`/`aimag()` component
extraction (`real(z, kind)` accepts a kind selector), `conjg()`/`dconjg()`, and
`abs()` (real magnitude via libm `hypot`); `complex(dp)`/`complex(wp)` resolve
the double-precision kind aliases. A `cmplx()` component or complex-assignment
operand may be an integer or a mismatched-precision real (widened or narrowed to
the target component); an integer/real scalar assigns to a complex as the real
part with zero imaginary part, and `complex(4)`/`complex(8)` assignments convert
across kinds. Fixed-size rank-1/rank-2 complex
arrays support element assignment, element reads, elemental `+`/`-`/`*`/`/`
between array elements, single-element `print`, and whole-array assignment
with elementwise `+`/`-`/`*`/`/`, whole-array copy, or scalar broadcast
(`c = a + b`, `c = a`, `c = (1.0, 2.0)`) between conforming complex arrays.
An `if` condition accepts any
scalar logical expression: `.not.`/`.and.`/`.or.`/`.eqv.`/`.neqv.` trees, a
logical array element, a derived-type logical component, `allocated(a)`, and
a contained logical function's result, including the one-line
`if (cond) stmt` form without `then`. Derived types take scalar
integer, real, logical, `c_ptr`, and fixed-length character
(`character(len=N)`) components, fixed-size rank-1 integer,
real, and logical array components (`real :: r(N)`, accessed as `x%r(i)`,
with whole-component assignment from an array constructor (`x%v = [1, 2, 3]`),
scalar broadcast (`x%v = 7`), whole-component copy (`y%v = x%v`), and reading
the whole component into a conforming plain array (`a = x%v`)),
and scalar nested derived components (`type(inner) :: c`, accessed
as `x%c%field` to any depth), and fixed-size arrays of derived
components (`type(inner) :: arr(N)`, an element and its fields reached by
subscript chain `x%arr(i)%field`, including deep chains
`obj%w(1)%z(2)%y(3)%leaf` with an allocatable-array leaf and per-element
default/descriptor initialisation), and support single inheritance
(`type, extends(parent) :: child`) with parent-first component layout. A
fixed-length character component supports reading, writing (blank-padded and
truncated to its declared length), comparison, concatenation, `print`, and
passing as an actual argument to a `character(len=*)` dummy, through
`x%name`. A
whole-derived scalar assignment (`y = x`) copies one instance into another,
and a scalar structure constructor over integer/real/logical/character
components (`x = t(1, 2.5, .true.)`, `x = person_t("Ada", 7)`, omitted
components keeping their defaults) stores
its positional arguments into the target, whether written as an executable
assignment or as a scalar variable initializer (`type(t) :: v = t(1, 2.5)`).
Integer, default-real (f32), and logical component default initialisers
materialise on default-initialised instances, propagating through nested
components so an inner type's own defaults show up inline (`x%c%field`). A
scalar allocatable component of intrinsic numeric or logical type
(`integer, allocatable :: v`) holds an inline data pointer that starts null;
`allocate(x%v)`, component read/write, `allocated(x%v)`, and `deallocate(x%v)`
manage it. A rank-1 allocatable array component of intrinsic numeric or logical
type (`integer, allocatable :: v(:)`) holds an inline 16-byte descriptor (data
pointer plus i64 element extent) that starts null; `allocate(x%v(n))` with a
runtime extent, element read/write `x%v(i)`, `allocated(x%v)`, `size(x%v)`, and
`deallocate(x%v)` manage it (whole-component assignment, whole-component reads,
and passing the component as an actual argument stay unsupported). A rank-1
allocatable array of a derived element type (`type(inner), allocatable :: c(:)`)
holds the same inline 16-byte descriptor; `allocate(x%c(n))` `calloc`s n
zero-initialised inner instances (so each element's own allocatable component
descriptors start unallocated), and element component access `x%c(i)%field`
loads the component data pointer, steps in by whole inner instances, and adds
the field slot. `allocated(x%c)`, `size(x%c)`, and `deallocate(x%c)` manage the
descriptor. These stack: an inner element may itself hold an allocatable derived
array or intrinsic allocatable array component, so `obj%z(i)%arr(j)%arr(k)`
reaches through several heap indirections. Non-zero scalar component defaults on
heap elements, whole-component assignment, and allocatable derived array dummy
arguments stay unsupported; genuinely polymorphic `class` forms (e.g.
`allocate(..., source=)` of a differing dynamic type) still decline. A
nested component may carry a bare `inner()` default-constructor initialiser, and
a bare `t()` constructor default-initialises an instance, including for a type
with nested components. A scalar derived `parameter` initialised by a
constructor (`type(t), parameter :: p = t(2, 3)`) is supported at program and
module scope. A type carrying a `final` binding is usable (finalisation is not
modelled, so the finaliser never runs). A `select type` on a monomorphic
declared-type selector - a `class(t)` scalar dummy or local whose dynamic type
is only ever its declared type `t` - resolves statically: the `type is (t)` or
`class is (t)` arm naming the declared type (else `class default`) is chosen at
compile time and lowered inline. A construct that discriminates a runtime
subtype needs a vtable and declines gracefully. A
contained function may return a fixed-size rank-1 array: the result lowers
through the sret ABI (the caller passes the destination buffer as a hidden
result pointer), so `r = vec_fn(...)` and `print *, vec_fn(...)` write the
result straight into the destination. A contained or module function may also
return an allocatable rank-1/2 array of an intrinsic element kind: the caller
passes a zeroed temporary descriptor as the hidden result pointer, the callee
allocates into it, and `lhs = vec_fn(...)` moves that descriptor into the
allocatable destination. When the result extent is a compile-time constant
(an array constructor or a constant-extent `allocate`), it propagates to the
destination so `size`, indexing, and whole-array print of `lhs` work.
A module procedure may `contains` internal procedures, lowered as
flat functions. A `logical`-valued function call (contained or module) prints
directly as `T`/`F`. A non-contained `integer(8)` function (module or
contained) returns through the i64 ABI, so a result wider than 32 bits round
trips correctly; an `integer(8)` scalar dummy argument is passed by reference at
its native width. A module function with a deferred-length
(`character(len=:)`) or runtime-length (`character(len=len(arg))`) character
result is callable and printable from a program in the same file. Module-level
integer, real, and logical scalar variables persist as globals and are visible
across `use`, in the same file or across separate compilation. A module
subroutine or integer function with integer, real, or logical scalar arguments
is callable from a separately compiled program: its signature round-trips
through the `.fmod` and the two objects link, each keeping its own string and
format literals (#284). A named generic interface over such procedures also
round-trips through the `.fmod`, so a `use`-associated generic call in a
separately compiled program resolves to the specific matching its first
argument's kind. A module that exports only contained procedures still writes a
`.fmod`, so a using unit resolves it. A module-scope
scalar variable of a registered derived type is likewise a flat slot global, its
compile-time component defaults folded into the static bytes and read through
`use` (including a `use ..., alias => var` rename); an explicit
structure-constructor initialiser with constant integer arguments folds into
those bytes, and a scalar derived `parameter` exports through the same slot
global (honouring `use, only:`). A fixed-size rank-1 module array of
`integer`/`real`/`logical` is emitted as one shared `[extent x elem]` global,
its extent a literal or compile-time named constant (`dimension(m)`) and any
plain array-constructor initialiser folded into the static bytes; host
association and `use` bind that global, so element access, whole-array ops, and
`size` inquiries all reach the one storage, while allocatable/pointer/character/
runtime-sized module arrays stay a clean diagnostic. A file whose
top-level units are one or more modules with no main program is a valid
translation unit: it lowers to a no-op main, so it compiles to an object with
`-c` (each module's procedures under their mangled symbols) and links to an
empty executable, matching gfortran's own module-only object. When such a file
also holds a main program after its modules, the program's own contained
procedures are registered and lowered from inside the multi-unit container, so
a call to a program-contained scalar function resolves as contained instead of
raising the unsupported-call diagnostic. A single-file A single-file
`submodule (m) s` implements the module procedures its parent module `m`
declares through interface bodies; both the restated signature form and the
separate `module procedure` form lower under the parent's mangled symbol, so a
`use m` call resolves regardless of which submodule holds the body. A parent
generic interface whose specific is a module-procedure interface body dispatches
a call through the generic name to the submodule body implementing that specific.
A plain explicit `interface` block that declares the signature of an external
top-level function (or a module function) whose real definition also appears in
the file reconciles the signature with the definition instead of colliding as a
duplicate, so the call resolves to the real symbol for every scalar result kind
(integer, real, logical).
The `associate` construct binds scalar selectors, a rank-1 unit-stride
array-section selector (`associate (x => a(lo:hi))`, reindexed to lower
bound 1), and a derived-type component selector (`associate (s => a%comp)`);
in all forms a write through the associate name flows back to the selector's
own storage. The
`where` construct masks elementwise assignment over rank-1 integer and real
arrays, including a final `elsewhere`. The `forall` construct, single-statement
and block form, lowers to a sequential loop nest over its index set, with an
optional scalar-comparison mask guarding the body. The
scalar numeric intrinsics `mod`, `modulo`, `sign`, `dim`, `abs`, `iabs`,
`int`, `nint`, `floor`, `ceiling`, `real`, `dble` are supported, as are the
bit intrinsics `iand`, `ior`, `ieor`, `not`, `ishft`, `ishftc`, `ibits`,
`ibset`, `ibclr`, `btest`, and `bit_size` on default `integer` values. The
character intrinsics `len`, `len_trim`, `trim`, `adjustl`,
`adjustr`, `index`, `scan`, `verify`, `repeat`, `achar`, `char`, and `iachar` are
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
constant. A local variable declared `character(len=len(other))`, where
`other` is an already-declared character variable, takes its length from
`other`'s runtime length at that point. The real transcendental intrinsics
lower to libm: `sqrt`, `exp`,
`log`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `gamma`,
`log_gamma`, and `hypot`. The real numeric-model inquiry intrinsics `tiny`,
`huge`, and `epsilon` of a real argument (kind 4 or 8) lower to the constant
real value for the argument's kind, usable in real expressions, conditions, and
`print`. Scalar `transfer(source, mold)` reinterprets `source`'s bit pattern as
`mold`'s type between `integer(4)`/`real(4)` and between `integer(8)`/`real(8)`,
via a typed stack-slot store/load round trip; same-kind `transfer` is the
identity. `open`/`close`/`rewind` map to `fopen`/`fclose`/
`rewind`, preserving an existing file's content when `status=` is omitted;
a file unit's list-directed and numeric-edit-descriptor `read` covers
integer, real, and fixed-length character scalars. Internal `read (buf, *)
value` (list-directed) and `write (buf, fmt) value` with a compound literal
format (`I`/`A` descriptors) are supported. `inquire` covers `exist=`,
`opened=`, and `iostat=` on `file=` and `unit=`. Invalid programs are
rejected during lowering: an integer `SELECT CASE` with overlapping
integer-literal CASE labels; a character-valued I/O specifier
(`STATUS=`, `ACCESS=`, `ADVANCE=`, `IOMSG=`, ...) handed a numeric or
logical literal (`status=1`, `advance=5.`); a relational comparison whose
operands have incompatible intrinsic type classes (`b == i` for logical
`b` and integer `i`, or `c == i` for character `c`); a fixed-size
array assigned an array constructor of the wrong length (`a = [1, 2, 3]`
for `integer :: a(4)`); a named generic interface whose two specific
procedures share an indistinguishable scalar dummy signature (`ambiguous
interfaces`, F2018 C1514); a scalar actual (literal or scalar variable)
passed where a procedure with an explicit interface in the same unit declares
an array dummy (`Rank mismatch in argument`, F2018 15.5.2.4); and a call passing
more actual arguments than that in-unit callee declares dummies (`More actual
than formal arguments`), each fail with a diagnostic.

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
