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
`reshape` stay rank-1/rank-2); rank-1 through rank-4 local automatic arrays
sized by a
runtime bound (`integer :: a(n)` or `real :: a(0:n)` inside a procedure, where
`n` is a dummy, host, or COMMON value the compiler cannot fold), covering
element read/write, whole-array scalar broadcast, and scalar `sum`, `product`,
`maxval`, and `minval` over integer or real elements, plus scalar `count` over
logical masks, through the runtime descriptor; runtime-bounded
scalar section assignment also supports rank-1
through rank-4 sections with multiple retained dimensions (for example
`a(2:n,1:m) = value`), using live bounds and column-major coordinates;
array-valued RHS forms and unsupported noncontiguous or ambiguous sections
remain explicit refusals. Rank-1 and rank-2 additionally support `size` and
`sum` (integer or real), while rank-1 also supports `lbound`/`ubound`,
whole-array `print`, and whole-array copy over a runtime
loop; fixed-size
rank-1 and
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
`size`, `shape`, `sum`, `product`, `maxval`, `minval`, `norm2`,
`dot_product`,
`matmul`, `transpose`, `reshape` (also as a declaration initializer, with
integer-source-to-real conversion, and with a `shape(X)` shape argument or
keyword `source=`/`shape=` form), `lbound`, `ubound`, `count`, `any`, `all`
(scalar-result reductions in any context: a bare whole logical array
(`if (.not. all(l))`), a logical array constructor (`all([.true., .false.])`),
or a whole-array/section/constructor/scalar elementwise comparison using a
relational or `.eqv.`/`.neqv.` operator, e.g. `if (any(a /= b))`,
`all(a == [1, 2, 3])`, `any(a(2:4) /= b(2:4))`, `any(d .neqv. mask)`; a
real array operand may use elemental `abs`, e.g.
`if (any(abs(samples) > tolerance))`, and
dim-wise
`sum`/`product`/`count`/`any`/`all` of a rank-2 source into a rank-1 target
with a compile-time `dim`, e.g. `s = sum(a, 1)`, `m = any(a == b, 2)`),
and rank-1 scalar `maxloc`/`minloc` (optional `dim=1` and `mask`); scalar
element read and write on an allocated rank-1 through rank-3 integer, real, or
logical allocatable (`a(i)`, `a(i,j)`, `a(i,j,k)`); whole-array assignment from an array
constructor to a 1-D allocatable with auto-reallocation (`a = [e1, e2, ...]`);
whole-array copy between separate rank-2 or rank-3 integer, real, or logical allocatables
(`a = b`) with both runtime descriptor extents, target reallocation, and a
bounded column-major copy loop;
whole-array assignment from a general elementwise expression to a 1-D
allocatable already allocated to a compile-time-constant extent
(`a = b + c`, `a = a * 3`); whole-array `print`, `size(a)`, and integer
`sum(a)` of a 1-D allocatable whether its extent is a compile-time constant or
a runtime value (`allocate(a(n))` with `n` a variable), reading a runtime extent
back from the descriptor, including `T`/`F` formatting for a logical
allocatable element or whole array. Rank-2 and rank-3 allocatables also
support runtime `size(a)` and `size(a, dim)` through descriptor extents; a
fixed-length rank-1
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
program-contained and module procedures. A contained `REAL :: x(..)` dummy
with one whole rank-1 actual and one `RANK (1)` arm uses the canonical borrowed
array descriptor boundary. The same boundary accepts rank-2, rank-3, and
rank-4 whole REAL actuals with one matching `RANK (2)`, `RANK (3)`, or
`RANK (4)` arm and uses descriptor-driven column-major scalar addressing. Rank
default/star, scalar or rank-5-and-higher actuals, dynamic shapes, sections,
global storage, aliases, and ownership are refused. The actual may also be a contiguous
rank-1 array section with compile-time bounds -- a stride-1 slice `a(2:4)` or a
whole column `m(:,j)` (integer, `real`, `real(8)`) -- whose extent folds from
the section and whose first-element address binds the dummy in place. A rank-1
or rank-2 assumed-shape dummy of a
subroutine also accepts an allocatable actual of runtime-only extent:
the per-dimension extents travel as hidden arguments, so `size(a)`,
`size(a, dim)`, `ubound(a, dim)`, element read/write (rank-2 uses the runtime
leading extent as the column-major stride), a `do` loop bound by
`size(a, dim)`, and scalar `sum(a)`/`product(a)`/`maxval(a)`/`minval(a)` over
default integer, `real`, or `real(8)` elements all work against the caller's
runtime allocation. An assumed-size dummy
(`a(*)`, `a(n1, ..., *)`, dummy arguments only) folds its leading dimensions
at compile time and binds to the actual's base address, so element read/write
and `lbound(a, dim)` work; the trailing dimension carries no extent, so
`ubound`/`size` on it and whole-array operations are not supported. A named
constant declared `a(*) = [...]` (single dimension, a non-implied-do array
constructor initializer) instead takes its extent from the initializer's
element count. A scalar
`integer`/`real`/`logical, allocatable` variable
supports `allocate`/`deallocate` and `allocate(x, source=<expr>)`/`mold=<expr>`
with any source expression. A scalar allocatable derived variable
(`type(t), allocatable :: x`) starts unassociated with no storage; `allocate(x)`
(bare or `allocate(t :: x)`) `malloc`s one instance and runs its default
initialisation, then component read/write (`x%field`), `allocated(x)`,
whole-scalar copy (`y = x`), auto-allocation on assignment to an unallocated
target (`x = t(...)`), `move_alloc(x, y)`, and `deallocate(x)` manage it; the
heap instance may itself hold allocatable array components
(`allocate(x%c(n))`). A rank-1/rank-2 allocatable array dummy argument
aliases the caller's own descriptor, so `allocate`/`deallocate`/element writes
inside the callee are visible to the caller. A single `allocate` or
`deallocate` may list several targets at once
(`allocate(a(N), b(N), c(M,K), stat=ierr)`, `deallocate(a, b, c)`); each target
is sized from its own subscripts and an optional `stat=` integer is set to 0 on
success. A rank-2 allocatable also supports whole-array scalar broadcast
(`m = 9`) once allocated to a compile-time-constant shape. Scalar
integer `pointer`/`target` with `p => t`, read/write through `p`,
`associated(p)`, and `nullify(p)` is supported, as is `allocate(p)` on a
scalar pointer, which gives it fresh heap storage. Rank-1 and rank-2 fixed-size
`integer`/`real`/`logical`/`complex` `pointer`/`target` arrays support whole-
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
across kinds. The `%re`/`%im` complex-part designators read and write a scalar complex or a
fixed-size complex array element (`x%re = 1`, `x(i)%im = 4`), and `real()`/`aimag()`
also accept a complex array element (`real(x(i))`). Fixed-size rank-1/rank-2 complex
arrays support element assignment, element reads, elemental `+`/`-`/`*`/`/`
between array elements, single-element `print`, and whole-array assignment
with elementwise `+`/`-`/`*`/`/`, whole-array copy, or scalar broadcast
(`c = a + b`, `c = a`, `c = (1.0, 2.0)`) between conforming complex arrays.
An `if` condition accepts any
scalar logical expression: `.not.`/`.and.`/`.or.`/`.eqv.`/`.neqv.` trees, a
logical array element, a derived-type logical component, `allocated(a)`, and
a contained logical function's result, including the one-line
`if (cond) stmt` form without `then`. An empty or behavioral-only type
(no data components, a bare `type :: t; end type` or a type with only a
`contains` type-bound-procedure block) registers with a hidden placeholder
slot, so declarations plus dispatch/allocation/extension resolve against it.
Derived types take scalar
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
Integer, default-real (f32), `real(real64)`, logical, fixed-length character,
and `c_null_ptr` component default initialisers materialise on
default-initialised instances and on components a structure constructor omits,
propagating through nested
components so an inner type's own defaults show up inline (`x%c%field`). A
scalar allocatable component of intrinsic numeric or logical type
(`integer, allocatable :: v`) holds an inline data pointer that starts null;
`allocate(x%v)`, component read/write, `allocated(x%v)`, and `deallocate(x%v)`
manage it. A deferred-length allocatable character component
(`character(len=:), allocatable :: s`) holds the canonical character descriptor
inline (data pointer plus i64 length) and starts unallocated; assignment
allocates to the right-hand side length and deep-copies the bytes, reading,
`len`, comparison, concatenation, and `print` follow the current length, a
whole-derived copy (`y = x`) gives the destination its own buffer, and
`deallocate(x%s)` frees the owned data once and clears the descriptor. A rank-1 or rank-2 allocatable array component of intrinsic integer, real, or
logical type (`integer, allocatable :: v(:,:)`) holds an inline descriptor
(data pointer plus one i64 extent per dimension; 16 or 24 bytes) that starts
null. `allocate(x%v(n))` and `allocate(x%v(m,n))` accept runtime extents;
element read/write, `allocated(x%v)`, `size(x%v)` (including
`size(x%v,dim)`), and `deallocate(x%v)` manage it. Rank-2 indexing is
column-major and uses the stored leading extent. Whole-component assignment,
whole-component reads, passing the component as an actual argument, aliases,
unsupported kinds, and higher-rank components stay explicitly unsupported. A rank-1
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
arguments stay unsupported; genuinely polymorphic `class` forms outside the
scalar dispatch slice (e.g. `allocate(..., source=)` of a differing dynamic
type) still decline. A rank-1 `class(base_t), allocatable :: a(:)` array may
be allocated with a compatible concrete type-spec (`allocate(child_t ::
a(n))`), preserving the dynamic type and concrete element stride through
`select type`. A rank-1 or rank-2 integer whole array may also bind to
`class(*), intent(in) :: values(:)` or `values(:,:)` through the canonical array
descriptor; `select type (items => values)` narrows that view for `size(items)`
and column-major element reads. Integer and `real(8)` whole-array actuals use
the canonical descriptor's element size, real type code, per-dimension shape
and byte stride, and reserved runtime type id. This bounded class-star array
slice refuses rank-3-and-higher arrays, default/single-precision real, logical
and other unsupported kinds, sections, allocatable ownership, and
pointer/target ownership. A scalar `class(base_t), pointer` may be explicitly
allocated with a compatible concrete type-spec (`allocate(child_t :: p)`), and
a type-bound function call through `p` dispatches through the allocated dynamic
type's vtable. A scalar `class(base_t), intent(in)` dummy borrows the same
descriptor and dispatches one non-generic default-`PASS` binding to a child
override at runtime. Class-pointer arrays, pointer reassociation, pointer
deallocation/ownership/finalization, generic/deferred bindings, and allocation
of a finalizable pointer type remain rejected. A
nested component may carry a bare `inner()` default-constructor initialiser, and
a bare `t()` constructor default-initialises an instance, including for a type
with nested components. A scalar derived `parameter` initialised by a
constructor (`type(t), parameter :: p = t(2, 3)`) is supported at program and
module scope. A type carrying a single scalar `final` binding runs that
finaliser exactly once when an owned scalar value of the type dies: when the
procedure owning a local derived variable ends execution, and when an owned
allocatable scalar is deallocated. Dummy arguments borrow their storage and are
never finalised; array finalisation and several `final` bindings on one type
stay rejected. A statically declared `type(child_t)` receiver selects a local
type-bound override in `type, extends(parent_t) :: child_t`, including default
`PASS` and `NOPASS`. A `select type` on a monomorphic
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
format literals (#284). A separately compiled module's `use, only:` list and
rename (`local => remote`) are checked against the `.fmod` exports and preserve
the remote name for linking (#328). A named generic interface over such
procedures also
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
`size` inquiries all reach the one storage. A rank-1/rank-2 module-scope
`allocatable` array of the same element kinds is instead a zero-initialised
40-byte descriptor global, so it starts unallocated and `allocate` (including a
multi-object `allocate(a(n), b(m))`), `deallocate`, `size`, and element access
in module procedures and using units all share that one descriptor;
pointer/character/derived-element and rank > 2 module arrays stay a clean
diagnostic. A file whose
top-level units are one or more modules with no main program is a valid
translation unit: it lowers to a no-op main, so it compiles to an object with
`-c` (each module's procedures under their mangled symbols) and links to an
empty executable, matching gfortran's own module-only object. When such a file
also holds a main program after its modules, the program's own contained
procedures are registered and lowered from inside the multi-unit container, so
a call to a program-contained scalar function resolves as contained instead of
raising the unsupported-call diagnostic. A contained procedure may reference a
host-associated named constant (an `integer, parameter` declared in the
containing program or module) in an integer expression: the value folds from
the declaration that FortFront's binding resolution names at the reference, so
an identically spelled constant in a module the unit never `use`s stays
invisible and still raises the undeclared-name diagnostic. Host-associated
*variables* are still not supported. A single-file A single-file
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
(integer, real, logical). When the definition lives in another translation unit
instead, the interface still types the call and the symbol resolves at link
time: a source of only top-level procedures compiles with `-c` to an object
that defines no `main` and links beside a separately compiled driver. Asking
for an executable from such a procedure-only source is rejected with `no main
program unit`.
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
`log_gamma`, and `hypot`. The real conversions `aint` (truncate toward zero)
and `anint` (round half away from zero) lower to libm `trunc`/`round` on
`real(4)` and `real(8)` operands, elementally over a whole real array, and
accept an optional `KIND` selector of 4 or 8 that fixes the result kind; any
other `KIND` is rejected. The real numeric-model inquiry intrinsics `tiny`,
`huge`, and `epsilon` of a real argument (kind 4 or 8) lower to the constant
real value for the argument's kind, usable in real expressions, conditions, and
`print`. Scalar `transfer(source, mold)` reinterprets `source`'s bit pattern as
`mold`'s type between `integer(4)`/`real(4)` and between `integer(8)`/`real(8)`,
via a typed stack-slot store/load round trip; same-kind `transfer` is the
identity, and a whole-array `source` contributes its first element. The same
reinterpretation runs elementwise for a whole-array `transfer` assignment
(`a = transfer(r, a)`), including a compile-time constant `size` argument that
must match the result array's size; `size` must not be negative and the source
must supply at least as many elements as the result. `open`/`close`/`rewind` map to `fopen`/`fclose`/
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
an array dummy (`Rank mismatch in argument`, F2018 15.5.2.4); a nonallocatable
actual with statically fewer available elements than an explicit-shape dummy
(`Actual argument contains too few elements`); a call passing
more actual arguments than that in-unit callee declares dummies (`More actual
than formal arguments`); a main-program- or module-scope array whose bound is a
function call (`array with nonconstant bounds`, e.g. `integer :: a(get_n())` or
`a(command_argument_count())`); a procedure-local automatic array placed in
`common` or `equivalence` (`cannot appear in COMMON` / `cannot be an
EQUIVALENCE object`); a main-program- or module-scope `class` entity that is
neither allocatable nor a pointer (`must be dummy, allocatable or pointer`); and
an explicit-interface body whose function result rank, base type, or `pointer`
attribute disagrees with the real definition (`mismatch in function result
between interface and definition`), each fail with a diagnostic.

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
export TMPDIR=/mnt/storage/lazy-fortran-artifacts-20260806
printf 'program main\nend program main\n' > "$TMPDIR/empty.f90"
LIBRARY_PATH=../liric/build fpm run ffc -- "$TMPDIR/empty.f90" -o "$TMPDIR/empty"
"$TMPDIR/empty"
echo $?
```

## Conformance

`docs/CONFORMANCE.md` documents the conformance gauntlet runner that
drives external Fortran test corpora through the full `ffc` pipeline. Each run
writes an expectation-neutral observation sidecar; expectation views, including an
XFAIL-disabled view, can be regenerated from it without compiling or running
the corpus again.

Single-command gate (build + all suites, fails on FAIL or XPASS):
```bash
scripts/conformance_check.sh
```

Fetch external corpora (lfortran, gfortran-dg):
```bash
scripts/fetch_corpora.sh
```

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
  (hard cap 100). Split growing lowerer code into modules or submodules
  with explicit interfaces. Do not add production `include` fragments.
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
