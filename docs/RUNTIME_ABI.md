# Runtime ABI

This document records the current direct-session MVP ABI. It is an internal
compiler ABI for the current supported subset, not a stable external ABI.
Changes require matching executable tests and updates to
`docs/SUPPORT_CONTRACT.md`.

## Stability Rule

Only the representations listed in this document are supported. Other values,
argument forms, descriptors, and runtime calls are unsupported until their issue
is closed with ABI documentation and executable tests.

## Program Entry

- `ffc` emits a native `main` function through LIRIC.
- `main` is declared `i32 main(i32 argc, ptr argv)`. The C runtime always
  passes argc/argv, so the parameters are present even when unused; they back
  the command-line argument intrinsics below.
- Falling off the end of `program main` returns zero.
- `stop <integer expression>` returns that integer expression as the process
  exit status.
- `command_argument_count()` lowers to `argc - 1` (argv(0) is the program
  name). `get_command_argument(i, value)` copies `argv(i)` into the
  fixed-length character variable `value` through the synthesized helper
  `.ffc.get_arg`, which uses `snprintf("%-*.*s", ...)` to blank-pad or truncate
  to the variable's declared length and rebinds the variable to the new buffer.
  Both are supported only in the main program; the optional `length`/`status`
  arguments and `get_command()` are not supported.

## Scalar Values

- `integer` values use LIRIC `i32` values.
- `real` values use LIRIC `f64` values.
- `logical` values use an `i32` representation: zero is false, nonzero is
  true. Printed logicals branch on that value and emit `T` or `F` (gfortran's
  list-directed ` T`/` F` once the separating blank is added).
- Scalar `character(len=N)` variables keep an `i8*` pointer to literal-backed
  storage plus the declared length `N` in the lowering symbol. Assignment from
  character literals stores exactly `N` characters by truncating long literals
  and blank-padding short literals. Assignment from a character expression
  (a variable, or a `trim`/`adjustl`/`adjustr`/`achar`/`repeat` result) copies
  `min(source length, N)` bytes into a fresh blank-filled `N`-byte buffer at
  runtime and rebinds the symbol to it, the same truncate/pad semantics with
  a runtime rather than compile-time-known source length. `==`, `/=`, `<`,
  `<=`, `>`, `>=` between character operands lower to a three-way compare
  (-1/0/1) over the blank-padded common length, then test that result against
  zero with the same predicate used for integer comparisons.
- The current lowerer keeps ordinary scalar symbols as SSA-like current values.

### Allocatable array descriptor

An `integer/real/logical, allocatable :: a(:)` declaration lowers to a 40-byte,
8-byte-aligned descriptor on the stack, zero-initialised at declaration. The
descriptor is element-kind-agnostic; the element kind lives on the symbol and
drives the per-element byte stride:

```
struct ffc_alloc_1d {
    ptr   data;    // offset 0; 0 means unallocated
    i64   lower1;  // offset 8;  lower bound (1 once allocated)
    i64   upper1;  // offset 16; upper bound (0 when unallocated)
    i64   lower2;  // offset 24; rank-2 only (0 for rank-1)
    i64   upper2;  // offset 32; rank-2 only (0 for rank-1)
};
```

The element byte stride is 4 for `integer`/`real(4)`/`logical` and 8 for
`real(8)`/`integer(8)`. Logical occupies a 4-byte slot, matching the fixed-array
representation.

`data == 0` marks the array unallocated. `allocate(a(N))` (N a literal or
runtime integer) calls `malloc(N*stride)`, stores the pointer in `data`, sets
`lower1 = 1` and `upper1 = N`. `deallocate(a)` calls `free(data)` then zeroes
`data` and `upper1`. `free(NULL)` is a no-op, so deallocating an unallocated
variable exits cleanly rather than erroring (a deliberate divergence from the
standard, which makes it a runtime error).

Element access on an allocated 1-D allocatable is supported as an rvalue and an
assignment target. `a(i)` loads `data` from offset 0 and `lower1` from offset 8
at runtime, computes the element address as `data + (i - lower1) * stride`, and
emits a kind-typed load or store. No alloc/free happens inside element loops:
`allocate` once and index many. Bounds are not checked.

Whole-array constructor assignment (`a = [e1, e2, ...]`) is supported for 1-D
integer, real, and logical allocatables: it frees the old storage, reallocates
for the constructor's element count, and stores each element. Whole-array print
(`print *, a`) is supported when the most recent allocation extent is a
compile-time constant; the print is unrolled over that extent. A runtime-only
extent leaves whole-array print unsupported. Rank-2 allocatables remain
integer-only. Only single-variable, default-lower-bound `allocate`/`deallocate`
are supported.

- Procedure reference arguments use LIRIC `alloca`/`load`/`store` slots at the
  call boundary.
- Scalar `abs`, `min`, and `max` intrinsics are supported for integer and real
  values. Integer-to-real `real()` conversion is supported. They lower inline
  through LIRIC scalar operations, comparisons, branch control, casts, and PHI
  values.

### Scalar pointer and target (#245 B3a)

A `target` local lives in a stable stack slot. On declaration, `ffc` emits an
`i32 alloca` for the slot and stores the initial value (zero) into it. The slot
address is recorded in the lowering symbol; reads and writes go through
`emit_i32_load`/`emit_i32_store` on that address, so any write to the target
is immediately visible through any pointer that shares the slot.

A `pointer` variable carries no separate storage. Instead, the lowering symbol
holds a copy of the target's slot address after pointer assignment. Three flags
track its state at compile time:

| Flag | Meaning |
|---|---|
| `is_pointer` | the variable was declared `pointer` |
| `has_address` / `is_reference` | an address is in scope; reads and writes dereference it |
| `is_associated` | the pointer is currently associated (not nullified) |

`p => t` copies the target's address operand into `p`'s lowering symbol and
sets all three flags. Subsequent reads of `p` emit `emit_i32_load` from that
address; writes emit `emit_i32_store` to the same slot, mutating `t`.

`nullify(p)` clears `has_address`, `is_reference`, and `is_associated` but
emits no code. `associated(p)` (one-argument form) is a compile-time boolean
derived from the `is_associated` flag; it folds to an `i32` immediate `0` or `1`.

This is a straight-line compile-time model. Re-pointing across a branch
(`if (cond) p => a; else p => b`) requires runtime pointer comparison; that
case is not yet supported.

### Two-argument ASSOCIATED (#245 B3c)

`associated(p, t)` returns true when `p` is associated and points to `t`.
In straight-line code `p => t` leaves both `p` and `t` sharing the same
address operand. The two-argument form checks whether `is_associated` is set
and whether the address payload stored in `p`'s symbol matches the address
payload of the named target. If so, it folds to `i32` immediate `1`; otherwise
`0`. No code is emitted; the comparison is entirely at compile time.

This covers the common single-block pattern. If `p` was pointed at a different
target in a preceding branch, the compile-time address payload may be stale and
the check is unsupported.

### Procedure pointers (#245 B3d)

A `procedure(...), pointer :: fp` declaration allocates one `ptr` alloca slot
on the stack. The slot holds the function address as an opaque pointer, zero on
entry. The lowering symbol records `is_proc_pointer = .true.` and
`value_kind = VALUE_PROC_PTR`.

`fp => my_func` writes the address of `my_func` into the slot with a
`ptr store`. The address is obtained through `lr_session_intern`, which returns
a symbol id for the function name, wrapped as a `global_operand`. `fp => null()`
clears `is_associated` on the symbol and emits no store.

A call through `fp` loads the slot with a `ptr load` to get the callee address,
then emits `LR_OP_CALL` with that loaded `ptr` vreg as the first operand
(indirect call). For a function result the return type is `i32`; for a
subroutine the return type is `void`. The argument list is passed to the same
reference-slot ABI used for direct contained-procedure calls.

The call site in the IR looks like:

```
%0 = ptr load <fp_alloca>          ; load function address
%1 = i32 call %0(arg1, arg2, ...)  ; indirect call through ptr
```

Re-pointing `fp` to a different function in a later statement replaces the slot
contents; the next call through `fp` picks up the new address.

## Derived Types

- The MVP derived-type layout supports only scalar integer components.
- Each scalar derived-type variable is stored as one LIRIC array alloca with
  `component_count` `i32` elements.
- Component order is source declaration order. Offset zero is the first
  declared component, offset one is the second component, and so on.
- Component assignment and reads use the same explicit storage operations as
  fixed-size arrays: alloca, aggregate GEP, `i32` store, and `i32` load.
- Constructors, inheritance, type parameters, type-bound procedures, nested
  derived types, derived type arrays, whole-derived assignment, and non-integer
  components have no ABI representation in this slice.

## Procedures

- The supported procedure slice is contained integer, real, and logical
  functions and subroutines with scalar parameters.
- Procedure parameters are currently lowered as LIRIC pointer parameters for
  integer, real, and logical arguments. Callers pass a reference slot; variable
  actual arguments are copied back after the call, and parameter assignment
  stores through the pointer.
- Function results are represented by assignment to the function result name.
- A contained function returning a whole derived value is emitted as a `void`
  function taking a hidden first pointer argument to the caller's result
  storage; the callee binds its result variable to that pointer and writes
  components through it, and `q = make_point()` passes `q`'s storage as the
  hidden argument (no copy). This mirrors the deferred-length character result
  ABI. The size is known at compile time, so no runtime allocation is needed.
- Subroutines return LIRIC `void`; explicit `CALL` statements emit `void` calls.
- Names are emitted as source names for the current single-file subset. A
  deterministic mangling scheme is still required before broader procedure and
  module support.
- A character dummy argument, fixed-length (`character(len=N)`) or
  assumed-length (`character(len=*)`), is bound through the same stack
  {data pointer, i64 length} descriptor the caller builds for the actual
  (see "Deferred-length character" below). An assumed-length dummy reads
  both fields at each use, so `len`/`len_trim` see the caller's runtime
  length. A fixed-length dummy reads only the data pointer at binding time
  and keeps its own declared width N as a compile-time constant, so it sees
  exactly the first N bytes of a (possibly longer) actual, matching
  gfortran's fixed-length dummy association. Character function results are
  supported for the deferred-length (`character(len=:), allocatable`) case;
  see "Deferred-length character" below.
- External procedures, modules, and separate compilation are unsupported; #54
  owns symbol mangling and link behavior.

### Assumed-shape runtime extent (W2)

A rank-1 assumed-shape dummy `a(:)` whose actual has no compile-time-foldable
shape (an allocatable actual) carries its element count as a hidden `i64`
argument, passed by reference like every other scalar reference argument: the
caller allocates an `i64` stack slot, stores the extent, and passes the
slot's address. The hidden argument is appended after all of the subroutine's
visible pointer parameters, one per such dummy, in dummy declaration order.
The callee loads and truncates it to `i32` once at entry and reuses that
value everywhere the dummy's extent is needed.

The existing whole-arena compile-time fold (a whole-array actual with a
literal or caller-scope-parameter extent) stays the fast path: it is tried
first, and only a dummy for which that fold fails gets a hidden parameter, so
an already-working compile-time-resolved assumed-shape dummy keeps its
original signature and no hidden argument.

This slice covers: `size(a)` (no `dim`, and `dim=1`), `ubound(a, 1)`, element
read and write `a(i)` (extent-independent for rank-1, so it needs no ABI
change), a `do` loop bound by `size(a)`, and `sum(a)` for `integer` elements
(a genuine runtime loop, since there is no compile-time extent to unroll
over). Rank-2 assumed-shape dummies, function (not subroutine) dummies,
array-section and array-constructor actuals, and `sum`/`product`/`maxval`/
`minval` over non-integer runtime-extent elements are not yet covered and
keep the pre-existing "assumed-shape dummy extent must come from a
whole-array actual of compile-time size" diagnostic.

`call obj%method(args)` (a type-bound subroutine call) inserts the passed-object
receiver ahead of the explicit `args` at the callee's passed-object dummy
position, so an explicit argument's call-site position is not its callee dummy
position whenever the callee has more than one dummy before that argument. The
hidden-extent lookup accounts for this: `prepare_reference_args` takes an
optional `self_position` (the receiver's 1-based dummy position) and maps each
call-site argument to its true dummy position before checking whether that
dummy needs a hidden extent, so an assumed-shape runtime-extent dummy reached
through a type-bound call resolves the correct actual.

## Runtime Calls

- Scalar `print` lowers to external C `printf`/`snprintf` calls.
- List-directed record layout: one separating blank is written before every
  value. The first blank is the record's carriage control. No
  blank is written between two consecutive character values, so they print
  concatenated (matching gfortran). Each value field below carries no leading
  blank of its own; a trailing newline closes the record.
- The FortFront standard-example corpus checks stdout and exit status
  byte-for-byte against `gfortran -w` for every example gfortran accepts.
  Files gfortran rejects are counted as `NOREF` by the conformance runner and
  still must compile and run through ffc.
- The per-value format globals are:
  - integer/logical: `%11d` for integers. The field plus its leading separator
    blank reproduce gfortran's default list-directed `integer(4)` width of 12.
    `integer(8)` (width 22) is deferred. Logicals print `T`/`F` (the leading
    blank is the separator, so gfortran's ` T`/` F` is reproduced).
  - real: emitted through the synthesized helper `.ffc.print_real8` (see
    below), not a single `printf` format.
  - character: `%s`.
- `real(8)` list-directed output is produced by a helper function
  `.ffc.print_real8(double)` synthesized once into the module. It reproduces
  gfortran exactly: 17 significant digits, fixed (F) notation for a decimal
  exponent in `[-1, 16]` (right-justified in 20 columns, five trailing
  blanks) and exponential notation otherwise (one digit before the point, 16
  after, an uppercase `E`, a sign, a three-digit exponent, right-justified in
  25 columns). `Infinity`/`-Infinity`/`NaN` are printed for non-finite values.
  The helper builds the digits with `snprintf("%.16e", ...)`, reads the
  decimal exponent with `atoi`, and formats the field accordingly. ffc lowers
  every Fortran `real` as `real(8)`, so `real(4)` literals also use this form;
  a kind-parametrised format is deferred until ffc lowers `real(4)` distinctly.
- Character literal print passes a pointer to a null-terminated global byte
  array to `printf`. Scalar character variable print passes a pointer to a
  global byte array containing the fixed-length value followed by a null
  terminator. The C `printf` shim consumes the terminator, not an explicit
  length argument.
- Object output may contain unresolved references such as `printf`; final
  linking is responsible for resolving the C runtime.
- This `printf`/`snprintf` path is the supported scalar I/O surface. Internal
  I/O and formatted `write` to file units are owned by later issues.

### Formatted print

`print fmt, items` with a literal format string lowers to direct `printf`
calls that honour the format, with no list-directed leading blank and one
record newline at the end. The supported edit-descriptor subset includes:

- `I0` maps to `%d`, `Iw` to `%wd` (for example `I5` is `%5d`).
- `A` maps to `%s`; `Aw` maps to `%ws`. Character items pass a pointer to a
  null-terminated buffer, so `%s` prints the variable's full declared width.
- Compound literal formats made from `I`, `X`, and `F` descriptors are
  supported on stdout. `X` emits blanks. `F w.d` lowers through `snprintf`
  into a temporary buffer, then prints that buffer as a string field.
- Repeat counts and reversion are still limited. `A` remains single-descriptor
  only in formatted `print`, and unsupported descriptors still fail with a
  diagnostic.

`print *, ...` remains the list-directed path described above.

### Internal write

`write (buf, fmt) value`, where `buf` is a fixed-length character variable and
`fmt` is a literal single edit descriptor (`I0`/`Iw`/`A`), formats the value
into `buf` and blank-pads it to the declared length. It uses two `snprintf`
calls: the first formats the field (`%d`/`%wd`/`%s`) into a temporary, the
second writes `%-*.*s` (left-justify, blank-pad and truncate to the buffer
length) into the variable's storage, which is then rebound to that buffer.
Single value only; compound formats and write-to-file-unit are not supported.

### Internal read

`read (buf, fmt) value`, where `buf` is a character variable and `fmt` is a
literal integer descriptor (`I0`/`Iw`), parses an integer from `buf` with
`sscanf(buf, "%d", &slot)` into a stack slot, then loads it into the integer
target. Integer scalar targets only; real/character reads and read-from-file
are rejected.

## Deferred-length character

A `character(len=:), allocatable` variable (and the `character(:), allocatable`
synonym) is a 16-byte descriptor split across two 8-byte stack slots:

```
data    : i8*   heap pointer, 0 when unallocated
length  : i64   current length in bytes, 0 when unallocated
```

- The two slots are stack allocas. On declaration both fields are zeroed, so
  an unallocated descriptor reads as `data == 0, length == 0`.
- Length is bytes (ASCII for now), not codepoints.
- Assignment and concatenation allocate the result (`malloc` for a function
  result that escapes its scope, otherwise a stack buffer), write the bytes
  plus a trailing null, and store the new pointer and length into the
  descriptor.
- The same descriptor shape backs an allocatable character function result
  and a deferred-length dummy passed by reference.
- Storage for a local deferred-length character is chosen so scope exit needs
  no explicit free: a literal assignment points `data` at a static global, and
  a concatenation result is a stack buffer. Only a deferred-length function
  result that must outlive its frame is `malloc`'d, and its owner is the
  caller. Because a local never owns heap memory, no `free` is emitted at
  scope exit and an unallocated descriptor is never freed.

## Derived-type info

Each `type ... end type` definition emits a compile-time constant describing
the type, the foundation for polymorphic dispatch (`select type`). It is a
compiler-private layout, not a Fortran-visible type.

```
struct ffc_type_info_t {
    i64 id;          // dense per-compilation-unit type index
    i64 size_bytes;  // storage size (each component is a 4-byte i32 slot)
};
```

The instance is a 16-byte const global named `__ffc_type_info_<typename>` (a
module prefix is added once module-scope types export type info). The `id` is
assigned monotonically as types are collected. Nothing references these
constants yet; later polymorphism slices compare a value's type pointer
against them.

## Module artefact format

`ffc -c <source>.f90` writes one `<modulename>.fmod` next to the object file
for each module the source defines. The file records the module's exported
interface so a later unit can resolve `use <module>` without the source. It is
a line-oriented subset of TOML, with no source locations or comments.

```toml
[module]
name = "shapes"
ffc_version = "0.1.0"

[[parameter]]
name = "max_pts"
kind = "integer"
value = 10

[[derived_type]]
name = "point_t"
components = [
    { name = "x", kind = "integer" },
    { name = "y", kind = "integer" },
]
```

- `[module]` carries the module name and the emitting `ffc` version.
- Each `[[parameter]]` is a named constant: `name`, `kind` (the normalised
  scalar type token), and the literal `value`.
- Each `[[derived_type]]` is a type definition with its `components`, each a
  `{ name, kind }` pair.
- `kind` is `integer`, `real`, `logical`, `character`, or `type(<name>)`.
- Module variables and module procedures are not yet exported.

## Unsupported ABI Work

- #53: array descriptors, allocatables, and pointer representation.
- #54: module and external symbol mangling.
- #55: runtime I/O beyond the current scalar `printf` shim.
