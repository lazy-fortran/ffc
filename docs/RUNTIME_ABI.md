# Runtime ABI

This document records the current direct-session MVP ABI. It is an internal
compiler ABI for the current supported subset, not a stable external ABI.
Changes require matching executable tests and updates to
`docs/SUPPORT_CONTRACT.md`.

## Architecture status

`array_descriptor_t` in `docs/ARRAY_DESCRIPTOR_ABI.md` is the canonical array
representation for migrated runtime paths. New lowering must not introduce a
second array, section, pointer, assumed-shape, character-array, or polymorphic
descriptor convention. The remaining legacy paths are migration work, not
additional supported ABIs; each must be removed with an ownership/lifetime
oracle when its issue closes.

OPEN/CLOSE and file-unit WRITE lowering is implemented in the descendant
`src/session_program_lowering_open_close.f90` with explicit interfaces in
`session_program_lowering_impl`; the old textual include is not part of the
compiler architecture. This move preserves the runtime ABI below, including
the current `STATUS=` character-value contract and unit-number resolution.

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

An `integer/real/logical, allocatable :: a(:)` or `a(:,:)` declaration lowers to
one canonical array descriptor, specified in full by `ARRAY_DESCRIPTOR_ABI.md`
(#336). The slot is a 200-byte, 8-byte-aligned record on the stack (or in
static storage for a module variable), zero-initialised at declaration. The
bespoke 40-byte `{data, lower1, upper1, lower2, upper2}` record it replaces is
gone; nothing reads or writes those offsets any more.

The descriptor records the element size and type code, so it is no longer
element-kind-agnostic. Element byte size is 4 for `integer`/`real(4)`/`logical`
and 8 for `real(8)`/`integer(8)`; logical occupies a 4-byte slot, matching the
fixed-array representation.

Field access happens only through the helpers in
`session_program_lowering_alloc_descriptor.f90`. The helpers are descendant
module procedures with explicit interfaces in the lowering implementation
module, so the byte offsets appear in exactly one implementation unit without
textual inclusion.

The base pointer stays at offset 0, so `base == 0` still marks the array
unallocated. What changed is that each dimension records
`(lower_bound, extent, stride_bytes)` rather than `(lower, upper)`. The upper
bound is not stored: it is recomputed as `lower + extent - 1`, so the shape has
a single representation.

`allocate(a(N))` (N a literal or runtime integer) calls `malloc(N*element_size)`,
stores the pointer at offset 0, and writes unit lower bounds, the requested
extents, and contiguous column-major byte strides
(`stride(1) = element_size`, `stride(d) = stride(d-1) * extent(d-1)`). It sets
the allocated, associated, owning, and contiguous flags.

`deallocate(a)` calls `free(base)` then returns the descriptor to the
unallocated state: null base, cleared flags, zero extents. The element size,
type, and rank survive, so a deallocated entity still describes what it can
hold. `free(NULL)` is a no-op, so deallocating an unallocated variable exits
cleanly rather than erroring (a deliberate divergence from the standard, which
makes it a runtime error), and a second `deallocate` frees nothing rather than
handing the same block to the deallocator twice.

`move_alloc(from, to)` copies the whole descriptor record and then clears
`from`. Ownership travels with the base pointer, so clearing frees nothing.

Because allocatable arrays and assumed-shape dummies now share this one layout,
an allocatable actual and the dummy it binds to agree on extents, on
column-major order, and on element addresses by construction rather than by
two representations happening to match.

Allocatable **components** of a derived type use an inline descriptor owned by
the containing instance. Intrinsic integer, real, and logical components of
rank one or two store `{data, extent1[, extent2]}` (16 or 24 bytes); the data
pointer is null until allocation and `size`/`allocated`/element access and
deallocation use the stored extents. This component descriptor is separate
from the canonical standalone array descriptor described above. Whole
component assignment, passing a rank-2 component as an actual, aliases,
unsupported kinds, and rank greater than two remain deliberate diagnostics.

Element access on an allocated 1-D allocatable is supported as an rvalue and an
assignment target. `a(i)` loads the base pointer and dimension 1's lower bound
at runtime, computes the element address as `base + (i - lower1) * stride`, and
emits a kind-typed load or store. No alloc/free happens inside element loops:
`allocate` once and index many. Bounds are not checked.

Whole-array constructor assignment (`a = [e1, e2, ...]`) is supported for 1-D
integer, real, and logical allocatables: it frees the old storage, reallocates
for the constructor's element count, and stores each element. Whole-array copy
from a separate rank-2 intrinsic allocatable (`a = b`) is also supported for
integer, real, and logical elements when both extents are runtime descriptor
values: the target is freed and reallocated, then a bounded column-major loop
copies all elements. Whole-array print (`print *, a`) is supported when the
most recent allocation extent is a compile-time constant; the print is
unrolled over that extent. A runtime-only extent leaves whole-array print
unsupported. Rank-2 allocatable expressions, components, aliases, global
descriptors, unsupported kinds, and higher-rank whole-array copies remain
unsupported. Only single-variable, default-lower-bound `allocate`/`deallocate`
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
(indirect call). For a function result the return type follows the statically
resolved scalar interface: `i32` for integer, `f32` for default real, and
`f64` for `real(8)`; for a subroutine the return type is `void`. The argument
list is passed to the same reference-slot ABI used for direct contained-
procedure calls. The direct real(8) procedure-pointer slice requires one
same-unit target assignment outside control flow; unresolved, generic,
incompatible, and flow-sensitive targets are rejected rather than assigned an
ABI by guess.

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
- Contained procedures use source names. Module procedures use
  `__<module>_MOD_<name>`, matching gfortran's symbol convention. A using unit
  loaded from `.fmod` calls that symbol as an external reference; the separately
  compiled module object supplies the definition.
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
- Separate compilation covers module subroutines and integer functions whose
  dummies are supported by-reference integer, real, or logical scalars. General
  top-level external-procedure signatures remain outside this ABI slice.

### Alternate returns

Each `*` in a dummy argument list is a distinct positional alternate-return
slot, not a named dummy: it occupies no parameter position and takes no
argument. A subroutine that declares one or more `*` dummies carries one hidden
trailing `i32`-by-reference selector parameter, appended after the visible
pointer parameters and after any hidden assumed-shape extent arguments. `return
n` stores `n` into that slot and returns; a plain `RETURN` (or falling off the
end) leaves the slot untouched.

The caller allocates the selector slot, stores `0` into it, drops each `*label`
actual argument from the passed argument list, and passes the slot's address as
the trailing argument. After the call it loads the selector and branches to the
block of the `n`-th `*label` argument when the value is `n`; `0` (and any value
outside `1..n`) falls through to the statement after the `CALL`, as the standard
requires. Because the branch targets are statement labels, a call with
alternate-return arguments is lowered only inside a body that carries statement
labels. `return n` with `n` outside the declared slot count is rejected at
compile time.

### Assumed-shape runtime extent (W2)

A rank-1 or rank-2 assumed-shape dummy (`a(:)` or `a(:,:)`) whose actual has no
compile-time-foldable shape (an allocatable actual) carries its per-dimension
extents as hidden `i64` arguments, passed by reference like every other scalar
reference argument: the caller allocates an `i64` stack slot per extent, stores
the extent, and passes the slot's address. The hidden arguments are appended
after all of the subroutine's visible pointer parameters, `rank` per such dummy
(one for a rank-1 dummy, two for a rank-2 dummy, in dimension order), and dummies
are taken in declaration order. The callee loads and truncates each to `i32`
once at entry and reuses those values everywhere the dummy's extents are needed.
The rank-2 extents come from the actual's descriptor bounds: dimension 1 from
offsets (8, 16), dimension 2 from offsets (24, 32).

The existing whole-arena compile-time fold (a whole-array actual with a
literal or caller-scope-parameter extent) stays the fast path: it is tried
first, and only a dummy for which that fold fails gets hidden parameters, so
an already-working compile-time-resolved assumed-shape dummy keeps its
original signature and no hidden arguments.

This slice covers, for a rank-1 or rank-2 runtime-extent dummy: `size(a)`
(no `dim`; rank-2 returns the product of both runtime extents), `size(a, d)`
and `ubound(a, d)` for each dimension, element read and write `a(i)` / `a(i, j)`
(rank-2 column-major addressing uses the runtime leading extent as the stride),
and a `do` loop bound by `size(a, d)`. `sum(a)` for `integer` elements uses a
genuine runtime loop (rank-1 only). Function (not subroutine) dummies,
array-section and array-constructor actuals, `sum`/`product`/`maxval`/`minval`
over non-integer or rank-2 runtime-extent elements, and rank-2 whole-array
operations (`print a`, `a = b`, `matmul`, `transpose`) are not yet covered and
keep the pre-existing "assumed-shape dummy extent must come from a
whole-array actual of compile-time size" diagnostic (or the relevant
whole-array diagnostic).

`call obj%method(args)` (a type-bound subroutine call) inserts the passed-object
receiver ahead of the explicit `args` at the callee's passed-object dummy
position, so an explicit argument's call-site position is not its callee dummy
position whenever the callee has more than one dummy before that argument. The
hidden-extent lookup accounts for this: `prepare_reference_args` takes an
optional `self_position` (the receiver's 1-based dummy position) and maps each
call-site argument to its true dummy position before checking whether that
dummy needs a hidden extent, so an assumed-shape runtime-extent dummy reached
through a type-bound call resolves the correct actual.

### Genuine assumed-rank `RANK (1)` / `RANK (2)` / `RANK (3)` / `RANK (4)` slice

The narrow genuine assumed-rank boundary is descriptor-only and uses the
canonical `array_descriptor_t` in `ARRAY_DESCRIPTOR_ABI.md`. For a contained
scalar-element `REAL :: x(..)` dummy, a call with a whole rank-1, rank-2,
rank-3, or rank-4 REAL actual passes one pointer to a borrowed 200-byte
descriptor. The
descriptor carries the actual rank, element size and REAL type code, and each
active dimension's `lower_bound=1`, runtime `extent`, and byte `stride`. There
are no hidden rank or extent arguments and no bare data-pointer fallback.

The callee retains the descriptor pointer at entry. A single statically valid
matching `RANK (1)`, `RANK (2)`, `RANK (3)`, or `RANK (4)` arm loads base and
all active extents; rank 1 also uses the descriptor stride, while rank 2,
rank 3, and rank 4 use column-major linear element addressing. For rank 3 the
linear index is `i1 + (i2-1)*extent(1) + (i3-1)*extent(1)*extent(2)`; rank 4
adds `(i4-1)*extent(1)*extent(2)*extent(3)`, with the descriptor element-size
stride. The callee does not release or own the descriptor or its storage.
`RANK DEFAULT`, `RANK (*)`, scalar or rank-five-and-higher actuals,
dynamic shapes, sections/aliases (including pointers), global or owning
storage, non-REAL elements, unsupported or non-matching rank arms, and
ownership are named lowering refusals;
static-rank SELECT RANK continues to use its existing compile-time dispatch.

## Runtime Calls

- Each object prefixes its per-unit `.ffc.*` string, format, and character
  content globals with a hash of the output path. Objects compiled separately
  therefore keep distinct literals when linked into one executable.
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

The canonical runtime descriptor for new character interfaces is specified in
`CHARACTER_DESCRIPTOR_ABI.md`. Current lowering paths use the following
16-byte representation until their descriptor migration issues land.

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

## Scalar class descriptor

A polymorphic scalar (`class(t)` dummy, local, or allocatable) is described by
one 32-byte, 8-byte-aligned descriptor. It is the single canonical convention
for scalar polymorphism; no lowering path may invent a competing one. The
Fortran-side definition and its accessors live in
`src/ffc_polymorphic_descriptor.f90`.

```
struct ffc_polymorphic_descriptor_t {
    i8*  data;           // offset  0: address of the value's storage
    i64  declared_type;  // offset  8: declared type identity
    i64  dynamic_type;   // offset 16: dynamic type identity
    i32  ownership;      // offset 24: 0 none, 1 borrowed, 2 owned
};                       // 32 bytes total (4 bytes tail padding)
```

- Both identities are the `id` field of the `ffc_type_info_t` constant of the
  named type. Ids are dense and monotonic within one linked program, so they
  are stable for the life of that program and are only ever compared for
  equality. Id `0` is reserved and means "no type": a null descriptor reads as
  `data == 0, declared_type == 0, dynamic_type == 0, ownership == 0`.
- `declared_type` is the type written in the declaration and never changes for
  a given entity. `dynamic_type` is the type of the value currently stored and
  is what `select type` and type-bound dispatch consult. They are equal for a
  base value and differ exactly when the value is an extension.
- A descriptor with a null `data` address but an associated `dynamic_type` is
  invalid and is rejected when the descriptor is built; an unallocated or
  unassociated class entity is the fully null descriptor instead.
- `ownership` records who frees `data`. A class dummy borrows its actual
  argument's storage: the callee never frees it and the storage outlives the
  call. An allocatable class value owns its storage; releasing the descriptor
  yields the address to free exactly once and resets the descriptor, so a
  borrowed or already released descriptor yields a null address and no double
  free is possible.
- The descriptor never aliases the value: `data` points at storage held
  elsewhere, and copying a descriptor copies a reference, never the value.

## Type-bound dispatch vtables

Each derived type with type-bound bindings emits one const global,
`__ffc_vtable_<typename>`, holding one 8-byte code address per binding slot.
Slot `k` is the `k`-th binding of the type in declaration order, counting
inherited bindings first: an extension copies its parent's slots in order and
an override replaces the target of the slot it overrides, so slot `k` names the
same binding in a type and in every extension of it. A slot whose
implementation is not defined in the current compilation unit stays null rather
than emitting an undefined relocation.

One link-unit table, `__ffc_vtable_table`, maps a type identity to its vtable:
entry `i` is the address of the vtable of the type whose `ffc_type_info_t.id`
is `i`, and entry `0` is null for the reserved "no type" id. A type without
bindings has a null entry.

Dispatch through a polymorphic receiver is therefore

```
vtable = __ffc_vtable_table[descriptor.dynamic_type];
callee = vtable[slot - 1];
```

Both loads read const, linker-initialised data; neither writes memory nor
aliases the receiver's storage. Keeping the identity-to-vtable mapping in this
table rather than in a fifth descriptor field leaves the 32-byte scalar class
descriptor above unchanged. A receiver whose dynamic type is fixed at compile
time (a `type(t)` entity) keeps a direct call: its dynamic type is its declared
type by definition, so the vtable would only re-derive a known answer.

## Type size table

`__ffc_type_size_table` is a link-unit array of `i64` byte sizes indexed the
same way as `__ffc_vtable_table`: entry `i` is the exact storage size of the
type whose `ffc_type_info_t.id` is `i`, and entry `0` is the reserved zero.
Allocating a class value whose dynamic type is only known at run time reads its
concrete size from here, so the storage is the dynamic type's whole layout and
not the declared type's prefix.

## Scalar class allocatables

A `class(t), allocatable` scalar owns one class descriptor in its frame — the
same 32-byte descriptor above, no separate convention. Unallocated is the null
descriptor with `declared_type` already filled in and `data == 0`,
`dynamic_type == 0`, `ownership == 0`.

`ALLOCATE` picks the concrete dynamic type — from `SOURCE=` (the source's own
dynamic type, loaded from its descriptor when the source is itself
polymorphic), from an explicit type-spec, or the declared type — allocates that
type's exact size, copies the whole source value for `SOURCE=`, and stores
`data`, `dynamic_type`, and `ownership = 2 (owned)`.

`DEALLOCATE` finalizes the value, frees the storage once, and resets the
descriptor to null, so the released address is no longer reachable and
ownership is given up exactly once. The type-compatibility rule of F2018 C946
is enforced at compile time: a `SOURCE=` or type-spec type must be the declared
type or an extension of it.

Because this is the same descriptor a class dummy receives, `SELECT TYPE` and
type-bound dispatch consult it with no path of their own.

## Module artefact format

`ffc -c <source>.f90` writes one `<modulename>.fmod` next to the object file
for each module the source defines. The file records the module's exported
interface so a later unit can resolve `use <module>` without the source. It is
a line-oriented subset of TOML, with no source locations or comments.

```toml
[module]
name = "shapes"
ffc_version = "0.1.0"
fmod_schema = 12

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

- `[module]` carries the module name, the emitting `ffc` version, and the
  mandatory `fmod_schema`. Writers emit schema 12. Readers accept schema 12,
  schema 11, and the read-only legacy schema 10. They reject missing and unknown schema
  values with a request to recompile the module.
- Each `[[parameter]]` is a named constant: `name`, `kind` (the normalised
  scalar type token), and the literal `value`.
- Each `[[derived_type]]` is a type definition with its `components`, each a
  `{ name, kind }` pair. Allocatable array components additionally record
  `alloc_rank` (1 or 2); schema 11 components without that field are read as
  rank one for compatibility.
- Each `[[variable]]` records a module variable's Fortran `name`, scalar `kind`,
  and optional mangled `c_name`.
- Each `[[procedure]]` records an exportable module procedure's `name`, result
  `kind` (or `subroutine`), `nargs`, and space-separated scalar `arg_kinds`.
- Each `[[generic]]` records a generic `name` and the space-separated specific
  procedure names it resolves to.
- `kind` is `integer`, `real`, `logical`, `character`, or `type(<name>)`.
- A later unit reads these tables on `use`, emits external references to module
  variables and mangled procedures, and links them against the module object.
  `use, only:` names and rename targets are validated against the records; a
  local rename keeps the recorded remote name for storage and linkage.

## Runtime delivery

`runtime/ffc_runtime.c` is the single source of truth for the ffc runtime.
Every entry point defined there is listed in `ffc_runtime_link`'s
`FFC_RUNTIME_SYMBOLS` and documented under [Runtime entry points](#runtime-entry-points)
below. Two independent consumers read that file.

### Linked into every emitted executable

This is how a compiled program gets its runtime, and the only mechanism the
lowerer may rely on.

`src/ffc_runtime_source.f90` embeds `runtime/ffc_runtime.c` verbatim in the
compiler binary; regenerate it with `scripts/generate_runtime_source.sh` after
every edit to the C file. At link time `ffc_runtime_link` materialises the
embedded source into a content-addressed file under `TMPDIR` and passes it to
`lr_session_emit_exe_objects`, which hands it to the same system C compiler
that already performs the link. Every executable `ffc` emits therefore carries
a definition of every runtime entry point.

The contract this fixes (issue #565):

- **Unconditional.** There is no environment variable, no artifact to install,
  and no discovery step. An executable is never emitted without its runtime.
- **No inline fallback.** The lowerer emits calls to runtime symbols and
  nothing else. The parallel path that synthesised the same entry points
  inline is retired; ROADMAP Chunk 3 forbids reintroducing it.
- **Mismatch is impossible by construction.** The runtime that gets linked is
  the one compiled into the running compiler, so a compiler and its runtime
  cannot drift apart, and there is no stale installed copy to pick up.
- **Failure is loud.** When the runtime cannot be materialised, lowering fails
  with a named error and emits nothing. It never silently produces a binary
  that dies with `undefined symbol` at run time.
- **Object output is unaffected.** `ffc -c` leaves runtime symbols undefined
  in the object, to be resolved by the link that consumes it.

`test_runtime_link_compiler` is the oracle: it checks that a normally compiled
program's symbol table defines the runtime entry points, that the linked
runtime is callable, that every declared symbol is really defined by the
embedded source, and that the embedded source is byte-identical to
`runtime/ffc_runtime.c`.

### Packaged into LIRIC runtime archives

The standalone CMake project in `runtime/` also packages the same file for
sessions that resolve runtime calls without a system linker.
`runtime/ffc_runtime.c` is compiled to LLVM bitcode with `clang -emit-llvm -c`,
and `liric_runtime_archive` packages that bitcode once per backend.
`install_runtime_archive` reads and installs one; the artifact directory is an
explicit argument, and a missing, unreadable, or backend-mismatched archive is
an error. It is not on the executable-emission path.

### Artifact names

Archives are written to a target-qualified directory,
`<build>/artifacts/<target>/`, so archives for different targets never
collide. Within it the names are:

| Backend | Artifact |
|---|---|
| isel | `ffc-runtime-v2-isel.lrarch` |
| copy-patch | `ffc-runtime-v2-copy-patch.lrarch` |
| llvm | `ffc-runtime-v2-llvm.lrarch` (only when LLVM is available) |

`v2` is the artifact naming version. It is bumped when the artifact layout or
the packaged payload contract changes, and is independent of the LIRIC archive
format version recorded inside each file.

The LIRIC session backend `default` is an alias for copy-patch and has no
artifact of its own: a consumer asking for `default` selects
`ffc-runtime-v2-copy-patch.lrarch`.

The `llvm` artifact is produced only when the configuration opts in with
`-DFFC_RUNTIME_ENABLE_LLVM=ON` and the LIRIC build exposes the LLVM backend, so
a LIRIC build without LLVM never silently omits a requested artifact.

### Archive format

Each file is a LIRIC runtime archive. Its header is the 8-byte magic
`LRARCH1\0`, then little-endian `u32` format version, `u32` backend code,
`u32` target-name length, `u64` IR text length, and `u64` blob-package length.
The current format version is 2. The backend code is the LIRIC session backend
enumerator: 1 for isel, 2 for copy-patch, 3 for LLVM. Because the backend is
recorded in the file, two archives built for different backends are never
byte-identical.

### Payload

The archive carries the runtime's LIRIC IR text plus a compiled blob package,
built from the same `runtime/ffc_runtime.c` that is linked into executables.

### Runtime entry points

The complete runtime ABI. Adding an entry point means editing
`runtime/ffc_runtime.c`, regenerating the embedding, adding the symbol to
`FFC_RUNTIME_SYMBOLS`, and adding it here in the same change.

| Symbol | Signature | Contract |
|---|---|---|
| `_ffc_runtime_probe` | `int _ffc_runtime_probe(void)` | Returns 42. Lets a consumer confirm end to end that the runtime it linked or loaded really resolves. |
| `_ffc_unit_newunit` | `int _ffc_unit_newunit(void)` | Lowest free unit at or above 1000, above anything a program names explicitly. Returns -1 and sets status 5005 when none is free. |
| `_ffc_unit_open` | `int _ffc_unit_open(int unit, const char *path, int path_len, const char *status)` | Connects `unit`. `path_len` is the byte count of the Fortran `FILE=` value, whose trailing blanks are padding and are trimmed here. `status` is the current Fortran `STATUS=` character value; comparison is case-insensitive and ignores fixed-length trailing blanks. A null, empty, or all-blank `path`, or `status` `scratch`, connects a temporary file removed on close. |
| `_ffc_unit_is_open` | `int _ffc_unit_is_open(int unit)` | 1 when the unit is connected, 0 otherwise. Never fails. |
| `_ffc_unit_file` | `FILE *_ffc_unit_file(int unit)` | The stream behind a unit, connecting an unopened numeric unit to `fort.<N>` on first use. NULL only when the unit is unusable. |
| `_ffc_unit_rewind` | `int _ffc_unit_rewind(int unit)` | Repositions to the first record. |
| `_ffc_unit_close` | `int _ffc_unit_close(int unit)` | Disconnects the unit. Succeeds on a unit that is not connected. |
| `_ffc_inquire_file_size` | `long long _ffc_inquire_file_size(const char *path)` | Returns the filesystem byte size, or -1 when `path` cannot be stat'ed. |
| `_ffc_inquire_unit_size` | `long long _ffc_inquire_unit_size(int unit)` | Flushes and returns the connected stream's byte size while restoring its position, or -1 when the unit is unusable or not seekable. |
| `_ffc_unit_status` | `int _ffc_unit_status(void)` | Status of the most recent unit operation. |
| `_ffc_random_seed_size` | `int _ffc_random_seed_size(void)` | Size of the seed array `RANDOM_SEED(SIZE=)` reports. The generator behind `RANDOM_NUMBER` has one integer of state, so this is 1. |
| `_ffc_random_seed_put` | `void _ffc_random_seed_put(const int *seed)` | `RANDOM_SEED(PUT=)`. Restarts the generator from `seed[0]`, so an identical PUT replays an identical sequence. Null is ignored. |
| `_ffc_random_seed_get` | `void _ffc_random_seed_get(int *seed)` | `RANDOM_SEED(GET=)`. Writes the current seed into `seed[0]`. Null is ignored. |
| `_ffc_random_seed_default` | `void _ffc_random_seed_default(void)` | Argument-less `RANDOM_SEED()`. Resets the generator to the processor's default seed, which repeats across runs (permitted by F2018 16.9.155). |

### File-unit state (#396)

The runtime owns file-unit state: which units are connected, the `FILE*`
behind each, and the status of the last operation. It is per process and keyed
by the unit number the program computes at run time, so a unit opened inside a
procedure is still connected after that procedure returns. Before #396 the
compiler emitted one stack slot per unit inside the function that opened it,
which scoped a connection to a lowered function.

Units run from 0 through 2048. `NEWUNIT=` allocates from 1000 upwards and
releases the number on `CLOSE`.

Status codes are stable. They are the values `IOSTAT=` reports:

| Code | Meaning |
|---|---|
| 0 | success |
| 5001 | unit number outside the supported range |
| 5002 | operation on a unit that is not connected |
| 5003 | `OPEN` on a unit that is already connected |
| 5004 | the file could not be opened |
| 5005 | no free unit left for `NEWUNIT=` |

### Scalar formatted output (#423)

Scalar output goes through the runtime. The compiler decides the unit and the
C conversion descriptor its edit descriptor implies; the runtime owns the
stream lookup, the conversion, and the status. Output bytes are unchanged from
the `printf` calls these replaced.

There is one entry point per scalar type rather than one call carrying a type
tag as data, so the type is resolved at compile time and **the calls are not
variadic**: a fixed-arity ABI is the same on every target, a variadic one is
not. `_ffc_write_text` carries record text with nothing to convert — the
list-directed separating blank and the record terminator.

Each returns 0, or the unit status when the unit is unusable, or 5006 when the
conversion fails.

| Symbol | Signature |
|---|---|
| `_ffc_write_i32` | `int _ffc_write_i32(int unit, const char *fmt, int value)` |
| `_ffc_write_i64` | `int _ffc_write_i64(int unit, const char *fmt, long long value)` |
| `_ffc_write_f64` | `int _ffc_write_f64(int unit, const char *fmt, double value)` |
| `_ffc_write_str` | `int _ffc_write_str(int unit, const char *fmt, const char *value)` |
| `_ffc_write_text` | `int _ffc_write_text(int unit, const char *text)` |

`INTEGER(1)` and `INTEGER(2)` widen to `i32` before the call, as they did for
`printf`. Logical and character scalars use `_ffc_write_i32` and
`_ffc_write_str`. Complex output, list-directed input, NAMELIST, and internal
I/O still use their established paths; they are named by their own issues.

### Scalar unformatted output

A stream-opened unit may use list-directed syntax to write scalar integer or
logical values as unformatted bytes. The compiler selects the entry point from
the declared kind, and each function writes exactly one value with `fwrite`.

| Symbol | Signature |
|---|---|
| `_ffc_write_unformatted_i8` | `int _ffc_write_unformatted_i8(FILE *fp, signed char value)` |
| `_ffc_write_unformatted_i16` | `int _ffc_write_unformatted_i16(FILE *fp, short value)` |
| `_ffc_write_unformatted_i32` | `int _ffc_write_unformatted_i32(FILE *fp, int value)` |
| `_ffc_write_unformatted_i64` | `int _ffc_write_unformatted_i64(FILE *fp, long long value)` |

The functions return 0 on success and update the runtime I/O status on failure.

### Preconnected units

Unit 5 is standard input, unit 6 standard output, and unit 0 standard error.
They are never opened as `fort.<N>` and never closed, and `OPEN` on one of
them without `FILE=` reconfigures the existing connection rather than
replacing it, so `open(6, sign='plus')` keeps writing to standard output.

### IOSTAT= and IOMSG= (#427)

Fortran reports I/O outcome through `IOSTAT=` and `IOMSG=`. The classes are
fixed by the standard and by what programs test for:

| IOSTAT | Meaning |
|---|---|
| 0 | success |
| -1 | end of file (gfortran's `IOSTAT_END`) |
| -2 | end of record (gfortran's `IOSTAT_EOR`) |
| > 0 | an error; the unit status codes above |

| Symbol | Signature | Contract |
|---|---|---|
| `_ffc_iostat` | `int _ffc_iostat(void)` | Fortran status of the most recent I/O operation. |
| `_ffc_iostat_set_end` | `void _ffc_iostat_set_end(void)` | Record an end-of-file condition. |
| `_ffc_iostat_clear` | `void _ffc_iostat_clear(void)` | Record success. |
| `_ffc_iomsg` | `void _ffc_iomsg(char *dest, int len)` | Message for the recorded status. |

`_ffc_iomsg` writes with Fortran character assignment semantics: the text is
truncated to `len` and the remainder blank filled. It writes exactly `len`
characters plus a terminating NUL, so `dest` must have room for `len + 1` —
the compiler's character values are NUL-terminated buffers of the declared
length. After a successful operation the destination is left all blanks rather
than untouched, so it is always defined; the standard defines `IOMSG=` only
when an error or end-of-file condition occurs.

`OPEN`'s own return value is its `IOSTAT=`. A file `WRITE` and a `READ` report
the runtime's record of the operation. A `READ` that reaches end of file
records it in the runtime as well, so `IOSTAT=` and `IOMSG=` describe the same
condition rather than being derived independently.

`IOSTAT=` targets must be default integers and `IOMSG=` targets character
variables with a declared length; both are rejected with a named diagnostic
otherwise.

### Descriptor storage allocation (#428)

Allocatable arrays and deferred-length characters reached `malloc` and `free`
directly from emitted code, so every size computation, every overflow check,
and every ownership decision was open-coded at each site. The runtime owns
that now. The compiler still decides shape and type; the runtime decides
whether a size is representable, whether a pointer may be released, and what
the status is.

| Symbol | Signature |
|---|---|
| `_ffc_alloc` | `void *_ffc_alloc(long long count, long long elem_size)` |
| `_ffc_calloc` | `void *_ffc_calloc(long long count, long long elem_size)` |
| `_ffc_realloc` | `void *_ffc_realloc(void *old, long long count, long long elem_size)` |
| `_ffc_dealloc` | `int _ffc_dealloc(void *p, int owns)` |
| `_ffc_alloc_status` | `int _ffc_alloc_status(void)` |

Sizes arrive as a separate element count and element size, never as a
product, so the multiplication that can overflow happens in the runtime, once,
where it is checked.

A count of zero is a valid request: Fortran allows a zero-sized array, and the
result is a non-null pointer released exactly like any other. Releasing a null
pointer succeeds, matching `deallocate` of an unallocated variable.

`owns` is the descriptor's `ARRAY_FLAG_OWNS_DATA` bit. A borrowed descriptor —
a section view or a dummy argument — never frees, and the runtime reports
`6005` rather than doing nothing silently. See the ownership and view-lifetime
rules in `ARRAY_DESCRIPTOR_ABI.md`, which this enforces at run time.

The runtime tracks the allocations it hands out, so releasing a pointer twice
is reported instead of corrupting the heap. That is also why emitted code must
not mix allocators: storage from a direct `malloc` is unknown to the runtime,
and releasing it would be reported as a double free.
`test_session_runtime_allocation_helpers_compiler` fails if any
compiler-emitted function still calls `malloc` or `free`.

| Code | Condition |
|---|---|
| 0 | success |
| 6001 | negative count or element size |
| 6002 | `count * elem_size` is not representable |
| 6003 | the allocator refused |
| 6004 | release of a pointer that is not live |
| 6005 | release of storage the descriptor does not own |

## Building the runtime

`runtime/ffc_runtime.c` is compiled by whichever C driver links an emitted
executable, and by clang in `runtime/CMakeLists.txt`. It states its own
language level and feature set — `#define _XOPEN_SOURCE 700` before any
include — instead of inheriting the invoking driver's defaults, because
`random`/`srandom` are POSIX rather than ISO C and a strict driver would
otherwise reach them only through an implicit declaration: a warning on a
lenient toolchain, a hard error elsewhere.

It is C, and only C flags apply to it. It lives in `runtime/`, never under
`src/`, where fpm would compile it with the Fortran flag set (`-J`,
`-ffree-form`, `-fimplicit-none`). `runtime/CMakeLists.txt` spells its clang
flags out rather than inheriting them. `test_runtime_link_compiler` enforces
both: it fails if any C source appears under `src/`, and it fails unless the
runtime builds under `cc -std=c11 -Wall -Wextra -Werror -pedantic`.

### Dependencies

`clang` and `liric_runtime_archive` are both required. Either one missing fails
configuration with a named `ffc runtime dependency missing: <name>` diagnostic;
neither is silently skipped. The archive tool is located through
`-DLIRIC_BUILD_DIR=`, the `LIRIC_BUILD_DIR` environment variable, or a sibling
LIRIC checkout.

## Unsupported ABI Work

- #53: array descriptors, allocatables, and pointer representation.
- Broader external-procedure signatures and descriptor-bearing `.fmod`
  procedure entries.
- #55: runtime I/O beyond the current scalar `printf` shim.
