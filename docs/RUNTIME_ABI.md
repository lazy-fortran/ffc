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
- `main` returns `i32`.
- Falling off the end of `program main` returns zero.
- `stop <integer expression>` returns that integer expression as the process
  exit status.

## Scalar Values

- `integer` values use LIRIC `i32` values.
- `real` values use LIRIC `f64` values.
- `logical` values currently use the MVP `i32` representation: zero is false,
  nonzero is true. Printed logicals therefore use the same integer `printf`
  path and currently print `0` or `1`.
- Scalar `character(len=N)` variables keep an `i8*` pointer to literal-backed
  storage plus the declared length `N` in the lowering symbol. Assignment from
  character literals stores exactly `N` characters by truncating long literals
  and blank-padding short literals.
- The current lowerer keeps ordinary scalar symbols as SSA-like current values.
- Procedure reference arguments use LIRIC `alloca`/`load`/`store` slots at the
  call boundary.
- Scalar `abs`, `min`, and `max` intrinsics are supported for integer and real
  values. Integer-to-real `real()` conversion is supported. They lower inline
  through LIRIC scalar operations, comparisons, branch control, casts, and PHI
  values.

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
- Subroutines return LIRIC `void`; explicit `CALL` statements emit `void` calls.
- Names are emitted as source names for the current single-file subset. A
  deterministic mangling scheme is still required before broader procedure and
  module support.
- Character procedure parameters and results are unsupported.
- External procedures, modules, and separate compilation are unsupported; #54
  owns symbol mangling and link behavior.

## Runtime Calls

- Minimal scalar `print` lowers to an external C `printf` declaration.
- The current format globals are:
  - integer/logical: `%12d` (a trailing newline is emitted separately). The
    field width 12 matches gfortran's default list-directed `integer(4)`
    output, which right-justifies in 12 columns. `integer(8)` (width 22) and
    Fortran-aware logical (`T`/`F`) output are deferred to later issues.
  - real: `%f`
  - character: `%s`
- Character literal print passes a pointer to a null-terminated global byte
  array to `printf`. Scalar character variable print passes a pointer to a
  global byte array containing the fixed-length value followed by a null
  terminator. The current C `printf` shim consumes the terminator, not an
  explicit length argument. #55 owns the Fortran-aware I/O runtime.
- Object output may contain unresolved references such as `printf`; final
  linking is responsible for resolving the C runtime.
- The current `printf` path is the only supported I/O surface. #55 owns the
  Fortran-aware scalar I/O runtime.

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

## Unsupported ABI Work

- #53: array descriptors, allocatables, and pointer representation.
- #54: module and external symbol mangling.
- #55: runtime I/O beyond the current scalar `printf` shim.
