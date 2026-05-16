# Runtime ABI

This document records the current direct-session MVP ABI. It is an internal
compiler ABI for the current supported subset, not a stable external ABI.
Changes require matching executable tests and updates to
`docs/SUPPORT_CONTRACT.md`.

## Stability Rule

Only the representations listed in this document are supported. All other
values, arguments, descriptors, and runtime calls are unsupported until their
issue is closed with ABI documentation and executable tests.

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
- The current lowerer keeps ordinary scalar symbols as SSA-like current values.
- Procedure reference arguments use LIRIC `alloca`/`load`/`store` slots at the
  call boundary.
- Scalar character variables are unsupported; #51 owns their value-plus-length
  ABI.
- Scalar `abs`, `min`, and `max` intrinsics are supported for integer and real
  values. Integer-to-real `real()` conversion is supported. They lower inline
  through LIRIC scalar operations, comparisons, branches, casts, and PHI values.

## Procedures

- The supported procedure slice is contained integer functions and subroutines
  with integer parameters.
- Procedure parameters are currently lowered as LIRIC pointer parameters for
  integer arguments. Callers pass a reference slot; variable actual arguments
  are copied back after the call, and parameter assignment stores through the
  pointer.
- Function results are represented by assignment to the function result name.
- Subroutines return LIRIC `void`; explicit `CALL` statements emit `void` calls.
- Names are emitted as source names for the current single-file subset. A
  deterministic mangling scheme is still required before broader procedure and
  module support.
- Non-integer procedure parameters and results are unsupported; #50 owns that
  ABI.
- External procedures, modules, and separate compilation are unsupported; #54
  owns symbol mangling and link behavior.

## Runtime Calls

- Minimal scalar `print` lowers to an external C `printf` declaration.
- The current format globals are:
  - integer/logical: `%d\n`
  - real: `%f\n`
  - character literal: `%s\n`
- Character literal print materializes a global null-terminated byte array and
  passes a pointer to `printf`.
- Object output may contain unresolved references such as `printf`; final
  linking is responsible for resolving the C runtime.
- The current `printf` path is the only supported I/O surface. #55 owns the
  Fortran-aware scalar I/O runtime.

## Unsupported ABI Work

- #50: non-integer pass-by-reference arguments and richer procedure
  signatures.
- #51: character value plus length passing.
- #52: fixed-size one-dimensional array lowering.
- #53: array descriptors, allocatables, and pointer representation.
- #54: module and external symbol mangling.
- #55: runtime I/O beyond the current scalar `printf` shim.
