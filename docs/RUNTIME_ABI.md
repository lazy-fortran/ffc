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
  and blank-padding short literals.
- The current lowerer keeps ordinary scalar symbols as SSA-like current values.

### Allocatable array descriptor

An `integer, allocatable :: a(:)` declaration lowers to a 24-byte,
8-byte-aligned descriptor on the stack, zero-initialised at declaration:

```
struct ffc_alloc_i32_1d {
    i32*  data;    // offset 0; 0 means unallocated
    i64   lower;   // offset 8; the lower bound (1 once allocated)
    i64   upper;   // offset 16; the upper bound (0 when unallocated)
};
```

`data == 0` marks the array unallocated. `allocate(a(N))` (N a literal or
runtime integer) calls `malloc(N*4)`, stores the pointer in `data`, sets
`lower = 1` and `upper = N`. `deallocate(a)` calls `free(data)` then zeroes
`data` and `upper`. `free(NULL)` is a no-op, so deallocating an unallocated
variable exits cleanly rather than erroring (a deliberate divergence from the
standard, which makes it a runtime error). Element access and whole-array
assignment on allocatables are not yet supported and are rejected with a
diagnostic. Only single-variable, one-dimensional, default-lower-bound
`allocate`/`deallocate` are supported.

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
- Character procedure parameters and results are unsupported.
- External procedures, modules, and separate compilation are unsupported; #54
  owns symbol mangling and link behavior.

## Runtime Calls

- Scalar `print` lowers to external C `printf`/`snprintf` calls.
- List-directed record layout: one separating blank is written before every
  value. The first blank also serves as the record's carriage control. No
  blank is written between two consecutive character values, so they print
  concatenated (matching gfortran). Each value field below carries no leading
  blank of its own; a trailing newline closes the record.
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
record newline at the end. The supported edit-descriptor subset is a single
descriptor applied to every item:

- `I0` maps to `%d`, `Iw` to `%wd` (for example `I5` is `%5d`).
- `A` maps to `%s`; `Aw` maps to `%ws`. Character items pass a pointer to a
  null-terminated buffer, so `%s` prints the variable's full declared width.

Compound formats (`'(I0, " ", A)'`), repeat counts, and other edit descriptors
are rejected with a diagnostic. Because ffc lowers every `real` as `real(8)`,
no real edit descriptors are accepted yet. `print *, ...` remains the
list-directed path described above.

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
