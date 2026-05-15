# LIRIC C API Usage

The active backend integration uses LIRIC through `iso_c_binding`.

## Binding Modules

- `liric_session_bindings`: direct session construction, function emission,
  instruction emission, and executable output.
- `liric_session_control_bindings`: block, branch, comparison, and PHI helpers.
- `liric_bindings`: temporary compiler API bridge used by reference tests.

New lowering code should target `liric_session_bindings`.

## Session Flow

```fortran
type(liric_session_t) :: session
character(len=:), allocatable :: error_msg

call liric_session_create(session, error_msg)
if (len_trim(error_msg) > 0) return

if (.not. session%begin_i32_main(error_msg)) return
! emit instructions
if (.not. session%finish_and_emit_exe(output_path, error_msg)) return
! or: if (.not. session%finish_and_emit_object(output_path, error_msg)) return

call session%destroy()
```

Every call that can fail returns a diagnostic string. Callers must stop lowering
on the first error and destroy any open session.

## Ownership

`ffc` owns the Fortran lowering and ABI choices. LIRIC owns native code
generation and executable emission. FortFront owns parsing, AST, semantic data,
diagnostics, and source mapping.

## Do Not Add

- New MLIR/HLFIR bindings.
- Backend code in FortFront.
- Text IR as the main compiler path.

The only text-IR code left is temporary reference coverage while the direct
LIRIC session path reaches feature parity for the MVP subset.
