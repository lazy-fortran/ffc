# LIRIC C API Usage

The active backend integration uses LIRIC through `iso_c_binding`.

## Binding modules

- `liric_session_bindings`: direct session construction,
  function/subroutine emission, instruction emission, object output, and
  executable output.
- `liric_session_control_bindings`: block, branch, comparison, and PHI
  helpers.
- `liric_session_procedure_bindings`: function/subroutine begin/finish
  and call helpers.
- `liric_session_io_bindings`: minimal `printf` shim and string
  materialisation used for `print *`.
- `liric_session_arrays.inc`: array alloca and element-address helpers
  bundled into `liric_session_bindings`.

New lowering code targets these modules. It does not target LLVM, MLIR,
HLFIR, or text IR.

## Session Flow

```fortran
type(liric_session_t) :: session
character(len=:), allocatable :: error_msg

call liric_session_create(session, error_msg)
if (len_trim(error_msg) > 0) return

if (.not. begin_i32_main(session, error_msg)) return
! emit instructions
if (.not. finish_and_emit_exe(session, output_path, error_msg)) return
! or: if (.not. finish_and_emit_object(session, output_path, error_msg)) return

call destroy(session)
```

Every call that can fail returns a diagnostic string. Callers must stop lowering
on the first error and destroy any open session.

## Ownership

`ffc` owns the Fortran lowering and ABI choices. LIRIC owns native code
generation and executable emission. FortFront owns parsing, AST, semantic data,
diagnostics, and source mapping.

## Do not add

- New MLIR/HLFIR bindings.
- Backend code in FortFront.
- Text IR as a compiler path.

The retired MLIR/HLFIR experiment lives only in git history.
