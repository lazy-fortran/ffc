# ffc API Reference

`ffc` is the compiler driver between FortFront and LIRIC. The active API
surface is intentionally small.

## FortFront Input

Use FortFront's compiler-facing API:

```fortran
use fortfront, only: compiler_frontend_options_t, compiler_frontend_result_t, &
                     compile_frontend_from_file, INPUT_MODE_STANDARD
```

The frontend result owns:

- `arena`: FortFront AST arena.
- `root_index`: top-level program node index.
- `semantic_ctx`: semantic analysis state.
- `tokens`: token stream retained for diagnostics and source mapping.
- `error_msg` and `diagnostic_text`: parse and semantic diagnostics.

FortFront remains backend-neutral. It must not depend on LIRIC.

## LIRIC Session API

New compiler work should use `liric_session_bindings`:

```fortran
use liric_session_bindings, only: liric_session_t, liric_session_create, &
                                  lr_operand_desc_t
```

The direct session path:

1. Creates a LIRIC session.
2. Begins a `main` function.
3. Lowers FortFront AST nodes into LIRIC instruction descriptors.
4. Emits a native executable or object file through LIRIC.

`session_program_lowering` is the current lowering entry point:

```fortran
use session_program_lowering, only: lower_program_to_liric_exe, &
                                    lower_program_to_liric_object
```

## Current Supported Direct-Session Subset

- Empty `program main`.
- Object and executable emission for the supported direct-session subset.
- Scalar integer declarations and assignments.
- Integer literals and arithmetic.
- Integer comparison `if` blocks with terminating branches or mergeable
  assignments.
- Literal-bound counted `do` loops by scalar expansion.
- `stop` with integer expression.
- Minimal scalar `print` for integer expressions, real values, character
  literals, and logical literals.
- Scalar real declarations, assignments, arithmetic, and printed variables.
- Scalar logical declarations, assignments, `if (flag)`, and printed
  variables.
- Simple contained integer functions and subroutines with integer parameters
  and integer call expressions/statements.

## Bootstrap Reference Path

`liric_bindings` and `empty_program_lowering` are temporary executable reference
coverage. They feed textual low-level IR into LIRIC's compiler API. Do not add
new language coverage there unless it is needed to compare behavior while the
direct session path catches up.

## Runtime ABI

The current direct-session runtime and scalar ABI is documented in
`docs/RUNTIME_ABI.md`.

## Legacy Source Tree

The old MLIR/HLFIR source tree under `src/` is outside the default fpm build.
It is reference material only. The active library source directory is
`src_mvp/`.
