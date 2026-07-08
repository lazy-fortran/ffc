# ffc API Reference

`ffc` is the compiler driver between FortFront and LIRIC. The active API
surface is intentionally small. The supported language contract is
`docs/SUPPORT_CONTRACT.md`.

## FortFront input

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

## LIRIC session API

New compiler work uses `liric_session_bindings`:

```fortran
use liric_session_bindings, only: liric_session_t, liric_session_create, &
                                  lr_operand_desc_t
```

The direct session path:

1. Creates a LIRIC session.
2. Begins a `main` function (or a contained function/subroutine).
3. Lowers FortFront AST nodes into LIRIC instruction descriptors.
4. Emits a native executable or object file through LIRIC.

`session_program_lowering` is the current lowering entry point:

```fortran
use session_program_lowering, only: lower_program_to_liric_exe, &
                                    lower_program_to_liric_object
```

## CLI

`app/ffc.f90` invokes `ffc_cli_options.parse_arguments` to parse argv.
Today the CLI accepts:

- positional: input source file.
- `-o <file>`: output path. Default is `a.out` (or `a.o` with `-c`).
- `-c`: emit an object file instead of an executable.
- `-I <dir>`: append a module search directory. Repeatable. Used to
  locate `.fmod` artefacts on `use`.
- `--backend default|isel|copy-patch|llvm`: select the LIRIC backend.
- `--json`: emit failed compiler diagnostics as JSON on stderr.

## Supported direct-session subset

`docs/SUPPORT_CONTRACT.md` is authoritative. Briefly:

- empty `program main`; scalar integer / real / logical / character;
- arithmetic, comparisons; block `if`; counted `do` with literal step
  (positive or negative); `SELECT CASE` with terminating arms;
- contained integer / real / logical functions and subroutines, plus
  early `return`;
- fixed-size 1-D integer arrays; simple derived types with scalar
  integer components;
- deferred-length character with assignment, concatenation including
  self-aliasing, and compile-time literal `//` folding;
- minimal `print *, expr`, `stop <expr>`, scalar `abs`/`min`/`max`/`mod`
  and integer-to-real `real()`;
- object and executable emission.

All other constructs are unsupported unless `docs/SUPPORT_CONTRACT.md`
lists them as supported.

## Runtime ABI

The current direct-session runtime and scalar ABI is documented in
`docs/RUNTIME_ABI.md`. Each new feature that changes layout or calling
convention updates that document in the same change.
