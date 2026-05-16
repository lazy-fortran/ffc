# ffc Support Contract

This document is the public contract for the active `ffc` compiler path. A
program is supported only when every construct it uses appears in "Supported
now". Everything else is unsupported until the linked issue is closed with
executable tests.

## Architecture Boundary

- FortFront owns lexing, parsing, AST storage, semantic analysis, type
  inference, diagnostics, and compiler-facing frontend query APIs.
- `ffc` owns lowering from FortFront semantic data to LIRIC, the runtime ABI,
  diagnostics produced during lowering, object emission, executable emission,
  and executable tests for every claimed feature.
- LIRIC owns native code generation behind its C API.
- New `ffc` compiler work uses LIRIC session C APIs through `iso_c_binding`.
- New `ffc` compiler work does not use LLVM bindings, MLIR bindings, HLFIR, or
  text IR as a compiler path.

## Supported Now

The current implementation is a single-file, direct-session scalar subset.

| Area | Supported contract |
| --- | --- |
| Input | Standard Fortran source parsed by FortFront's compiler-facing API. Lazy Fortran and Infer-mode inputs are accepted only when FortFront produces the same supported semantic constructs. |
| Output | Native executable or object file emitted by LIRIC. |
| Program unit | One main program. |
| Scalars | Scalar `integer`, `real`, and `logical` declarations and assignments. |
| Expressions | Integer and real literals, scalar variable references, integer and real arithmetic, integer comparisons, and logical conditions from scalar logical values. |
| Termination | Falling off `program main` returns zero. `stop <integer expression>` returns the integer expression as process status. |
| `if` | Integer comparison `if` blocks with terminating branches or branches that assign mergeable integer values; scalar logical `if (flag)` conditions. |
| `do` | Counted `do` loops with integer bounds and literal integer step. |
| Procedures | Contained integer functions and subroutines with integer parameters only. Integer procedure actual arguments use pointer parameters with copy-back for variable actuals. |
| Intrinsics | Scalar `abs`, `min`, and `max` for integer and real values, plus integer-to-real `real()` conversion. These lower inline with LIRIC scalar operations, comparisons, branches, and PHI operations. |
| `print` | Minimal scalar `print *, expr` through the current `printf` shim for integer expressions, real values, character literals, logical literals, real variables, and logical variables. |

## Unsupported Work

| Issue | Missing feature | Required result before support is claimed |
| --- | --- | --- |
| #50 | Non-integer scalar procedure ABI | Real, logical, and character scalar function/subroutine arguments and results have ABI tests and executable tests. |
| #51 | Scalar character variables and ABI | Character variables have a documented value-plus-length representation and executable behavior. |
| #52 | Fixed-size one-dimensional arrays | Declaration, indexing, assignment, and printing tests pass for explicit-shape rank-one arrays. |
| #53 | Allocatable arrays | Descriptor layout, allocation, deallocation, bounds, and element access are documented and tested. |
| #54 | Modules and separate compilation | Module symbols, external procedure symbols, name mangling, and link behavior are deterministic and tested. |
| #55 | Fortran-aware scalar I/O runtime | The `printf` shim is replaced for supported scalar output semantics. |
| #56 | General control-flow value merging | Non-terminating branches merge all supported scalar value kinds, not only the current integer subset. |
| #57 | Source locations and diagnostics | Unsupported features fail with targeted diagnostics containing source locations. |
| #58 | FortFront compiler query boundary | `ffc` no longer depends on FortFront arena internals for lowering decisions. |
| #59 | Standard and Infer-mode conformance tests | Supported constructs have reference behavior tests against the intended standard and Infer-mode semantics. |
| #60 | Derived types and component access | Simple derived type declarations, scalar components, assignment, and access are lowered and tested. |

Unsupported features must fail during parsing, semantic analysis, or lowering
with a diagnostic. Silent partial lowering is a bug.

Generic specialization and cross-module inference are out of scope for `ffc`
until FortFront and package-level orchestration contracts exist.

## Feature Admission Rule

A feature becomes supported only after all of these are true:

1. The FortFront API surface used by `ffc` is public and documented.
2. Any new runtime ABI is documented in `docs/RUNTIME_ABI.md`.
3. The direct LIRIC session lowerer implements the feature.
4. Executable tests cover success behavior and relevant diagnostics.
5. This support contract, `README.md`, and `ROADMAP.md` are updated in the
   same change.
