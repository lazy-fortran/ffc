# ffc Support Contract

This document is the public contract for the active `ffc` compiler path.
A program is supported only when every construct it uses appears in
"Supported now". Everything else is unsupported until the linked issue is
closed with executable tests.

## Architecture boundary

- FortFront owns lexing, parsing, AST storage, semantic analysis, type
  inference, diagnostics, and the compiler-facing frontend query APIs.
- `ffc` owns lowering from FortFront semantic data to LIRIC, the runtime
  ABI, diagnostics produced during lowering, object emission, executable
  emission, and behavioural tests for every claimed feature.
- LIRIC owns native code generation behind its session C API.
- New `ffc` compiler work uses LIRIC session C APIs through
  `iso_c_binding`.
- New `ffc` compiler work does not use LLVM bindings, MLIR bindings,
  HLFIR, or text-IR compiler paths.

## Supported now

| Area | Supported contract |
| --- | --- |
| Input | Standard Fortran source parsed by FortFront's compiler-facing API. Lazy Fortran and Infer-mode inputs are accepted only when FortFront produces the same supported semantic constructs. |
| Output | Native executable or object file emitted by LIRIC. |
| Program unit | One main program plus contained procedures. |
| Scalars | Scalar `integer`, `real`, and `logical` declarations and assignments. Compound declarations (`integer :: a, b, c`). |
| Characters | Character literals, fixed-length `character(len=N)` variables, deferred-length `character(len=:), allocatable` with assignment, concatenation including self-aliasing and three-way concat, and compile-time `//` folding for character literal chains. |
| Expressions | Integer and real literals, scalar variable references, integer and real arithmetic, integer comparisons, scalar logical conditions. |
| Termination | Falling off `program main` returns zero. `stop <integer expression>` returns the integer expression as process status. |
| `if` | Integer comparison `if` blocks with terminating branches or branches that assign mergeable scalar values; scalar logical `if (flag)` conditions. Array-element writes inside `if` branches are recognised even when the FortFront semantic flag is missing. |
| `do` | Counted `do` loops with integer bounds and literal integer step (positive or negative). |
| `select case` | Single-arm or multi-arm `SELECT CASE` with literal integer labels, ranges, character labels, multi-label arms (`case (1, 5)`), and `case default`. Arms may terminate (`stop`/`return`) or fall through to a merge after the construct; non-terminated arms carry mergeable scalar values via an N-way PHI. |
| Procedures | Contained integer, real, and logical functions and subroutines, including early `return` inside each. Scalar procedure actual arguments use pointer parameters with copy-back for variable actuals. |
| Intrinsics | Scalar `abs`, `min`, `max`, and `mod` for integer values; `abs`, `min`, `max` for real values; integer-to-real `real()` conversion. These lower inline with LIRIC scalar operations and control-flow PHI values. `command_argument_count()` and `get_command_argument(i, value)` are supported in the main program (no optional `length`/`status`). |
| `print` | Scalar `print *, expr` for integer expressions, real values, character literals, character variables, logical literals, real variables, and logical variables. List-directed output matches gfortran byte-for-byte, including `real(8)` G-format (fixed and exponential forms, `Infinity`/`NaN`), `integer(4)` width, `T`/`F` logicals, and the inter-item separator rules. Every Fortran `real` is lowered as `real(8)`. Formatted `print fmt, items` with a literal format string is supported for a single `I`/`A` edit descriptor (`I0`, `Iw`, `A`, `Aw`) applied to each item; compound formats and real descriptors are rejected. |
| Arrays | Fixed-size one-dimensional integer arrays with compile-time integer bounds. Scalar integer parameters and explicit `lower:upper` bounds are supported. Element assignment, element reads, `print`, `stop`, counted-loop subscripts, and whole-array assignment from an array constructor (`a = [e1, e2, ...]`, element count must match the declared size) are supported. Runtime bounds checks are not emitted. |
| Derived types | Simple program-scope derived type definitions with scalar integer components. Scalar variables of those types are supported. Component assignment, component reads, `print`, and `stop` are supported. Constructors, inheritance, type parameters, type-bound procedures, nested derived types, derived-type arrays, whole-derived assignment, and non-integer components are unsupported. |
| CLI | `-o <file>`, `-c`, `-I <dir>` (the include directories are stored but not yet consumed by lowering). |

## Unsupported categories

The full open issue list is on GitHub. Major categories tracked by long-
running tracker issues:

| Issue | Category |
| --- | --- |
| #53 | Allocatable arrays — descriptor layout, lifecycle, bounds. Slices: #184–#186. |
| #54 | Modules and separate compilation, including `.fmod` artefacts. Slices: #187–#190. |
| #55 | Fortran-aware scalar I/O runtime replacing the `printf` shim. |
| #56 | Generalised control-flow value merging (SELECT CASE fallthrough #180, derived-type IF merge #175, deferred-character IF merge #176). |
| #57 | Source locations and structured diagnostics. |
| #58 | FortFront compiler query boundary — replace arena reach-ins with public queries. |
| #59 | Standard- and Infer-mode conformance against the fortfront example corpus. |
| #167 | Self-hosting roadmap and dependency map. |
| #173 | Upstream: FortFront compiler-API queries needed by the self-hosting roadmap. |
| #174 | Upstream: LIRIC C-API capabilities needed by the self-hosting roadmap. |

Unsupported features must fail during parsing, semantic analysis, or
lowering with a diagnostic. Silent partial lowering is a bug.

Generic specialization and cross-module inference wait for FortFront and
package-level orchestration contracts.

## Feature admission rule

A feature becomes supported only after all of these are true:

1. The FortFront API surface used by `ffc` is public and documented.
2. Any new runtime ABI is documented in `docs/RUNTIME_ABI.md`.
3. The direct LIRIC session lowerer implements the feature.
4. Behavioural tests cover success behaviour and relevant diagnostics.
5. The support contract, README, and roadmap are updated in the same
   change.
