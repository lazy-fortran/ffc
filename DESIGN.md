# ffc Architecture

`ffc` is the compiler driver. FortFront owns the frontend; LIRIC owns native
code generation. `ffc` connects them.

## Active Pipeline

```
Fortran / Lazy Fortran source
        |
        v
FortFront compiler API
        |
        v
typed AST, semantic context, diagnostics
        |
        v
ffc lowering and runtime ABI
        |
        v
LIRIC session C API
        |
        v
object file / executable
```

FortFront must stay backend-neutral. It should expose stable compiler-facing
queries for typed AST, symbols, scopes, diagnostics, and source mapping. It
should not know about LIRIC, object formats, runtime ABI, or executable linking.

`ffc` owns:

- lowering FortFront nodes to backend operations
- Fortran ABI and runtime call decisions
- program entry and object/executable emission
- compiler flags and backend selection
- feature-level executable tests

## Backend Rule

Use LIRIC through ISO C bindings to its C API.

The old MLIR/HLFIR source tree remains as legacy reference material outside the
default fpm build. Do not expand that path unless the backend decision is
reopened.

## Current Backend State

Two LIRIC paths exist:

- `liric_bindings`: bootstrap compiler API binding. It feeds generated text IR
  to LIRIC's compiler API. This proves end-to-end execution, but it is not the
  target architecture.
- `liric_session_bindings`: direct session API binding. It emits LIRIC
  instruction descriptors through `lr_session_emit()` and can build a runnable
  executable directly.

New lowering work should target `liric_session_bindings`. Keep the bootstrap
path only until the direct session path has equivalent executable coverage.

## Capability Order

Each step must leave `ffc` able to compile and run at least the previous
supported subset.

1. Empty `program main`.
2. Scalar integer literals, variables, assignment, arithmetic, comparison.
3. Minimal `print *, expr` for integers, reals, logicals, and strings.
4. Block `if`, fallthrough integer merges, and literal-bound counted `do`.
5. Simple contained integer functions with explicit ABI tests.
6. Runtime counted `do`, subroutines, character representation, and a fuller
   print/runtime surface.
7. Arrays, modules, derived types, allocatables, and generics.

## Runtime And ABI Decisions

The current MVP ABI is documented in `docs/RUNTIME_ABI.md`. Before broadening
language coverage, keep documenting and testing:

- program entry and exit status convention
- name mangling
- pass-by-reference versus value passing
- scalar return values and function result variables
- logical representation
- character storage and length passing
- array descriptor shape
- I/O runtime call surface

## Performance Direction

Direct LIRIC session lowering should replace text generation on the hot path.
The expected performance wins are fewer string allocations, no parse-back step
for generated text IR, and a cleaner path to incremental compilation.

Measure compile latency against the bootstrap path once the direct session
lowerer has runtime loops, procedures, and the same scalar executable coverage
as the bootstrap path.
