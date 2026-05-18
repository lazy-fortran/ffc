# ffc Architecture

`ffc` is the compiler driver. FortFront owns the frontend; LIRIC owns
native code generation. `ffc` lowers FortFront's typed AST into LIRIC
session calls.

## Pipeline

```
Fortran / Lazy Fortran source
  -> FortFront compiler API
  -> typed AST, semantic context, diagnostics
  -> ffc lowering + runtime ABI
  -> LIRIC session C API
  -> object file / executable
```

## Component boundaries

FortFront stays backend-neutral. It exposes:

- a typed AST arena and root index;
- semantic analysis state and tokens for diagnostics;
- public compiler-facing queries used by `ffc`.

`ffc` owns:

- lowering FortFront nodes to LIRIC instruction descriptors;
- Fortran ABI and runtime call decisions;
- program entry, object emission, and executable emission;
- CLI parsing and backend invocation;
- behavioural executable tests for every claimed feature.

LIRIC owns native code generation behind its session C API. `ffc` reaches
LIRIC through ISO C bindings only.

## Backend rule

New compiler work targets LIRIC's session C API. `ffc` does not add LLVM
bindings, MLIR bindings, HLFIR, or text-IR compiler paths.

## Capability order

Each step must leave `ffc` able to compile and run at least the previous
supported subset. The current supported surface is in
`docs/SUPPORT_CONTRACT.md`; broadly:

1. `program main`, integer scalars, arithmetic, `stop`.
2. Minimal `print *, expr` for integers, reals, logicals, characters.
3. Block `if` with PHI merges; counted `do` with literal step.
4. `SELECT CASE` with terminating arms.
5. Contained integer / real / logical functions and subroutines.
6. Fixed-size 1-D integer arrays; simple derived types with scalar
   integer components.
7. Deferred-length character with assignment, concatenation including
   self-aliasing, and `len()` queries.

Arrays beyond fixed-size, modules and separate compilation, polymorphism,
type-bound procedures, allocatables, the full intrinsic set, and a
Fortran-aware I/O runtime are unsupported and tracked as GitHub issues.

## Runtime and ABI decisions

The current ABI is documented in `docs/RUNTIME_ABI.md`. Before broadening
language coverage, document and test:

- program entry and exit-status convention;
- name mangling;
- pass-by-reference vs pass-by-value;
- scalar return values and function result variables;
- logical and character representation, character length passing;
- array descriptor shape;
- I/O runtime call surface.

## Performance direction

The direct LIRIC session path replaces text-IR generation on the hot
path. The expected wins are fewer string allocations, no parse-back step
for generated text IR, and a cleaner road to incremental compilation.
Measurement is deferred until the supported surface is wider; today the
priority is correctness and coverage.
