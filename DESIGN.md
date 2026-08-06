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

`session_program_lowering` is the public lowering facade. It exports only
`lower_program_to_liric_exe` and `lower_program_to_liric_object` from the
private-by-default `session_program_lowering_impl` module. Extracted lowering
units are descendants of the implementation module and use its explicit
descendant API. This keeps compiler clients independent of the internal split
and gives GCC 14 external symbols for procedures used across submodule object
boundaries.

New lowering units are modules or submodules with explicit interfaces. Do not
add production `include` fragments; the existing fragments are migration debt.

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

## Symbol identity

Lowering symbols are keyed by FortFront binding identity, not by text
(`src/session_symbol_table.f90`, #327). FortFront resolves a name
reference to a `declaration_binding_t`; the triple
`(declaration_node_index, declaration_entity_index, scope_node_index)`
identifies the declared entity, and the binding table maps that identity
onto a slot in the lowering context's symbol array. Text names remain in
`symbol_t` for diagnostics and mangling only.

This keeps ownership clean: FortFront alone applies Fortran's shadowing,
host-association, USE and accessibility rules, and `ffc` alone owns
storage and ABI metadata. `ffc` never synthesises a symbol for a name
FortFront could not resolve — an unresolved reference keeps the
undeclared-name diagnostic.

Reference sites still fall back to the historical text lookup when a
reference has no FortFront binding, because `ffc` also creates symbols
with no declaration behind them (inferred lazy-Fortran locals, DO
variables, ABI temporaries). Retiring that fallback is the work of the
remaining scope issues.

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
