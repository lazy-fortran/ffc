# ffc Plan

## Current Reality

`ffc` now has a minimal working compiler path for the direct-session scalar
MVP. The exact support claim is in `docs/SUPPORT_CONTRACT.md`.

The old plan assumed an HLFIR/MLIR-first backend. The source tree still
contains that experiment, but it is legacy and not part of the default fpm
build.

The active plan is FortFront frontend, `ffc` lowering/runtime, and LIRIC
backend.

## Architecture Decision

Do not put LIRIC in FortFront.

- FortFront owns lexing, parsing, semantic analysis, transformation, and
  compiler-facing frontend APIs.
- `ffc` owns lowering, ABI/runtime decisions, backend calls, object emission,
  and executable emission.
- LIRIC is accessed from `ffc` through ISO C bindings to the C API.
- `ffc` does not add LLVM or MLIR compiler paths.
- `ffc` does not add HLFIR or text-IR compiler paths.

## Phase 0: Retire the Misleading MLIR Default

Goal: make the repository honest and buildable for the new backend direction.

Tasks:

- Done: default fpm build points at `src_mvp/`, not the old MLIR tree.
- Done: default manifest links LIRIC instead of the old backend libraries.
- Done: CLI source path calls FortFront's compiler API.
- Tracked: decide whether to move legacy MLIR code under an explicit archive
  directory or keep it as reference source outside the default build.

Verification:

- `LIBRARY_PATH=<liric-build> fpm build`
- CLI parses a file through FortFront and reports diagnostics instead of using a
  hardcoded root index.
- Done: CLI emits executables and object files through the direct LIRIC session
  path for the currently supported direct-session subset.

## Phase 1: FortFront Compiler Boundary

Goal: consume a stable typed-AST frontend result.

Required FortFront contract:

- input source or file path
- output arena/root index
- semantic diagnostics
- semantic/type data usable by lowering
- no forced standard Fortran code emission

`ffc` uses public FortFront APIs only. A need for private AST layout is tracked
as FortFront API work, not as an `ffc` workaround.

Verification:

- Done in FortFront: `compile_frontend_from_string/file` returns arena,
  root index, semantic context, tokens, and diagnostics.
- Tracked by #58: add narrower compiler query APIs so `ffc` does not depend on
  arena representation details.

## Phase 2: Direct LIRIC Session Backend

Goal: keep the compiler path on direct LIRIC session calls.

Current supported language subset:

- main program
- scalar integer/real/logical literals
- scalar declarations
- assignment
- arithmetic
- comparisons
- `if`
- counted `do`
- simple function/subroutine calls
- scalar `abs`, `min`, and `max` intrinsics for integer and real values, plus
  integer-to-real `real()` conversion
- minimal `print` runtime call

Verification:

- Done: compile and run `program main; end program`
- Done: compile and run integer declaration, assignment, arithmetic, and
  `print *, x` with known output
- Done: compile and run one-line `if (x == 5) print *, 1`
- Done: compile and run counted `do i = 1, 3` with known output
- Done: compile and run `print *, 2.5`
- Done: compile and run `print *, "hello"`
- Done: compile and run `print *, .true.`
- Done: compile and run real variables/arithmetic
- Done: compile and run block `if`
- Done: compile and run scalar integer/real `abs`, `min`, and `max`
  intrinsics and integer-to-real `real()` conversion
- Done: compile and run simple contained scalar integer, real, and logical
  functions and subroutines.
- Tracked by #55: fuller print/runtime surface.
- Tracked by #59: compare output against a reference compiler for the
  supported subset.

Tasks:

- Done: add ISO C bindings for the initial `lr_session_*` surface
- Done: emit and run an executable through direct LIRIC session calls
- Done: lower empty `program main` to a direct LIRIC session executable
- Done: lower integer literal/binary arithmetic stop codes to direct LIRIC
  vregs and return values
- Done: lower integer declarations and assignments to direct-session values
- Done: use direct session lowering from the CLI for the supported subset
- Done: lower terminating integer comparison `if` blocks to LIRIC blocks
- Done: lower fallthrough integer comparison `if` blocks that merge assigned
  integer values before a later `stop`
- Done: lower counted `do` loops with runtime-computed integer bounds through
  direct-session blocks and PHI backedges
- Done: map current integer, real, and logical scalar values to LIRIC types
- Done: lower integer, real, and logical expressions to direct-session values
- Done: lower scalar integer, real, character literal, and logical literal
  `print` through the current `printf` ABI shim
- Done: lower logical declarations, assignment, printed logical variables, and
  `if (flag)` conditions
- Done: lower simple contained integer functions and integer call expressions
- Done: lower simple contained integer subroutines and explicit `CALL`
  statements
- Done: lower integer procedure arguments through pointer parameters with
  copy-back for variable actual arguments
- Done: lower simple contained real and logical functions and subroutines with
  scalar parameters and results
- Done: lower integer and real scalar `abs`, `min`, and `max` intrinsics, plus
  integer-to-real `real()` conversion, inline through direct-session scalar ops
  and PHI values
- Tracked by #56: generalize non-terminating blocks/control flow with merge
  values.
- Done: emit objects directly from the session
- Done: remove the bootstrap text-IR reference path from `src_mvp/` and
  `test_mvp/`
- Done: document the current direct-session MVP ABI in `docs/RUNTIME_ABI.md`

Verification:

- same executable tests as Phase 2
- compare generated output and execution behavior
- measure compile latency for the direct session path

## Phase 3: Fortran Runtime and ABI

Goal: move beyond toy programs without hiding semantics in ad hoc lowering.

Decisions to document and test:

- Done: program entry and exit-code convention.
- Done for current subset: integer, real, and logical scalar storage, procedure
  parameter, and return-value convention.
- #54: name mangling for modules, external procedures, and separate
  compilation.
- #51: character representation.
- #52 and #53: array descriptors, allocatables, and pointers.
- Done for current subset: integer and real scalar `abs`, `min`, and `max`
  intrinsics, plus integer-to-real `real()` conversion.
- #55: I/O runtime surface.

Verification:

- executable parity tests grouped by language feature
- each new runtime call has a focused ABI test

## Unsupported Beyond The Current Contract

- Full Fortran module support: #54.
- Derived types: #60.
- Allocatable arrays and descriptors: #53.
- Generic specialization: out of scope for `ffc` until FortFront and
  package-level monomorphization contracts exist.
- Cross-module inference: out of scope for `ffc` until FortFront and
  package-level orchestration contracts exist.
- Optimization beyond basic lowering correctness: out of scope until the
  supported contract has reference conformance coverage.
