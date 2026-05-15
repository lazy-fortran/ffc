# ffc Roadmap

## Current Reality

`ffc` now has a minimal working compiler path for the empty-program MVP and the
first straight-line integer scalar slice.

The old roadmap assumed an HLFIR/MLIR-first backend. The source tree still
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

## Phase 0: Retire the Misleading MLIR Default

Goal: make the repository honest and buildable for the new backend direction.

Tasks:

- Done: default fpm build points at `src_mvp/`, not the old MLIR tree.
- Done: default manifest links LIRIC instead of the old backend libraries.
- Done: CLI source path calls FortFront's compiler API.
- Pending: decide whether to move legacy MLIR code under an explicit archive
  directory or keep it as reference source outside the default build.

Verification:

- `LIBRARY_PATH=/home/ert/code/liric/build fpm build`
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

`ffc` should use public FortFront APIs only. If private AST layout is required,
that is a FortFront API bug, not an `ffc` workaround.

Verification:

- Done in FortFront: `compile_frontend_from_string/file` returns arena,
  root index, semantic context, tokens, and diagnostics.
- Pending: add narrower compiler query APIs so `ffc` does not depend on arena
  representation details for real scalar lowering.

## Phase 2: LIRIC Bootstrap Reference Backend

Goal: preserve the first executable reference path while direct LIRIC session
lowering catches up.

Use the simple LIRIC compiler API:

- create compiler/session
- emit minimal text IR for a tiny subset
- feed text through LIRIC's compiler API
- Done: emit executable through direct LIRIC session
- Done: emit object files through direct LIRIC session

This path is not the CLI default and should not receive broad new features.

Initial language subset:

- main program
- scalar integer/real/logical literals
- scalar declarations
- assignment
- arithmetic
- comparisons
- `if`
- counted `do`
- simple function/subroutine calls
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
- Remaining: dynamic counted `do`, simple procedures, and a fuller
  print/runtime surface
- compare output against a reference compiler for the supported subset

## Phase 3: Direct LIRIC Session Backend

Goal: remove text IR from the hot path.

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
- Done: lower literal-bound counted `do` loops by direct-session scalar
  operation expansion
- Done: map current integer, real, and logical scalar values to LIRIC types
- Done: lower integer, real, and logical expressions to direct-session values
- Done: lower scalar integer, real, character literal, and logical literal
  `print` through the current `printf` ABI shim
- Done: lower logical declarations, assignment, printed logical variables, and
  `if (flag)` conditions
- Done: lower simple contained integer functions and integer call expressions
- generalize non-terminating blocks/control flow with merge values
- lower runtime counted `do` loops to LIRIC blocks with backedge PHI values
  after krystophny/liric#519
- lower subroutines and richer function signatures with an explicit ABI
- Done: emit objects directly from the session

Verification:

- same executable tests as Phase 2
- compare generated output and execution behavior
- measure compile latency against the text bootstrap path

## Phase 4: Fortran Runtime and ABI

Goal: move beyond toy programs without hiding semantics in ad hoc lowering.

Decisions to document and test:

- program entry and exit-code convention
- name mangling
- pass-by-reference semantics
- return values
- character representation
- array descriptors
- allocatables and pointers
- intrinsic/runtime call surface
- I/O runtime surface

Verification:

- executable parity tests grouped by language feature
- each new runtime call has a focused ABI test

## Deferred

- Full Fortran module support
- Derived types
- Allocatable arrays and descriptors
- Generic specialization
- Cross-module inference
- Optimization beyond basic lowering correctness

Those are important, but they should not block the first working compiler.
