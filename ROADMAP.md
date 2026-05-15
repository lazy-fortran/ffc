# ffc Roadmap

## Current Reality

`ffc` now has a minimal working compiler path for the empty-program MVP and the
first straight-line integer scalar slice.

The old roadmap assumed an HLFIR/MLIR-first backend. The source tree still
contains MLIR C API bindings and text-emitting MLIR generators, but that code is
legacy and not part of the default fpm build.

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
- Done: default manifest links LIRIC instead of MLIR/LLVM libraries.
- Done: CLI source path calls FortFront's compiler API.
- Pending: decide whether to move legacy MLIR code under an explicit archive
  directory or keep it as reference source outside the default build.

Verification:

- `LIBRARY_PATH=/home/ert/code/liric/build fpm build`
- CLI parses a file through FortFront and reports diagnostics instead of using a
  hardcoded root index.

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

## Phase 2: LIRIC Bootstrap Backend

Goal: produce the first executable without LLVM bindings.

Use the simple LIRIC compiler API:

- create compiler/session
- emit minimal LLVM IR text for a tiny subset
- feed text through `lr_compiler_feed_ll()`
- emit object/executable

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
- Remaining: real/logical/character literals
- Remaining: comparisons, `if`, counted `do`, simple procedures, and a fuller
  print/runtime surface
- compare output against a reference compiler for the supported subset

## Phase 3: Direct LIRIC Session Backend

Goal: remove text IR from the hot path.

Tasks:

- add ISO C bindings for `lr_session_*`
- map FortFront scalar types to LIRIC types
- lower expressions to vregs
- lower blocks/control flow to LIRIC blocks
- lower calls with an explicit ABI
- emit objects/executables directly from the session

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
