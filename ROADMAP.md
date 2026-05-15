# ffc Roadmap

## Current Reality

`ffc` is not a working compiler today.

The old roadmap assumed an HLFIR/MLIR-first backend. The source tree still
contains MLIR C API bindings and text-emitting MLIR generators, but the CLI
source path is stubbed and MLIR lowering/object/executable emission is not
implemented.

The active plan is to repurpose `ffc` as the Lazy Fortran compiler driver:
FortFront frontend, `ffc` lowering/runtime, LIRIC backend.

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

- Mark MLIR docs and modules as legacy.
- Remove MLIR/LLVM libraries from the default build path once LIRIC bindings
  exist.
- Keep old MLIR tests either disabled with clear naming or delete them when the
  LIRIC path replaces them.
- Replace the CLI source-file stub with a real FortFront frontend call.

Verification:

- `fpm build`
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

- parse `program main; end program`
- parse `x = 1` in infer mode
- diagnostics preserve source locations

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

- compile and run `program main; end program`
- compile and run scalar arithmetic with known exit/output
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
