# ffc (Fortran Fortran Compiler) Architecture Design

**CORE PRINCIPLE: HLFIR-First Compilation**
- **ALWAYS use HLFIR** operations wherever possible
- **FIR as fallback** only when HLFIR lacks specific operation
- **Let LLVM handle lowering** - trust the optimization pipeline
- **Progressive capability** - start simple, add complexity incrementally

## Executive Summary

ffc is a production Fortran compiler that directly translates Fortran AST to HLFIR (High-Level Fortran IR) using MLIR's C API, producing optimized native executables through LLVM's backend.

**Core Mission**: Incrementally buildable Fortran compiler that can compile SOMETHING immediately (even just `program hello; end program`), then systematically add capabilities while maintaining a working compiler at every step.

## Architecture Overview

```
Source Files (.f90, .lf)
         ↓
    [fortfront]
    AST + Type Info
         ↓
      [ffc]
    HLFIR Generation
         ↓
   [MLIR/LLVM]
  HLFIR → FIR → LLVM
         ↓
    Object/Executable
```

## Key Architectural Principles

### 1. HLFIR-First Strategy
- **HARD RESTRICTION**: Always use HLFIR operations wherever possible
- Use FIR only when HLFIR lacks equivalent operations  
- Let LLVM handle all low-level lowering
- Preserve high-level semantics as long as possible

### 2. Incremental Capability
- Start with absolute minimum compilable program
- Each issue adds ONE specific, testable capability
- System remains compilable and working after EVERY merge
- Clear documentation of what we can compile at each stage

### 3. Debug-First Development
- Emit .ll files (LLVM IR text) for debugging at every stage
- Provide clear visibility into compilation pipeline
- Enable comparison with flang/gfortran output

## Phase 1: Foundation (Immediate Priority)

### Milestone 1.0: Empty Program Compilation
**Goal**: Compile `program hello; end program` to executable

**Capabilities Added**:
- fortfront AST integration (**simplified by fpm automatic linking**)
- Basic HLFIR module structure generation
- MLIR → LLVM → executable pipeline
- .ll file emission for debugging

**Verification**:
```bash
echo "program hello; end program" > test.f90
ffc test.f90 -o test
./test  # Should exit cleanly
ffc test.f90 --emit-llvm -o test.ll  # Debug output
```

### Milestone 1.1: Print Statement
**Goal**: Compile `program hello; print *, "Hello, World!"; end program`

**Capabilities Added**:
- String literal handling
- Basic I/O runtime integration
- hlfir.declare for implicit variables

**Verification**:
```bash
echo 'program hello; print *, "Hello, World!"; end program' > hello.f90
ffc hello.f90 -o hello
./hello  # Output: Hello, World!
```

### Milestone 1.2: Variable Declaration
**Goal**: Compile programs with explicit variable declarations

**Capabilities Added**:
- Integer/Real/Logical/Character type mapping
- hlfir.declare for explicit variables
- Basic memory allocation

**Examples**:
```fortran
program vars
    integer :: i = 42
    real :: x = 3.14
    print *, i, x
end program
```

## Phase 2: Basic Operations

### Milestone 2.0: Arithmetic Operations
**Goal**: Support basic arithmetic expressions

**Capabilities Added**:
- Binary operations (+, -, *, /, **)
- Type coercion (integer ↔ real)
- hlfir.elemental for array operations

### Milestone 2.1: Assignment Statements
**Goal**: Support variable assignments

**Capabilities Added**:
- hlfir.assign with proper semantics
- Scalar and array assignments
- Character assignment with padding

### Milestone 2.2: Simple Control Flow
**Goal**: Support if-then-else statements

**Capabilities Added**:
- Conditional branching
- Block structure generation
- SSA value management

## Phase 3: Arrays and Loops

### Milestone 3.0: Fixed-Size Arrays
**Goal**: Support static array declarations and operations

**Capabilities Added**:
- Array type generation (!fir.array<NxT>)
- hlfir.designate for array indexing
- Array initialization

### Milestone 3.1: Do Loops
**Goal**: Support simple do loops

**Capabilities Added**:
- Loop structure generation
- Loop variable management
- Array operations in loops

### Milestone 3.2: Array Sections
**Goal**: Support array slicing and sections

**Capabilities Added**:
- hlfir.designate with bounds
- Array section operations
- Temporary management

## Phase 4: Procedures

### Milestone 4.0: Subroutines
**Goal**: Support subroutine definition and calls

**Capabilities Added**:
- Function signature generation
- Argument passing (by reference)
- Call site generation

### Milestone 4.1: Functions
**Goal**: Support function definition and calls

**Capabilities Added**:
- Return value handling
- Function result variables
- Expression evaluation

### Milestone 4.2: Intrinsic Functions
**Goal**: Support common intrinsic functions

**Capabilities Added**:
- Intrinsic recognition
- Runtime library calls
- Type-specific intrinsics

## Phase 5: Advanced Features

### Milestone 5.0: Allocatable Arrays
**Goal**: Support dynamic memory management

**Capabilities Added**:
- !fir.box types for descriptors
- Allocation/deallocation
- Automatic deallocation

### Milestone 5.1: Derived Types
**Goal**: Support user-defined types

**Capabilities Added**:
- Type definition mapping
- Component access via hlfir.designate
- Type constructors

### Milestone 5.2: Modules
**Goal**: Support module compilation and use

**Capabilities Added**:
- Module file generation
- Symbol import/export
- Dependency management

## Phase 6: Lazy Fortran Support

### Milestone 6.0: Implicit Declarations
**Goal**: Support lazy Fortran's implicit variable declarations

**Capabilities Added**:
- Type inference from usage
- Automatic hlfir.declare insertion
- .lf file support

**Example**:
```fortran
! input.lf
x = 2
y = 3.14
print *, x + y
```

### Milestone 6.1: Minimal Syntax
**Goal**: Full lazy Fortran support

**Capabilities Added**:
- Implicit program wrapper
- Automatic type selection
- Smart inference

## Implementation Architecture

### Core Modules

#### AST Integration (`src/frontend/`) - **fmp automatic linking**
- `fortfront_bridge.f90` - AST traversal and node access
- `ast_to_hlfir.f90` - AST → HLFIR conversion
- `type_mapper.f90` - Fortran → MLIR type mapping
- **Integration architecture**: Static linking via fmp path dependency

#### HLFIR Generation (`src/codegen/`)
- `hlfir_builder.f90` - High-level HLFIR construction
- `hlfir_operations.f90` - Operation generators
- `ssa_tracker.f90` - SSA value management

#### Pipeline Management (`src/passes/`)
- `compilation_pipeline.f90` - Full compilation flow
- `lowering_passes.f90` - HLFIR → FIR → LLVM
- `optimization_passes.f90` - Optional optimizations

#### Runtime Integration (`src/runtime/`)
- `fortran_runtime.f90` - Runtime library interface
- `io_runtime.f90` - I/O operation support
- `intrinsic_runtime.f90` - Intrinsic function support

### Debug and Alternative Output

#### LLVM IR Text Output
- `--emit-llvm` flag for .ll file generation
- Human-readable LLVM IR for debugging
- Comparison with other compiler output

#### HLFIR/FIR Text Output
- `--emit-hlfir` for high-level IR
- `--emit-fir` for mid-level IR
- Pipeline visibility at each stage

## Testing Strategy

### Unit Tests
- Each HLFIR operation individually
- Type conversion accuracy
- SSA value correctness

### Integration Tests
- Complete programs at each milestone
- Comparison with gfortran output
- Runtime behavior validation

### Regression Tests
- Ensure capabilities never regress
- Automated test suite growth
- CI/CD integration

## Current Infrastructure Status

### ✅ What's Working
- Complete MLIR C API wrapper (51 modules)
- Type conversion system
- IR builder infrastructure
- Memory management
- Test harness (82 tests passing)

### 🔴 What's Missing
- fortfront AST integration (**build system ready via fpm**)
- Actual HLFIR operation generation
- Runtime library integration
- Executable generation

### 🚧 Blockers
- fortfront Issue #32: Node accessor functions needed
- fortfront lacks component access nodes
- fortfront lacks array section representation

## Success Metrics

### Phase 1 Success (Immediate Goal)
- Can compile "Hello, World!" program
- Generates working executable
- Produces readable .ll files

### Phase 2 Success (Short Term)
- Can compile basic arithmetic programs
- Supports simple control flow
- Handles variable assignments

### Phase 3 Success (Medium Term)
- Array operations work correctly
- Loop constructs compile
- Performance comparable to gfortran -O0

### Phase 6 Success (Long Term)
- Full lazy Fortran support
- Type inference works correctly
- .lf files compile seamlessly

## Development Workflow

### For Each Capability
1. Write failing test for new feature
2. Implement minimal HLFIR generation
3. Verify .ll output is correct
4. Ensure executable runs properly
5. Document what's now compilable
6. Update examples and tests

### Debugging Process
1. Generate .ll file with `--emit-llvm`
2. Compare with flang/gfortran output
3. Identify HLFIR generation issues
4. Fix and verify with execution

## Appendix: HLFIR Operation Priority

### Critical Operations (Phase 1-2)
- `hlfir.declare` - Variable declarations
- `hlfir.assign` - Assignments
- `func.func` - Program/function structure

### Important Operations (Phase 3-4)
- `hlfir.designate` - Array/component access
- `hlfir.elemental` - Array operations
- `scf.for` - Loop structures

### Advanced Operations (Phase 5-6)
- `hlfir.associate` - Temporaries
- `hlfir.expr` - Expression values
- `fir.allocmem` - Dynamic allocation