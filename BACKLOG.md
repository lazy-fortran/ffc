# fortfc Development Backlog

## Overview

This backlog details the tasks required to transition fortfc from text-based MLIR generation to using ISO C bindings. Tasks are organized by epic and follow a strict RED/GREEN/REFACTOR test-driven development approach.

## Epic 1: MLIR C API Foundation

### 1.1 Create Basic C Bindings Module [5 story points]
**RED Tests:**
- [ ] Test MLIR context creation and destruction
- [ ] Test MLIR module creation
- [ ] Test location creation
- [ ] Test string ref handling

**GREEN Implementation:**
- [ ] Create `src/mlir_c/mlir_c_core.f90`
- [ ] Define opaque types for Context, Module, Operation, Value
- [ ] Add C interface declarations
- [ ] Implement Fortran wrappers with error handling

**REFACTOR:**
- [ ] Add RAII-style resource management
- [ ] Create builder pattern for common operations

### 1.2 Type System Bindings [8 story points]
**RED Tests:**
- [ ] Test integer type creation (i1, i8, i16, i32, i64)
- [ ] Test float type creation (f32, f64)
- [ ] Test array type creation
- [ ] Test reference type creation

**GREEN Implementation:**
- [ ] Create `src/mlir_c/mlir_c_types.f90`
- [ ] Implement type construction functions
- [ ] Add type query functions
- [ ] Handle type caching

**REFACTOR:**
- [ ] Create type factory with memoization
- [ ] Add type validation

### 1.3 Attribute System Bindings [5 story points]
**RED Tests:**
- [ ] Test integer attribute creation
- [ ] Test float attribute creation
- [ ] Test string attribute creation
- [ ] Test array attribute creation

**GREEN Implementation:**
- [ ] Create `src/mlir_c/mlir_c_attributes.f90`
- [ ] Implement attribute builders
- [ ] Add attribute getters

**REFACTOR:**
- [ ] Unify attribute creation API
- [ ] Add attribute validation

### 1.4 Operation Builder Infrastructure [8 story points]
**RED Tests:**
- [ ] Test operation state creation
- [ ] Test operand addition
- [ ] Test result type specification
- [ ] Test attribute attachment
- [ ] Test operation creation and verification

**GREEN Implementation:**
- [ ] Create `src/mlir_c/mlir_c_operations.f90`
- [ ] Implement operation state management
- [ ] Add operation builder helpers
- [ ] Create operation verification wrapper

**REFACTOR:**
- [ ] Create fluent API for operation building
- [ ] Add operation templates

## Epic 2: Dialect C Bindings

### 2.1 Generate FIR Dialect C Bindings [13 story points]
**RED Tests:**
- [ ] Test FIR dialect registration
- [ ] Test fir.declare operation
- [ ] Test fir.load operation
- [ ] Test fir.store operation
- [ ] Test fir.alloca operation
- [ ] Test fir.do_loop operation
- [ ] Test fir.if operation

**GREEN Implementation:**
- [ ] Create C++ binding generator for FIR
- [ ] Generate `src/dialects/fir_c_api.h`
- [ ] Generate `src/dialects/fir_c_api.cpp`
- [ ] Create Fortran interface `src/dialects/fir_dialect.f90`
- [ ] Implement operation builders for each FIR op

**REFACTOR:**
- [ ] Create operation builder templates
- [ ] Optimize common patterns

### 2.2 Generate HLFIR Dialect C Bindings [13 story points]
**RED Tests:**
- [ ] Test HLFIR dialect registration
- [ ] Test hlfir.declare operation
- [ ] Test hlfir.designate operation
- [ ] Test hlfir.elemental operation
- [ ] Test hlfir.associate operation
- [ ] Test hlfir.end_associate operation

**GREEN Implementation:**
- [ ] Create C++ binding generator for HLFIR
- [ ] Generate `src/dialects/hlfir_c_api.h`
- [ ] Generate `src/dialects/hlfir_c_api.cpp`
- [ ] Create Fortran interface `src/dialects/hlfir_dialect.f90`
- [ ] Implement operation builders

**REFACTOR:**
- [ ] Share code with FIR bindings
- [ ] Create dialect-agnostic helpers

### 2.3 Standard Dialects Integration [5 story points]
**RED Tests:**
- [ ] Test func dialect operations
- [ ] Test arith dialect operations
- [ ] Test scf dialect operations

**GREEN Implementation:**
- [ ] Create interfaces for standard dialects
- [ ] Add operation builders
- [ ] Test with simple examples

**REFACTOR:**
- [ ] Unify dialect registration

## Epic 3: High-Level IR Builder

### 3.1 Builder Context Management [8 story points]
**RED Tests:**
- [ ] Test builder creation and cleanup
- [ ] Test insertion point management
- [ ] Test block creation
- [ ] Test region handling

**GREEN Implementation:**
- [ ] Create `src/builder/mlir_builder.f90`
- [ ] Implement builder state management
- [ ] Add insertion point stack
- [ ] Create scope management

**REFACTOR:**
- [ ] Add builder validation
- [ ] Optimize state transitions

### 3.2 SSA Value Management [8 story points]
**RED Tests:**
- [ ] Test SSA value generation
- [ ] Test value naming
- [ ] Test value type tracking
- [ ] Test use-def chains

**GREEN Implementation:**
- [ ] Create `src/builder/ssa_manager.f90`
- [ ] Implement value table
- [ ] Add type tracking
- [ ] Create value naming scheme

**REFACTOR:**
- [ ] Optimize value lookups
- [ ] Add debug helpers

### 3.3 Type Conversion System [13 story points]
**RED Tests:**
- [ ] Test integer type conversion (i8, i16, i32, i64)
- [ ] Test real type conversion (f32, f64)
- [ ] Test logical type conversion (i1)
- [ ] Test character type conversion (!fir.char)
- [ ] Test complex type conversion (!fir.complex)
- [ ] Test fixed-size array conversion (!fir.array<NxT>)
- [ ] Test assumed-shape array conversion (!fir.box<!fir.array<?xT>>)
- [ ] Test allocatable array conversion (!fir.ref<!fir.box<!fir.heap<!fir.array<?xT>>>>)
- [ ] Test pointer type conversion (!fir.ref<!fir.box<!fir.ptr<T>>>)
- [ ] Test derived type conversion (!fir.type<name{fields}>)
- [ ] Test function type signatures

**GREEN Implementation:**
- [ ] Create `src/builder/fortfc_type_converter.f90` as per TYPE_CONVERSION.md
- [ ] Implement `mlir_type_converter_t` type with context management
- [ ] Implement `convert_type` function for basic types
- [ ] Implement `get_mlir_type_string` for type descriptor generation
- [ ] Add `create_integer_type`, `create_float_type` C binding wrappers
- [ ] Add `create_fir_char_type` for character types
- [ ] Implement `create_array_type` with shape handling
- [ ] Add support for !fir.box types (descriptors)
- [ ] Add support for !fir.heap types (allocatable)
- [ ] Add support for !fir.ptr types (pointers)
- [ ] Implement derived type name mangling (_QTtypename)
- [ ] Create type caching mechanism

**REFACTOR:**
- [ ] Extract type factory patterns
- [ ] Optimize type descriptor string generation
- [ ] Add comprehensive type validation
- [ ] Create type compatibility checking
- [ ] Add debug type dumping utilities

### 3.4 Type Conversion Helpers [5 story points]
**RED Tests:**
- [ ] Test type builder helper functions
- [ ] Test array shape extraction from fortfront
- [ ] Test reference type wrapping logic
- [ ] Test type equivalence checking

**GREEN Implementation:**
- [ ] Create `get_array_descriptor` for array type strings
- [ ] Create `wrap_with_reference_type` for !fir.ref handling
- [ ] Create `wrap_with_box_type` for descriptor handling
- [ ] Create `mangle_derived_type_name` for Fortran name mangling
- [ ] Add `is_assumed_shape`, `is_allocatable`, `is_pointer` helpers
- [ ] Create `get_element_type` for nested type extraction

**REFACTOR:**
- [ ] Consolidate type helper patterns
- [ ] Add type assertion utilities

## Epic 4: AST to MLIR Conversion

### 4.1 Program and Module Generation [8 story points]
**RED Tests:**
- [ ] Test empty program generation
- [ ] Test module with functions
- [ ] Test module variables
- [ ] Test use statements

**GREEN Implementation:**
- [ ] Create `src/codegen/program_gen.f90`
- [ ] Implement module structure generation
- [ ] Add function prototypes
- [ ] Handle module dependencies

**REFACTOR:**
- [ ] Optimize module ordering
- [ ] Add dependency analysis

### 4.2 Function and Subroutine Generation [13 story points]
**RED Tests:**
- [ ] Test function signature generation
- [ ] Test parameter handling
- [ ] Test local variable declarations
- [ ] Test return value handling

**GREEN Implementation:**
- [ ] Create `src/codegen/function_gen.f90`
- [ ] Implement function body generation
- [ ] Add argument handling
- [ ] Create return value management

**REFACTOR:**
- [ ] Unify function/subroutine handling
- [ ] Optimize calling conventions

### 4.3 Statement Generation [21 story points]
**RED Tests:**
- [ ] Test assignment statements
- [ ] Test if-then-else statements
- [ ] Test do loop statements
- [ ] Test while loops
- [ ] Test select case statements
- [ ] Test print statements
- [ ] Test read statements

**GREEN Implementation:**
- [ ] Create `src/codegen/statement_gen.f90`
- [ ] Implement each statement type
- [ ] Add control flow handling
- [ ] Create I/O operation builders

**REFACTOR:**
- [ ] Extract common patterns
- [ ] Optimize control flow generation

### 4.4 Expression Generation [13 story points]
**RED Tests:**
- [ ] Test literal expressions
- [ ] Test variable references
- [ ] Test binary operations
- [ ] Test unary operations
- [ ] Test function calls
- [ ] Test array subscripts

**GREEN Implementation:**
- [ ] Create `src/codegen/expression_gen.f90`
- [ ] Implement expression evaluation
- [ ] Add type coercion
- [ ] Handle intrinsic functions

**REFACTOR:**
- [ ] Optimize expression trees
- [ ] Add constant folding

## Epic 5: Pass Management and Optimization

### 5.1 Pass Manager Integration [8 story points]
**RED Tests:**
- [ ] Test pass manager creation
- [ ] Test pass pipeline configuration
- [ ] Test pass execution
- [ ] Test pass verification

**GREEN Implementation:**
- [ ] Create `src/passes/pass_manager.f90`
- [ ] Integrate MLIR pass infrastructure
- [ ] Add pass pipeline builder
- [ ] Create verification wrapper

**REFACTOR:**
- [ ] Add pass caching
- [ ] Optimize pass ordering

### 5.2 Lowering Pipeline [13 story points]
**RED Tests:**
- [ ] Test HLFIR to FIR lowering
- [ ] Test FIR to LLVM lowering
- [ ] Test optimization passes
- [ ] Test debug info preservation

**GREEN Implementation:**
- [ ] Create `src/passes/lowering_pipeline.f90`
- [ ] Configure standard passes
- [ ] Add custom passes
- [ ] Integrate with backend

**REFACTOR:**
- [ ] Optimize pass configuration
- [ ] Add pass profiling

## Epic 6: Backend Integration

### 6.1 Refactor MLIR Backend [13 story points]
**RED Tests:**
- [ ] Test C API backend against existing tests
- [ ] Test compilation to object files
- [ ] Test executable generation
- [ ] Test optimization levels

**GREEN Implementation:**
- [ ] Create `src/backend/mlir_c_backend.f90`
- [ ] Replace text generation with C API calls
- [ ] Update compilation pipeline
- [ ] Maintain backward compatibility

**REFACTOR:**
- [ ] Remove text generation code
- [ ] Optimize compilation pipeline
- [ ] Clean up interfaces

### 6.2 Memory Management [8 story points]
**RED Tests:**
- [ ] Test memory leak detection
- [ ] Test large program handling
- [ ] Test error recovery
- [ ] Test resource cleanup

**GREEN Implementation:**
- [ ] Add memory tracking
- [ ] Implement proper cleanup
- [ ] Add error recovery
- [ ] Create resource guards

**REFACTOR:**
- [ ] Optimize memory usage
- [ ] Add memory profiling

## Epic 7: Testing and Documentation

### 7.1 Comprehensive Test Suite [13 story points]
**Tasks:**
- [ ] Port all existing tests to C API
- [ ] Add unit tests for each component
- [ ] Create integration test suite
- [ ] Add performance benchmarks
- [ ] Create memory leak tests
- [ ] Add type conversion validation tests:
  - [ ] Compare generated types with flang output
  - [ ] Test all TYPE_CONVERSION.md examples
  - [ ] Validate array descriptor formats
  - [ ] Test edge cases (zero-size arrays, etc.)
  - [ ] Verify derived type name mangling

### 7.2 Documentation [8 story points]
**Tasks:**
- [ ] Document C API usage
- [ ] Create developer guide
- [ ] Add API reference
- [ ] Create migration guide
- [ ] Update README

### 7.3 CI/CD Integration [5 story points]
**Tasks:**
- [ ] Update build system for C++ components
- [ ] Add C API tests to CI
- [ ] Create performance tracking
- [ ] Add memory leak detection
- [ ] Update release process

## Prioritization

### Phase 1 (Weeks 1-2): Foundation
1. Epic 1.1: Basic C Bindings Module
2. Epic 1.2: Type System Bindings
3. Epic 1.3: Attribute System Bindings
4. Epic 1.4: Operation Builder Infrastructure

### Phase 2 (Weeks 3-4): Dialects
1. Epic 2.1: FIR Dialect C Bindings
2. Epic 2.2: HLFIR Dialect C Bindings
3. Epic 2.3: Standard Dialects Integration

### Phase 3 (Weeks 5-6): IR Builder
1. Epic 3.1: Builder Context Management
2. Epic 3.2: SSA Value Management
3. Epic 3.3: Type Conversion System
4. Epic 3.4: Type Conversion Helpers

### Phase 4 (Weeks 7-8): Code Generation
1. Epic 4.1: Program and Module Generation
2. Epic 4.2: Function Generation
3. Epic 4.3: Statement Generation (partial)
4. Epic 4.4: Expression Generation (partial)

### Phase 5 (Weeks 9-10): Integration
1. Epic 4.3: Statement Generation (complete)
2. Epic 4.4: Expression Generation (complete)
3. Epic 5.1: Pass Manager Integration
4. Epic 5.2: Lowering Pipeline
5. Epic 6.1: Backend Integration

### Phase 6 (Week 11+): Polish
1. Epic 6.2: Memory Management
2. Epic 7.1: Test Suite
3. Epic 7.2: Documentation
4. Epic 7.3: CI/CD Integration

## Success Metrics

- All existing tests pass with C API implementation
- No memory leaks detected by valgrind
- Compilation time improved by >20%
- Code coverage >90%
- Zero text-based MLIR generation remaining

## Notes

- Each task should start with failing tests (RED)
- Implementation should be minimal to pass tests (GREEN)
- Refactoring happens only after tests pass (REFACTOR)
- No feature is complete without tests and documentation
- Performance optimization is part of REFACTOR phase