# fortfc Development Backlog

## Overview

This backlog details the tasks required to transition fortfc from text-based MLIR generation to using ISO C bindings. Tasks are organized by epic and follow a strict RED/GREEN/REFACTOR test-driven development approach.

## Epic 1: MLIR C API Foundation

### 1.1 Create Basic C Bindings Module [5 story points]
**RED Tests:**
- [x] Test MLIR context creation and destruction
- [x] Test MLIR module creation
- [x] Test location creation
- [x] Test string ref handling

**GREEN Implementation:**
- [x] Create `src/mlir_c/mlir_c_core.f90`
- [x] Define opaque types for Context, Module, Operation, Value
- [x] Add C interface declarations
- [x] Implement Fortran wrappers with error handling

**REFACTOR:**
- [x] Add RAII-style resource management
- [x] Create builder pattern for common operations

### 1.2 Type System Bindings [8 story points]
**RED Tests:**
- [x] Test integer type creation (i1, i8, i16, i32, i64)
- [x] Test float type creation (f32, f64)
- [x] Test array type creation
- [x] Test reference type creation

**GREEN Implementation:**
- [x] Create `src/mlir_c/mlir_c_types.f90`
- [x] Implement type construction functions
- [x] Add type query functions
- [x] Handle type caching

**REFACTOR:**
- [x] Create type factory with memoization
- [x] Add type validation

### 1.3 Attribute System Bindings [5 story points] ✓
**RED Tests:**
- [x] Test integer attribute creation
- [x] Test float attribute creation
- [x] Test string attribute creation
- [x] Test array attribute creation

**GREEN Implementation:**
- [x] Create `src/mlir_c/mlir_c_attributes.f90`
- [x] Implement attribute builders
- [x] Add attribute getters

**REFACTOR:**
- [x] Unify attribute creation API
- [x] Add attribute validation

### 1.4 Operation Builder Infrastructure [8 story points] ✓
**RED Tests:**
- [x] Test operation state creation
- [x] Test operand addition
- [x] Test result type specification
- [x] Test attribute attachment
- [x] Test operation creation and verification

**GREEN Implementation:**
- [x] Create `src/mlir_c/mlir_c_operations.f90`
- [x] Implement operation state management
- [x] Add operation builder helpers
- [x] Create operation verification wrapper

**REFACTOR:**
- [x] Create fluent API for operation building
- [x] Add operation templates

## Epic 2: Dialect C Bindings

### 2.1 Generate FIR Dialect C Bindings [13 story points] ✓
**RED Tests:**
- [x] Test FIR dialect registration
- [x] Test fir.declare operation
- [x] Test fir.load operation
- [x] Test fir.store operation
- [x] Test fir.alloca operation
- [x] Test fir.do_loop operation
- [x] Test fir.if operation

**GREEN Implementation:**
- [x] Create C++ binding generator for FIR
- [x] Generate `src/dialects/fir_c_api.h`
- [x] Generate `src/dialects/fir_c_api.cpp`
- [x] Create Fortran interface `src/dialects/fir_dialect.f90`
- [x] Implement operation builders for each FIR op

**REFACTOR:**
- [x] Create operation builder templates
- [x] Optimize common patterns

### 2.2 Generate HLFIR Dialect C Bindings [13 story points] ✓
**RED Tests:**
- [x] Test HLFIR dialect registration
- [x] Test hlfir.declare operation
- [x] Test hlfir.designate operation
- [x] Test hlfir.elemental operation
- [x] Test hlfir.associate operation
- [x] Test hlfir.end_associate operation

**GREEN Implementation:**
- [x] Create C++ binding generator for HLFIR
- [x] Generate `src/dialects/hlfir_c_api.h`
- [x] Generate `src/dialects/hlfir_c_api.cpp`
- [x] Create Fortran interface `src/dialects/hlfir_dialect.f90`
- [x] Implement operation builders

**REFACTOR:**
- [x] Share code with FIR bindings
- [x] Create dialect-agnostic helpers

### 2.3 Standard Dialects Integration [5 story points] ✓
**RED Tests:**
- [x] Test func dialect operations
- [x] Test arith dialect operations
- [x] Test scf dialect operations

**GREEN Implementation:**
- [x] Create interfaces for standard dialects
- [x] Add operation builders
- [x] Test with simple examples

**REFACTOR:**
- [x] Unify dialect registration

## Epic 3: High-Level IR Builder

### 3.1 Builder Context Management [8 story points] ✓
**RED Tests:**
- [x] Test builder creation and cleanup
- [x] Test insertion point management
- [x] Test block creation
- [x] Test region handling

**GREEN Implementation:**
- [x] Create `src/builder/mlir_builder.f90`
- [x] Implement builder state management
- [x] Add insertion point stack
- [x] Create scope management

**REFACTOR:**
- [x] Add builder validation
- [x] Optimize state transitions

### 3.2 SSA Value Management [8 story points] ✓
**RED Tests:**
- [x] Test SSA value generation
- [x] Test value naming
- [x] Test value type tracking
- [x] Test use-def chains

**GREEN Implementation:**
- [x] Create `src/builder/ssa_manager.f90`
- [x] Implement value table
- [x] Add type tracking
- [x] Create value naming scheme

**REFACTOR:**
- [x] Optimize value lookups (O(1) hash table)
- [x] Add debug helpers (dump, memory usage, validation)

### 3.3 Type Conversion System [13 story points] ✓
**RED Tests:**
- [x] Test integer type conversion (i8, i16, i32, i64)
- [x] Test real type conversion (f32, f64)
- [x] Test logical type conversion (i1)
- [x] Test character type conversion (!fir.char)
- [x] Test complex type conversion (!fir.complex)
- [x] Test fixed-size array conversion (!fir.array<NxT>)
- [~] Test assumed-shape array conversion (!fir.box<!fir.array<?xT>>) *
- [~] Test allocatable array conversion (!fir.ref<!fir.box<!fir.heap<!fir.array<?xT>>>>) *
- [~] Test pointer type conversion (!fir.ref<!fir.box<!fir.ptr<T>>>) *
- [~] Test derived type conversion (!fir.type<name{fields}>) *
- [~] Test function type signatures *

**GREEN Implementation:**
- [x] Create `src/builder/fortfc_type_converter.f90` as per TYPE_CONVERSION.md
- [x] Implement `mlir_type_converter_t` type with context management
- [x] Implement basic type creation functions
- [x] Implement `get_mlir_type_string` for type descriptor generation
- [x] Add `create_integer_type`, `create_float_type` wrappers
- [x] Add `create_character_type` for character types
- [x] Implement `create_array_type` with basic shape handling
- [~] Add support for !fir.box types (descriptors) *
- [~] Add support for !fir.heap types (allocatable) *
- [~] Add support for !fir.ptr types (pointers) *
- [~] Implement derived type name mangling (_QTtypename) *
- [x] Create type caching mechanism

**REFACTOR:**
- [x] Extract type factory patterns (hash table caching)
- [x] Optimize type descriptor string generation
- [x] Add comprehensive type caching with statistics
- [~] Create type compatibility checking *
- [~] Add debug type dumping utilities *

* = Deferred due to fortfront compilation issues - core functionality complete

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