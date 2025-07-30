# fortfc MLIR C Bindings Strategy

## Executive Summary

This document outlines the strategy for transitioning fortfc from generating MLIR as text to using ISO C bindings for direct programmatic construction of MLIR, following the same HLFIR → FIR → LLVM IR → object code pipeline as flang.

## Current State

### What We Have
- Text-based MLIR generation in `mlir_backend.f90` and related modules
- Working HLFIR dialect text emission
- External tool invocation (`tco`, `llc`) for compilation
- Basic test infrastructure
- Integration with fortfront AST

### Pain Points
- Text generation is error-prone and hard to maintain
- No compile-time verification of MLIR correctness
- String manipulation overhead
- Difficult to handle complex MLIR attributes and types
- No direct access to MLIR optimization passes

## Target Architecture

### Overview
```
fortfront AST → fortfc → MLIR C API → HLFIR Operations → FIR Operations → LLVM IR → Object Code
                            ↓
                      In-memory MLIR
```

### Key Components

1. **MLIR C Bindings Layer** (`src/mlir_c/`)
   - Fortran interfaces to MLIR C API
   - Context and module management
   - Operation builders
   - Type system bindings
   - Attribute construction

2. **Dialect Bindings** (`src/dialects/`)
   - HLFIR dialect C bindings (to be created)
   - FIR dialect C bindings (to be created)
   - Standard dialect bindings
   - LLVM dialect bindings

3. **IR Builder** (`src/builder/`)
   - High-level Fortran API for MLIR construction
   - Type-safe operation builders
   - SSA value management
   - Block and region handling

4. **Pass Management** (`src/passes/`)
   - Direct access to MLIR optimization passes
   - Custom pass registration
   - Pass pipeline configuration

## Implementation Approach

### Phase 1: Foundation (Weeks 1-2)
1. Create MLIR C API bindings module
2. Implement basic context and module creation
3. Add operation and type construction primitives
4. Create initial test framework

### Phase 2: Dialect Support (Weeks 3-4)
1. Generate C bindings for FIR dialect
2. Generate C bindings for HLFIR dialect
3. Create Fortran wrappers for dialect operations
4. Test basic operation construction

### Phase 3: IR Builder (Weeks 5-6)
1. Design high-level IR builder API
2. Implement SSA value tracking
3. Add block and control flow support
4. Create type system integration

### Phase 4: Backend Integration (Weeks 7-8)
1. Refactor `mlir_backend_impl.f90` to use C API
2. Update code generation functions
3. Integrate pass management
4. Remove text generation code

### Phase 5: Testing and Optimization (Weeks 9-10)
1. Comprehensive test suite
2. Performance optimization
3. Memory management improvements
4. Documentation

## Technical Design

### MLIR C API Bindings

```fortran
module mlir_c_api
    use iso_c_binding
    implicit none
    
    ! Opaque types
    type :: mlir_context_t
        type(c_ptr) :: ptr = c_null_ptr
    end type
    
    type :: mlir_module_t
        type(c_ptr) :: ptr = c_null_ptr
    end type
    
    type :: mlir_operation_t
        type(c_ptr) :: ptr = c_null_ptr
    end type
    
    interface
        ! Context management
        function mlirContextCreate() bind(c, name="mlirContextCreate")
            import :: c_ptr
            type(c_ptr) :: mlirContextCreate
        end function
        
        ! Module operations
        function mlirModuleCreateEmpty(location) bind(c, name="mlirModuleCreateEmpty")
            import :: c_ptr
            type(c_ptr), value :: location
            type(c_ptr) :: mlirModuleCreateEmpty
        end function
    end interface
end module
```

### FIR/HLFIR Dialect Bindings

```fortran
module fir_c_api
    use mlir_c_api
    use iso_c_binding
    implicit none
    
    interface
        ! Register FIR dialect
        subroutine mlirContextRegisterFIRDialect(context) &
            bind(c, name="mlirContextRegisterFIRDialect")
            import :: c_ptr
            type(c_ptr), value :: context
        end subroutine
        
        ! FIR operations
        function fir_createDeclareOp(builder, memref, name) &
            bind(c, name="fir_createDeclareOp")
            import :: c_ptr
            type(c_ptr), value :: builder, memref, name
            type(c_ptr) :: fir_createDeclareOp
        end function
    end interface
end module
```

### High-Level IR Builder

```fortran
module mlir_builder
    use mlir_c_api
    use fir_c_api
    implicit none
    
    type :: mlir_builder_t
        type(mlir_context_t) :: context
        type(mlir_module_t) :: module
        type(c_ptr) :: insertion_point
        integer :: next_ssa_id = 0
    contains
        procedure :: create_function
        procedure :: create_declare
        procedure :: create_load
        procedure :: create_store
        procedure :: finalize
    end type
    
contains
    
    function create_builder() result(builder)
        type(mlir_builder_t) :: builder
        
        ! Initialize MLIR context
        builder%context%ptr = mlirContextCreate()
        
        ! Register dialects
        call mlirContextRegisterFIRDialect(builder%context%ptr)
        call mlirContextRegisterHLFIRDialect(builder%context%ptr)
        
        ! Create empty module
        builder%module%ptr = mlirModuleCreateEmpty(...)
    end function
end module
```

## Testing Strategy

### Test-Driven Development (TDD) Approach

1. **RED Phase**: Write failing tests first
   - Unit tests for each C binding function
   - Integration tests for operation construction
   - End-to-end compilation tests

2. **GREEN Phase**: Implement minimal code to pass
   - Focus on correctness over optimization
   - Use simple implementations initially

3. **REFACTOR Phase**: Improve code quality
   - Extract common patterns
   - Optimize performance
   - Improve error handling

### Test Categories

1. **Unit Tests** (`test/unit/`)
   - C API binding tests
   - Type system tests
   - Operation builder tests

2. **Integration Tests** (`test/integration/`)
   - AST to MLIR conversion
   - Pass pipeline tests
   - Compilation tests

3. **Regression Tests** (`test/regression/`)
   - Ensure text output compatibility
   - Performance benchmarks
   - Memory usage tests

## Migration Path

### Backward Compatibility
1. Keep text generation as fallback initially
2. Add feature flag for C API usage
3. Gradual migration of test suite
4. Deprecate text generation after validation

### Risk Mitigation
1. Incremental refactoring
2. Parallel implementation
3. Extensive testing at each phase
4. Performance monitoring

## Success Criteria

1. **Functional**: All existing tests pass with C API
2. **Performance**: ≥20% faster than text generation
3. **Maintainability**: Reduced code complexity
4. **Reliability**: No memory leaks or crashes
5. **Compatibility**: Same output as flang for equivalent input

## Dependencies

### Build Dependencies
- LLVM/MLIR development libraries
- ISO C binding support in Fortran compiler
- CMake for C/C++ components

### Runtime Dependencies
- MLIR runtime libraries
- FIR/HLFIR dialect libraries

## Open Questions

1. Should we contribute FIR/HLFIR C bindings upstream to LLVM?
2. How to handle version compatibility with MLIR?
3. Integration with flang's runtime library?

## Timeline

- **Month 1**: Foundation and basic bindings
- **Month 2**: Dialect support and IR builder
- **Month 3**: Backend integration and testing

## Conclusion

This strategy provides a clear path to modernize fortfc's MLIR generation, making it more robust, performant, and maintainable while following flang's architecture closely.