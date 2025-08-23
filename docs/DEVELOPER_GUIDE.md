# FortFC Developer Guide

## Overview

FortFC is a Fortran compiler that generates HLFIR (High-Level FIR) using the MLIR C API exclusively. This guide covers the development process, architecture, and contribution guidelines.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Fortran AST   │───▶│   HLFIR (C API)  │───▶│  FIR → LLVM IR  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Object/Binary  │
                       └─────────────────┘
```

### Key Principles

1. **MLIR C API Only**: No text-based MLIR generation
2. **HLFIR First**: Generate high-level operations, lower through passes
3. **TDD Approach**: Test-Driven Development with RED-GREEN-REFACTOR
4. **Memory Safety**: RAII patterns and automatic resource management
5. **Performance**: Efficient C API usage with minimal overhead

## Fortfront Integration

**fpm Automatic Linking**: Fortran Package Manager automatically handles static linking to fortfront via the path dependency configured in fpm.toml. This greatly simplifies the integration process.

```bash
# Build ffc with automatic fortfront linking
cd /path/to/ffc && fmp build

# Test ffc with fortfront automatically linked
cd /path/to/ffc && fpm test
```

**Key Integration Points:**
- **AST Parsing**: fortfront provides the Fortran AST that ffc consumes
- **Module Separation**: ffc uses `ffc_` prefixed modules, fortfront uses unprefixed names
- **Automatic Dependency**: fpm handles fortfront build and linking automatically
- **Static Linking**: No runtime dependencies - fortfront linked into ffc executable
- **Simplified Workflow**: Standard fpm commands handle complex integration automatically

**Import Pattern:**
```fortran
! In ffc modules - use our own prefixed modules
use ffc_error_handling, only: make_error
use ffc_pass_manager, only: create_lowering_pipeline

! Import from fortfront when available
use ast_node, only: ast_program_node_t  ! From fortfront
use semantic_analyzer, only: analyze_program  ! From fortfront
```

## Directory Structure

```
src/
├── mlir_c/              # MLIR C API bindings
│   ├── mlir_c_core.f90       # Core context/module/operation types
│   ├── mlir_c_types.f90      # Type system bindings
│   ├── mlir_c_attributes.f90 # Attribute system bindings
│   └── mlir_c_operations.f90 # Operation builder bindings
├── dialects/            # Dialect-specific bindings
│   ├── fir_dialect.f90       # FIR dialect operations
│   ├── hlfir_dialect.f90     # HLFIR dialect operations
│   └── standard_dialects.f90 # func, arith, scf dialects
├── builder/             # High-level IR building
│   ├── mlir_builder.f90      # Builder context management
│   ├── ssa_manager.f90       # SSA value generation
│   └── type_conversion_helpers.f90 # Type conversion utilities
├── codegen/             # AST to MLIR conversion
│   ├── program_gen.f90       # Program/module generation
│   ├── function_gen.f90      # Function/subroutine generation
│   ├── statement_gen.f90     # Statement generation
│   └── expression_gen.f90    # Expression generation
├── passes/              # Pass management
│   ├── ffc_pass_manager.f90  # Pass manager integration
│   └── lowering_pipeline.f90 # HLFIR→FIR→LLVM lowering
├── backend/             # Backend integration
│   ├── mlir_c_backend.f90    # C API backend implementation
│   └── backend_factory.f90   # Backend selection
└── utils/               # Utilities
    ├── memory_tracker.f90    # Memory usage tracking
    ├── memory_guard.f90      # RAII resource management
    └── resource_manager.f90  # Resource lifecycle management

test/
├── test_harness.f90          # Test framework
├── comprehensive_test_runner.f90 # Full test suite
├── performance_benchmarks.f90     # Performance tests
└── test_type_conversion_validation.f90 # Type validation
```

## Development Workflow

### 1. Test-Driven Development (TDD)

Always follow the RED-GREEN-REFACTOR cycle:

#### RED Phase: Write Failing Tests
```fortran
function test_new_feature() result(passed)
    logical :: passed
    ! Test the feature that doesn't exist yet
    passed = .false.
    error stop "Feature not implemented - expected in RED phase"
end function test_new_feature
```

#### GREEN Phase: Minimal Implementation
```fortran
function test_new_feature() result(passed)
    logical :: passed
    type(new_feature_t) :: feature
    
    call feature%init()
    passed = feature%is_working()
    call feature%cleanup()
end function test_new_feature
```

#### REFACTOR Phase: Improve Implementation
```fortran
! REFACTOR: Enhanced implementation with better error handling
subroutine feature_init(this)
    class(new_feature_t), intent(inout) :: this
    integer :: stat
    
    call this%validate_preconditions()
    
    allocate(this%resources, stat=stat)
    if (stat /= 0) then
        this%initialized = .false.
        return
    end if
    
    this%initialized = .true.
end subroutine feature_init
```

### 2. Memory Management

Always use RAII patterns:

```fortran
subroutine process_with_resources()
    type(memory_guard_t) :: guard
    type(mlir_context_t) :: context
    
    call guard%init()
    
    context = create_mlir_context()
    call guard%register_resource(context, "context")
    
    ! Use context...
    
    ! Automatic cleanup when guard goes out of scope
end subroutine process_with_resources
```

### 3. C API Usage Patterns

#### Context Management
```fortran
! Always check validity
context = create_mlir_context()
if (.not. context%is_valid()) then
    error stop "Failed to create MLIR context"
end if

! Register required dialects
call register_func_dialect(context)
call register_hlfir_dialect(context)
call register_fir_dialect(context)
```

#### Operation Creation Pattern
```fortran
function create_operation_pattern(builder, name, location) result(op)
    type(mlir_builder_t), intent(in) :: builder
    character(len=*), intent(in) :: name
    type(mlir_location_t), intent(in) :: location
    type(mlir_operation_t) :: op
    
    type(mlir_operation_state_t) :: state
    type(mlir_string_ref_t) :: op_name
    
    ! 1. Create operation state
    op_name = create_string_ref(name)
    state = create_operation_state(op_name, location)
    
    ! 2. Add operands/results/attributes
    ! call state%add_operand(...)
    ! call state%add_result_type(...)
    ! call state%add_attribute(...)
    
    ! 3. Create operation
    op = create_operation(state)
    
    ! 4. Verify operation
    if (.not. verify_operation(op)) then
        error stop "Invalid operation created"
    end if
end function create_operation_pattern
```

## Code Style Guidelines

### 1. Naming Conventions

```fortran
! Types: snake_case with _t suffix
type :: mlir_context_t
type :: ssa_manager_t

! Functions: snake_case
function create_mlir_context() result(context)
subroutine destroy_mlir_context(context)

! Constants: UPPER_CASE
integer, parameter :: MAX_SSA_VALUES = 10000
integer, parameter :: RESOURCE_CONTEXT = 1

! Variables: snake_case
type(mlir_context_t) :: main_context
integer :: ssa_counter
```

### 2. FFC Module Naming Convention

All ffc modules use the `ffc_` prefix to prevent naming conflicts with external dependencies:

```fortran
! Module names: ffc_<descriptive_name>
module ffc_error_handling
module ffc_pass_manager
module ffc_type_converter

! Import without conflicts
use ffc_error_handling, only: make_error
use fortfront_pass_manager, only: semantic_pass  ! External dependency
```

**Why ffc_ prefix:**
- Prevents module name conflicts with fortfront and other dependencies
- Clearly identifies ffc-owned modules in import statements
- Enables compilation with external Fortran projects
- Follows namespace best practices for library development

### 3. Documentation

```fortran
! REFACTOR: Enhanced function with comprehensive documentation
!
! Purpose: Creates an HLFIR declare operation for variable declaration
! 
! Parameters:
!   builder - MLIR builder for operation insertion
!   memref - Memory reference to the variable storage
!   var_name - Unique name for the variable
!
! Returns:
!   HLFIR declare operation with proper typing
!
! Example:
!   declare_op = create_hlfir_declare(builder, alloca_op, "x")
!
function create_hlfir_declare(builder, memref, var_name) result(declare_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: memref
    character(len=*), intent(in) :: var_name
    type(mlir_operation_t) :: declare_op
```

### 4. Error Handling

```fortran
! Always validate inputs
if (.not. context%is_valid()) then
    error stop "Invalid MLIR context"
end if

! Check allocations
allocate(resources(count), stat=stat)
if (stat /= 0) then
    error stop "Failed to allocate resources"
end if

! Verify operations
op = create_operation(state)
if (.not. verify_operation(op)) then
    call print_operation_errors(op)
    error stop "Invalid operation"
end if
```

## Testing

### 1. Unit Tests

Each module must have comprehensive unit tests:

```fortran
program test_my_module
    use test_harness
    use my_module
    
    type(test_suite_t) :: suite
    
    suite = create_test_suite("My Module Tests")
    
    call add_test_case(suite, "Basic functionality", test_basic_function)
    call add_test_case(suite, "Error handling", test_error_cases)
    call add_test_case(suite, "Memory management", test_memory_safety)
    
    call run_test_suite(suite)
end program test_my_module
```

### 2. Integration Tests

Test full compilation pipeline:

```fortran
function test_hello_world_compilation() result(passed)
    logical :: passed
    type(mlir_c_backend_t) :: backend
    type(backend_options_t) :: options
    character(len=:), allocatable :: output
    character(len=1024) :: error_msg
    
    call backend%init()
    
    ! Test HLFIR generation
    options%emit_hlfir = .true.
    call backend%generate_code(arena, prog_index, options, output, error_msg)
    passed = (len_trim(error_msg) == 0) .and. verify_hlfir_output(output)
    
    ! Test FIR lowering
    options%emit_hlfir = .false.
    options%emit_fir = .true.
    call backend%generate_code(arena, prog_index, options, output, error_msg)
    passed = passed .and. verify_fir_output(output)
    
    call backend%cleanup()
end function test_hello_world_compilation
```

### 3. Performance Tests

Monitor performance with benchmarks:

```fortran
function benchmark_operation_creation() result(passed)
    logical :: passed
    integer, parameter :: ITERATIONS = 10000
    real :: start_time, end_time, elapsed
    
    call cpu_time(start_time)
    
    do i = 1, ITERATIONS
        op = create_test_operation()
    end do
    
    call cpu_time(end_time)
    elapsed = end_time - start_time
    
    print '(A,I0,A,F8.3,A)', "Created ", ITERATIONS, " operations in ", elapsed, " seconds"
    print '(A,F8.3,A)', "Rate: ", real(ITERATIONS)/elapsed, " ops/second"
    
    passed = elapsed < 5.0  ! Should be fast
end function benchmark_operation_creation
```

## Debugging

### 1. MLIR Verification

```fortran
! Enable verification in debug builds
call context%enable_verifier()

! Verify operations after creation
op = create_operation(state)
if (.not. verify_operation(op)) then
    call dump_operation(op)
    call print_verification_errors(op)
    error stop "Operation verification failed"
end if
```

### 2. Memory Debugging

```fortran
use memory_tracker

type(memory_tracker_t) :: tracker
call tracker%init()
call tracker%enable_peak_tracking()

! ... perform operations ...

if (tracker%has_memory_leaks()) then
    call tracker%print_leak_report()
end if

call tracker%print_statistics()
```

### 3. Performance Profiling

```fortran
! Use CPU time for basic profiling
real :: start_time, end_time
call cpu_time(start_time)

! ... operations to profile ...

call cpu_time(end_time)
print *, "Operation took:", end_time - start_time, "seconds"
```

## Contributing

### 1. Pull Request Process

1. **Follow TDD**: Write tests first (RED phase)
2. **Implement minimally**: Make tests pass (GREEN phase)  
3. **Refactor**: Improve implementation (REFACTOR phase)
4. **Update documentation**: Keep docs current
5. **Run full test suite**: Ensure no regressions

### 2. Code Review Checklist

- [ ] Follows TDD methodology
- [ ] Uses MLIR C API exclusively (no text generation)
- [ ] Proper memory management with RAII
- [ ] Comprehensive tests with good coverage
- [ ] Error handling for all failure cases
- [ ] Performance considerations
- [ ] Documentation updated
- [ ] No fortfront API limitations introduced

### 3. Commit Message Format

```
feat: implement Epic X.Y - Feature Name

RED/GREEN/REFACTOR TDD Implementation:
- RED: Created failing tests for new functionality
- GREEN: Implemented minimal working solution using MLIR C API
- REFACTOR: Enhanced with better error handling and performance

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Common Patterns

### 1. Module Template

```fortran
module new_module
    use iso_c_binding
    use mlir_c_core
    implicit none
    private
    
    public :: new_type_t
    
    type :: new_type_t
        private
        logical :: initialized = .false.
        ! ... private members ...
    contains
        procedure :: init => new_init
        procedure :: cleanup => new_cleanup
        procedure :: is_initialized => new_is_initialized
        ! ... public interface ...
    end type new_type_t
    
contains

    subroutine new_init(this)
        class(new_type_t), intent(inout) :: this
        
        if (this%initialized) return
        
        ! ... initialization ...
        
        this%initialized = .true.
    end subroutine new_init
    
    subroutine new_cleanup(this)
        class(new_type_t), intent(inout) :: this
        
        ! ... cleanup resources ...
        
        this%initialized = .false.
    end subroutine new_cleanup
    
    function new_is_initialized(this) result(is_init)
        class(new_type_t), intent(in) :: this
        logical :: is_init
        
        is_init = this%initialized
    end function new_is_initialized

end module new_module
```

### 2. Test Template

```fortran
program test_new_module
    use test_harness
    use new_module
    
    type(test_suite_t) :: suite
    
    print *, "=== New Module Tests ==="
    
    suite = create_test_suite("New Module")
    
    call add_test_case(suite, "Initialization", test_init)
    call add_test_case(suite, "Core functionality", test_core)
    call add_test_case(suite, "Error handling", test_errors)
    call add_test_case(suite, "Memory management", test_memory)
    
    call run_test_suite(suite, verbose=.true.)
    
    call suite%cleanup()
    
contains

    function test_init() result(passed)
        logical :: passed
        type(new_type_t) :: obj
        
        call obj%init()
        passed = obj%is_initialized()
        call obj%cleanup()
    end function test_init
    
    ! ... other test functions ...

end program test_new_module
```

## Performance Considerations

1. **Minimize C API calls**: Batch operations when possible
2. **Reuse contexts**: Don't create/destroy contexts unnecessarily  
3. **Memory pools**: Use memory guards for automatic cleanup
4. **Type caching**: Cache frequently used types
5. **SSA efficiency**: Use SSA manager for proper value naming
6. **Pass optimization**: Configure pass pipelines for best performance

## Troubleshooting

### Common Issues

1. **Context not registered**: Ensure all required dialects are registered
2. **Memory leaks**: Use memory_guard for automatic cleanup
3. **Invalid operations**: Always verify operations after creation
4. **Type mismatches**: Use type converter for consistent types
5. **Performance issues**: Use benchmarks to identify bottlenecks

### Debug Mode

```fortran
! Enable debug mode for additional validation
#ifdef DEBUG
    call context%enable_verifier()
    call tracker%enable_detailed_tracking()
#endif
```

This developer guide provides the foundation for contributing to FortFC while maintaining our strict TDD methodology and MLIR C API exclusive approach.