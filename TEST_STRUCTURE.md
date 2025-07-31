# ffc Test Structure

## Overview
The ffc test suite contains 88 tests organized into functional categories:
- 64 active tests
- 24 disabled tests (marked with .disabled extension)

## Test Categories

### Core MLIR C API Tests (13 tests)
Foundation tests for MLIR C API bindings:
- `test_mlir_c_core.f90` - Context, module, location creation
- `test_mlir_c_types.f90` - Type system bindings  
- `test_mlir_c_attributes.f90` - Attribute system
- `test_mlir_c_operations.f90` - Operation builders
- `test_mlir_c_builder.f90` - Builder infrastructure
- `test_mlir_c_type_factory.f90` - Type factory patterns
- `test_mlir_c_attribute_builder.f90` - Attribute builder patterns
- `test_mlir_c_operation_builder.f90` - Operation builder patterns
- `test_mlir_builder.f90` - High-level builder
- `test_mlir_builder_simple.f90` - Simple builder patterns
- `test_builder_minimal.f90` - Minimal builder functionality
- `test_ssa_manager.f90` - SSA value management
- `test_helpers.f90` - Test utilities

### Dialect Tests (9 tests)
Dialect-specific functionality:
- `test_fir_dialect.f90` - FIR dialect operations
- `test_fir_dialect_builder.f90` - FIR operation builders
- `test_hlfir_dialect.f90` - HLFIR dialect operations
- `test_hlfir_program_generation.f90` - HLFIR program generation
- `test_standard_dialects.f90` - Standard dialect integration
- `test_dialect_registry.f90` - Dialect registration
- `test_dialect_shared.f90` - Shared dialect utilities
- `test_string_type.f90` - String type handling
- `test_backend_interface.f90` - Backend interface

### Type System Tests (6 tests)
Type conversion and management:
- `test_type_converter_simple.f90` - Basic type conversions
- `test_type_converter_expanded.f90` - Advanced type conversions
- `test_type_helpers.f90` - Type helper functions
- `test_type_helpers_simple.f90` - Simple type helpers
- Backend tests:
  - `backend/test_backend_factory.f90` - Backend factory
  - `backend/test_fortran_backend.f90` - Fortran backend

### MLIR Generation Tests (40 active, 24 disabled)
End-to-end MLIR code generation:

#### Active Generation Tests (40)
- `mlir/test_basic_generation.f90` - Basic MLIR generation
- `mlir/test_ast_mapping.f90` - AST to MLIR mapping
- `mlir/test_types.f90` - Type integration
- `mlir/test_optimization.f90` - Optimization passes
- `mlir/test_llvm_lowering.f90` - LLVM lowering
- `mlir/test_enzyme_ad.f90` - Enzyme automatic differentiation
- `mlir/test_mlir_generation.f90` - MLIR generation core
- `mlir/test_mlir_harness.f90` - Test harness functionality
- `mlir/test_mlir_allocate_deallocate.f90` - Memory management
- `mlir/test_format_descriptor_parsing.f90` - Format parsing

Array Support (6):
- `mlir/test_array_support.f90` - Array operations
- `mlir/test_array_literals.f90` - Array literals
- `mlir/test_array_intrinsics.f90` - Array intrinsics
- `mlir/test_array_allocate_debug.f90` - Array allocation debugging
- `mlir/test_array_allocate_green.f90` - Array allocation (GREEN)
- `mlir/test_array_allocate_red.f90` - Array allocation (RED)

Control Flow (7):
- `mlir/test_do_while_loops.f90` - Do-while loops
- `mlir/test_named_do_loops.f90` - Named do loops
- `mlir/test_select_case.f90` - Select case statements
- `mlir/test_where_constructs.f90` - Where constructs
- `mlir/test_forall_constructs.f90` - Forall constructs
- `mlir/test_exit_cycle_statements.f90` - Exit/cycle statements
- `mlir/test_assignment_debug.f90` - Assignment debugging

Assignment Overloading (2):
- `mlir/test_assignment_overload_green.f90` - Assignment overload (GREEN)
- `mlir/test_assignment_overload_red.f90` - Assignment overload (RED)

I/O Operations (4):
- `mlir/test_io_operations.f90` - Basic I/O operations
- `mlir/test_io_proper_implementation.f90` - Proper I/O implementation
- `mlir/test_io_statement_options.f90` - I/O statement options
- `mlir/test_iostat_err_end_specifiers.f90` - IOSTAT/ERR/END specifiers
- `mlir/test_hlfir_read_statement.f90` - HLFIR read statements

Data Types (4):
- `mlir/test_character_strings.f90` - Character strings
- `mlir/test_complex_numbers.f90` - Complex numbers
- `mlir/test_derived_types.f90` - Derived types
- `mlir/test_subroutines_functions.f90` - Functions/subroutines

Module System (2):
- `mlir/test_module_generation.f90` - Module generation
- `mlir/test_module_system.f90` - Module system

#### Disabled Generation Tests (24)
Tests disabled due to incomplete functionality or dependencies:

Memory Management:
- `mlir/test_allocate_deallocate.f90.disabled` - Allocate/deallocate
- `mlir/test_arithmetic_operations.f90.disabled` - Arithmetic operations

HLFIR Support:
- `mlir/test_hlfir_generation.f90.disabled` - HLFIR generation
- `mlir/test_hlfir_generation_simple.f90.disabled` - Simple HLFIR generation
- `mlir/test_hlfir_compile_command.f90.disabled` - HLFIR compilation
- `mlir/test_hlfir_write_statement.f90.disabled` - HLFIR write statements

Pointer Support:
- `mlir/test_pointer_assignment.f90.disabled` - Pointer assignment
- `mlir/test_pointer_declarations.f90.disabled` - Pointer declarations
- `mlir/test_associated_intrinsic.f90.disabled` - ASSOCIATED intrinsic

Advanced Features:
- `mlir/test_generic_procedures.f90.disabled` - Generic procedures
- `mlir/test_interface_blocks.f90.disabled` - Interface blocks
- `mlir/test_operator_overloading.f90.disabled` - Operator overloading

I/O Runtime:
- `mlir/test_io_runtime_integration.f90.disabled` - I/O runtime integration
- `mlir/test_io_runtime_library.f90.disabled` - I/O runtime library
- `mlir/test_simple_io_runtime.f90.disabled` - Simple I/O runtime
- `mlir/test_multi_argument_print.f90.disabled` - Multi-argument print
- `mlir/test_print_statement_full.f90.disabled` - Full print statement

LLVM Integration:
- `mlir/test_mlir_to_llvm_lowering.f90.disabled` - MLIR to LLVM lowering
- `mlir/test_llvm_to_object.f90.disabled` - LLVM to object code
- `mlir/test_object_to_executable.f90.disabled` - Object to executable

Error Handling:
- `mlir/test_mlir_error_reporting.f90.disabled` - Error reporting
- `mlir/test_mlir_generation_proper.f90.disabled` - Proper MLIR generation

Utilities:
- `mlir/test_string_constants.f90.disabled` - String constants
- `mlir/test_unknown_module_lookup.f90.disabled` - Unknown module lookup

## Test Configuration

### Test Harness
- `mlir/mlir_test_config.toml` - MLIR test configuration
- `mlir/run_mlir_tests.sh` - MLIR test runner script

### Test Organization Improvements Made
1. Fixed misplaced test file in nested directory
2. Removed empty nested directories
3. Created comprehensive test documentation

## Epic Progress Mapping

Based on BACKLOG.md:
- **Epic 1-3**: Complete (Foundation, Dialects, IR Builder) ✅
- **Epic 4**: In Progress (AST to MLIR Conversion) 🟡
  - Program/Module Generation: Partially complete
  - Function Generation: Partially complete  
  - Statement Generation: In progress
  - Expression Generation: In progress

## Test Execution Strategy

Run all tests:
```bash
fpm test
```

Run specific test:
```bash
fpm test <test_name>
```

Enable disabled tests by removing .disabled extension when ready.