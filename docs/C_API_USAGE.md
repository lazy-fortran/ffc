# MLIR C API Usage Guide

## Overview

This guide documents how FortFC uses the MLIR C API exclusively for HLFIR code generation. **No text-based MLIR generation is used** - all MLIR operations are created in-memory using C API calls.

## Architecture

```
Fortran AST → HLFIR (C API) → FIR (C API) → LLVM IR → Object Code
```

All transformations use MLIR C API bindings with proper memory management and error handling.

## Core Components

### 1. MLIR Context Management

```fortran
use mlir_c_core

! Create MLIR context
type(mlir_context_t) :: context
context = create_mlir_context()

! Register required dialects
call register_func_dialect(context)
call register_hlfir_dialect(context)
call register_fir_dialect(context)

! Always cleanup
call destroy_mlir_context(context)
```

### 2. Type System

```fortran
use mlir_c_types

! Create basic types
type(mlir_type_t) :: i32_type, f64_type, i1_type
i32_type = create_i32_type(context)
f64_type = create_f64_type(context)
i1_type = create_i1_type(context)

! Create array types
type(mlir_type_t) :: array_type
array_type = create_array_type(context, [10, 20], f64_type)
! Results in: !fir.array<10x20xf64>

! Create reference types
type(mlir_type_t) :: ref_type
ref_type = create_reference_type(context, f64_type)
! Results in: !fir.ref<f64>
```

### 3. Operation Builder

```fortran
use mlir_c_operations

! Create operation state
type(mlir_operation_state_t) :: state
type(mlir_string_ref_t) :: op_name
op_name = create_string_ref("hlfir.declare")
state = create_operation_state(op_name, location)

! Add operands
call state%add_operand(memref_value)

! Add result types
call state%add_result_type(declared_type)

! Add attributes
type(mlir_attribute_t) :: name_attr
name_attr = create_string_attribute(context, "variable_name")
call state%add_attribute("sym_name", name_attr)

! Create operation
type(mlir_operation_t) :: op
op = create_operation(state)
```

## HLFIR Operations

### Variable Declaration

```fortran
! Generate: %1 = hlfir.declare %0 {uniq_name = "x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
function create_hlfir_declare(builder, memref, var_name) result(declare_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: memref
    character(len=*), intent(in) :: var_name
    type(mlir_operation_t) :: declare_op
    
    type(mlir_operation_state_t) :: state
    type(mlir_location_t) :: loc
    type(mlir_attribute_t) :: name_attr
    
    loc = builder%get_unknown_location()
    state = create_operation_state(create_string_ref("hlfir.declare"), loc)
    
    call state%add_operand(memref)
    call state%add_result_type(memref%get_type())
    call state%add_result_type(memref%get_type())
    
    name_attr = create_string_attribute(builder%get_context(), var_name)
    call state%add_attribute("uniq_name", name_attr)
    
    declare_op = create_operation(state)
end function create_hlfir_declare
```

### Assignment Operations

```fortran
! Generate: hlfir.assign %rhs to %lhs : f64, !fir.ref<f64>
function create_hlfir_assign(builder, rhs, lhs) result(assign_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: rhs, lhs
    type(mlir_operation_t) :: assign_op
    
    type(mlir_operation_state_t) :: state
    type(mlir_location_t) :: loc
    
    loc = builder%get_unknown_location()
    state = create_operation_state(create_string_ref("hlfir.assign"), loc)
    
    call state%add_operand(rhs)
    call state%add_operand(lhs)
    
    assign_op = create_operation(state)
end function create_hlfir_assign
```

### Elemental Operations

```fortran
! Generate: %result = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !fir.array<10xf64>
function create_hlfir_elemental(builder, shape, element_type) result(elemental_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: shape
    type(mlir_type_t), intent(in) :: element_type
    type(mlir_operation_t) :: elemental_op
    
    type(mlir_operation_state_t) :: state
    type(mlir_location_t) :: loc
    type(mlir_region_t) :: body_region
    
    loc = builder%get_unknown_location()
    state = create_operation_state(create_string_ref("hlfir.elemental"), loc)
    
    call state%add_operand(shape)
    call state%add_result_type(element_type)
    
    ! Create body region
    body_region = create_empty_region(builder%get_context())
    call state%add_region(body_region)
    
    elemental_op = create_operation(state)
end function create_hlfir_elemental
```

## Type Conversion Examples

### Fortran Types to MLIR Types

```fortran
! integer :: i          → i32
! integer(8) :: i8      → i64
! real :: r             → f32
! real(8) :: r8         → f64
! logical :: l          → i1
! character(len=10) :: s → !fir.char<1,10>

! Arrays:
! real :: a(10,20)                    → !fir.array<10x20xf32>
! integer :: b(:,:)                   → !fir.box<!fir.array<?x?xi32>>
! real, allocatable :: c(:)           → !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! integer, pointer :: p(:)            → !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

use fortfc_type_converter

type(mlir_type_converter_t) :: converter
call converter%init(context)

! Convert basic types
mlir_type = converter%get_mlir_type_string("integer", 4)  ! → "i32"
mlir_type = converter%get_mlir_type_string("real", 8)     ! → "f64"

! Convert array types
array_desc = get_array_descriptor([10, 20], "real", 4, .false., .false., .false.)
! → "!fir.array<10x20xf32>"

! Convert allocatable arrays
alloc_desc = get_array_descriptor([-1], "integer", 4, .false., .true., .false.)
! → "!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>"
```

## Memory Management

### RAII Pattern

```fortran
use memory_guard

type(memory_guard_t) :: guard
call guard%init()

! Register resources for automatic cleanup
context = create_mlir_context()
call guard%register_resource(context, "main_context")

builder = create_mlir_builder(context)
call guard%register_resource(builder, "main_builder")

! Resources are automatically cleaned up when guard goes out of scope
! or on error conditions
```

### Memory Tracking

```fortran
use memory_tracker

type(memory_tracker_t) :: tracker
call tracker%init()
call tracker%enable_peak_tracking()

! Track allocations
call tracker%record_allocation("context", 1024_8)
call tracker%record_allocation("builder", 512_8)

! Track deallocations
call tracker%record_deallocation("builder", 512_8)

! Check for leaks
if (tracker%has_memory_leaks()) then
    call tracker%print_leak_report()
end if

print *, "Peak memory usage:", tracker%get_peak_usage()
```

## Builder Pattern Usage

### Creating Functions

```fortran
use mlir_builder

type(mlir_builder_t) :: builder
type(mlir_module_t) :: module
type(mlir_operation_t) :: func_op

builder = create_mlir_builder(context)
module = create_empty_module(create_unknown_location(context))

! Create function signature
type(mlir_type_t) :: func_type
func_type = create_function_type(context, [], [create_i32_type(context)])

! Create function operation
func_op = create_func_op(builder, "main", func_type, location)

! Get entry block
type(mlir_block_t) :: entry_block
entry_block = func_op%get_entry_block()

! Set insertion point
call builder%set_insertion_point_to_start(entry_block)

! Insert operations...
```

### SSA Value Management

```fortran
use ssa_manager

type(ssa_manager_t) :: ssa_mgr
call ssa_mgr%init()

! Generate SSA values
character(len=:), allocatable :: value_name
value_name = ssa_mgr%next_value()  ! → "%1"
value_name = ssa_mgr%next_value()  ! → "%2"

! Track value types
call ssa_mgr%set_value_type(value_name, f64_type)
stored_type = ssa_mgr%get_value_type(value_name)
```

## Error Handling

### Context Validation

```fortran
context = create_mlir_context()
if (.not. context%is_valid()) then
    error stop "Failed to create MLIR context"
end if
```

### Operation Verification

```fortran
op = create_operation(state)
if (.not. verify_operation(op)) then
    call print_operation_errors(op)
    error stop "Invalid operation created"
end if
```

### Resource Management

```fortran
type(resource_manager_t) :: manager
call manager%init()

! Register resources
pm = create_pass_manager(context)
call manager%register_pass_manager(pm, "main_pm")

! Cleanup specific resource
call manager%cleanup_resource("main_pm")

! Verify all resources freed
if (.not. manager%verify_all_freed()) then
    call manager%print_detailed_report()
    error stop "Resource leak detected"
end if
```

## Best Practices

1. **Always use C API**: Never generate MLIR as text strings
2. **Proper resource management**: Use RAII patterns and cleanup resources
3. **Verify operations**: Check operation validity after creation
4. **Memory tracking**: Monitor memory usage in large compilations
5. **Error handling**: Check return values and handle errors gracefully
6. **Type safety**: Use type converter for consistent type generation
7. **SSA management**: Use SSA manager for proper value naming
8. **Performance**: Use performance benchmarks to monitor efficiency

## Integration Example

```fortran
program example_usage
    use mlir_c_core
    use mlir_builder
    use hlfir_dialect
    use memory_guard
    
    type(memory_guard_t) :: guard
    type(mlir_context_t) :: context
    type(mlir_builder_t) :: builder
    type(mlir_module_t) :: module
    
    ! Initialize with automatic cleanup
    call guard%init()
    
    ! Create MLIR infrastructure
    context = create_mlir_context()
    call guard%register_resource(context, "context")
    
    builder = create_mlir_builder(context)
    call guard%register_resource(builder, "builder")
    
    ! Register dialects
    call register_hlfir_dialect(context)
    call register_fir_dialect(context)
    
    ! Create module and generate HLFIR
    module = create_empty_module(builder%get_unknown_location())
    call generate_hlfir_program(builder, module)
    
    ! Resources cleaned up automatically by guard destructor
end program example_usage
```

This demonstrates the complete MLIR C API usage pattern with proper memory management and HLFIR generation.