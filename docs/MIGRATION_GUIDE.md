# Migration Guide: Text Generation to MLIR C API

## Overview

This guide helps developers migrate from text-based MLIR generation to using the MLIR C API exclusively. This migration is essential for performance, memory safety, and maintainability.

## Key Changes

### Before: Text-Based Generation

```fortran
! OLD: Text generation approach (WRONG)
character(len=1024) :: mlir_code
mlir_code = "func.func @main() -> i32 {"
mlir_code = trim(mlir_code) // new_line('a') // "  %0 = arith.constant 42 : i32"
mlir_code = trim(mlir_code) // new_line('a') // "  func.return %0 : i32"
mlir_code = trim(mlir_code) // new_line('a') // "}"
! Then somehow parse this text back into MLIR...
```

### After: MLIR C API

```fortran
! NEW: C API approach (CORRECT)
use mlir_builder
use memory_guard

type(memory_guard_t) :: guard
type(mlir_context_t) :: context
type(mlir_builder_t) :: builder
type(mlir_module_t) :: module

call guard%init()

context = create_mlir_context()
call guard%register_resource(context, "context")

builder = create_mlir_builder(context)
call guard%register_resource(builder, "builder")

! Create operations in-memory using C API
module = create_empty_module(builder%get_unknown_location())
func_op = create_func_op(builder, "main", func_type, location)

! Resources cleaned up automatically
```

## Migration Steps

### Step 1: Replace Text Generation

**Find patterns like:**
```fortran
! Pattern to REMOVE
write(mlir_unit, '(A)') "hlfir.declare"
write(mlir_unit, '(A,A,A)') "  %", trim(var_name), " = hlfir.declare"
```

**Replace with:**
```fortran
! Pattern to USE
use hlfir_dialect
declare_op = create_hlfir_declare(builder, memref, var_name)
```

### Step 2: Update Function Signatures

**Before:**
```fortran
subroutine generate_mlir_function(output_file, function_name)
    character(len=*), intent(in) :: output_file, function_name
    integer :: unit
    open(newunit=unit, file=output_file, action='write')
    write(unit, '(A)') "func.func @" // trim(function_name) // "() {"
    ! ... more text generation
    close(unit)
end subroutine
```

**After:**
```fortran
function create_mlir_function(builder, function_name) result(func_op)
    type(mlir_builder_t), intent(in) :: builder
    character(len=*), intent(in) :: function_name
    type(mlir_operation_t) :: func_op
    
    type(mlir_type_t) :: func_type
    type(mlir_location_t) :: location
    
    func_type = create_function_type(builder%get_context(), [], [])
    location = builder%get_unknown_location()
    func_op = create_func_op(builder, function_name, func_type, location)
end function
```

### Step 3: Update Type Generation

**Before:**
```fortran
! Pattern to REMOVE
function get_fortran_type_mlir(fort_type, kind) result(mlir_type_str)
    character(len=*), intent(in) :: fort_type
    integer, intent(in) :: kind
    character(len=64) :: mlir_type_str
    
    select case (fort_type)
    case ("integer")
        write(mlir_type_str, '(A,I0)') "i", kind * 8
    case ("real")
        if (kind == 4) then
            mlir_type_str = "f32"
        else
            mlir_type_str = "f64"
        end if
    end select
end function
```

**After:**
```fortran
! Pattern to USE
use fortfc_type_converter

type(mlir_type_converter_t) :: converter
type(mlir_type_t) :: mlir_type

call converter%init(context)
mlir_type = converter%create_integer_type(kind * 8)
mlir_type = converter%create_float_type(kind)
```

### Step 4: Update Operation Creation

**Before:**
```fortran
! Pattern to REMOVE
subroutine emit_alloca(unit, var_name, type_str)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: var_name, type_str
    write(unit, '(A)') "  %" // trim(var_name) // "_addr = fir.alloca " // trim(type_str)
end subroutine
```

**After:**
```fortran
! Pattern to USE
function create_fir_alloca(builder, var_type, var_name) result(alloca_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_type_t), intent(in) :: var_type
    character(len=*), intent(in) :: var_name
    type(mlir_operation_t) :: alloca_op
    
    type(mlir_operation_state_t) :: state
    type(mlir_location_t) :: location
    
    location = builder%get_unknown_location()
    state = create_operation_state(create_string_ref("fir.alloca"), location)
    call state%add_result_type(create_reference_type(builder%get_context(), var_type))
    alloca_op = create_operation(state)
end function
```

## Common Patterns

### Pattern 1: Variable Declaration

**Before:**
```fortran
write(unit, '(A)') "  %decl_" // trim(var_name) // " = hlfir.declare %" // &
                   trim(var_name) // "_addr {uniq_name = """ // trim(var_name) // """}"
```

**After:**
```fortran
alloca_op = create_fir_alloca(builder, var_type, var_name)
declare_op = create_hlfir_declare(builder, alloca_op%get_result(0), var_name)
```

### Pattern 2: Assignment

**Before:**
```fortran
write(unit, '(A)') "  hlfir.assign %" // trim(rhs_value) // " to %" // trim(lhs_value)
```

**After:**
```fortran
assign_op = create_hlfir_assign(builder, rhs_value, lhs_value)
```

### Pattern 3: Function Creation

**Before:**
```fortran
write(unit, '(A)') "func.func @" // trim(func_name) // "() -> i32 {"
write(unit, '(A)') "  %0 = arith.constant 0 : i32"
write(unit, '(A)') "  func.return %0 : i32"
write(unit, '(A)') "}"
```

**After:**
```fortran
func_type = create_function_type(context, [], [create_i32_type(context)])
func_op = create_func_op(builder, func_name, func_type, location)

entry_block = func_op%get_entry_block()
call builder%set_insertion_point_to_start(entry_block)

const_op = builder%create_constant_op(create_i32_type(context), 0)
return_op = create_func_return(builder, [const_op%get_result(0)])
```

## Memory Management

### Before: Manual File Handling

```fortran
! OLD: Manual resource management
integer :: unit
open(newunit=unit, file="output.mlir", action='write')
! ... generate text ...
close(unit)
! Memory leaks possible, no cleanup on errors
```

### After: RAII with Guards

```fortran
! NEW: Automatic resource management
type(memory_guard_t) :: guard
call guard%init()

context = create_mlir_context()
call guard%register_resource(context, "context")

builder = create_mlir_builder(context)
call guard%register_resource(builder, "builder")

! Automatic cleanup on scope exit or errors
```

## Error Handling

### Before: Limited Error Handling

```fortran
! OLD: Basic error handling
write(unit, '(A)') mlir_code
if (iostat /= 0) then
    print *, "Error writing MLIR"
    return
end if
```

### After: Comprehensive Validation

```fortran
! NEW: Operation verification
op = create_operation(state)
if (.not. verify_operation(op)) then
    call print_operation_errors(op)
    error stop "Invalid operation created"
end if

! Context validation
if (.not. context%is_valid()) then
    error stop "Invalid MLIR context"
end if
```

## Testing Migration

### Before: Text Comparison

```fortran
! OLD: String comparison testing
expected_mlir = "func.func @main() -> i32 {"
call generate_mlir_function("test.mlir", "main")
read(unit, '(A)') actual_mlir
if (trim(expected_mlir) /= trim(actual_mlir)) then
    test_passed = .false.
end if
```

### After: Operation Validation

```fortran
! NEW: Operation structure testing
func_op = create_mlir_function(builder, "main")
test_passed = verify_operation(func_op)
test_passed = test_passed .and. (func_op%get_name() == "main")
test_passed = test_passed .and. func_op%get_num_results() > 0
```

## Performance Benefits

### Text Generation Issues:
- String concatenation overhead
- File I/O bottlenecks  
- Parsing overhead
- Memory fragmentation
- No type safety

### C API Benefits:
- Direct in-memory operations
- Type-safe operation construction
- Immediate verification
- Better memory management
- Optimal performance

## Debugging

### Before: Text Inspection

```fortran
! OLD: Debug by reading generated text
print *, "Generated MLIR:"
print *, trim(mlir_text)
```

### After: Operation Introspection

```fortran
! NEW: Debug using MLIR introspection
if (debug_mode) then
    call dump_operation(op)
    call print_operation_structure(op)
    call verify_operation_with_diagnostics(op)
end if
```

## Common Pitfalls

### 1. Forgetting Resource Cleanup

```fortran
! WRONG: No cleanup
context = create_mlir_context()
builder = create_mlir_builder(context)
! ... use resources ...
! No cleanup - memory leak!

! RIGHT: Proper cleanup
type(memory_guard_t) :: guard
call guard%init()
context = create_mlir_context()
call guard%register_resource(context, "context")
! Automatic cleanup
```

### 2. Not Validating Operations

```fortran
! WRONG: No validation
op = create_operation(state)
! Use op without checking if it's valid

! RIGHT: Always validate
op = create_operation(state)
if (.not. verify_operation(op)) then
    error stop "Invalid operation"
end if
```

### 3. String-Based Type Handling

```fortran
! WRONG: String manipulation
type_str = "!fir.ref<f64>"
! Parse and manipulate strings

! RIGHT: Type objects
ref_type = create_reference_type(context, create_f64_type(context))
element_type = get_referenced_type(ref_type)
```

## Complete Example

### Before (Text Generation):

```fortran
subroutine old_generate_hello_world(filename)
    character(len=*), intent(in) :: filename
    integer :: unit
    
    open(newunit=unit, file=filename, action='write')
    write(unit, '(A)') 'module {'
    write(unit, '(A)') '  func.func @main() -> i32 {'
    write(unit, '(A)') '    %0 = arith.constant 0 : i32'
    write(unit, '(A)') '    func.return %0 : i32'
    write(unit, '(A)') '  }'
    write(unit, '(A)') '}'
    close(unit)
end subroutine
```

### After (C API):

```fortran
function new_generate_hello_world(context) result(module)
    type(mlir_context_t), intent(in) :: context
    type(mlir_module_t) :: module
    
    type(memory_guard_t) :: guard
    type(mlir_builder_t) :: builder
    type(mlir_operation_t) :: func_op, const_op, return_op
    type(mlir_type_t) :: func_type, i32_type
    type(mlir_location_t) :: location
    type(mlir_block_t) :: entry_block
    
    call guard%init()
    
    builder = create_mlir_builder(context)
    call guard%register_resource(builder, "builder")
    
    location = builder%get_unknown_location()
    module = create_empty_module(location)
    
    ! Create function type and operation
    i32_type = create_i32_type(context)
    func_type = create_function_type(context, [], [i32_type])
    func_op = create_func_op(builder, "main", func_type, location)
    
    ! Add function to module
    call module%add_operation(func_op)
    
    ! Create function body
    entry_block = func_op%get_entry_block()
    call builder%set_insertion_point_to_start(entry_block)
    
    const_op = builder%create_constant_op(i32_type, 0)
    return_op = create_func_return(builder, [const_op%get_result(0)])
    
    ! Verify
    if (.not. verify_operation(func_op)) then
        error stop "Invalid function operation"
    end if
end function
```

This migration guide provides a comprehensive path from text-based generation to the proper MLIR C API approach, ensuring better performance, type safety, and maintainability.