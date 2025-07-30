# Additional fortfront API Requirements for fortfc

## Overview

This document outlines additional API requirements needed from fortfront to enable fortfc to use MLIR C bindings directly instead of generating MLIR text. These requirements extend the base API defined in ../fluff/FORTFRONT.md.

## 1. Enhanced AST Node Access

### Direct Node Access by Index
```fortran
! Get typed node directly from arena (avoiding polymorphism overhead)
function get_node_as_program(arena, index) result(node)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: index
    type(program_node), pointer :: node
end function

function get_node_as_function_def(arena, index) result(node)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: index
    type(function_def_node), pointer :: node
end function

! Similar functions for all node types...
```

### Node Type Queries
```fortran
! Get the specific node type as an enum
function get_node_type(arena, index) result(node_type)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: index
    integer :: node_type  ! NODE_PROGRAM, NODE_FUNCTION_DEF, etc.
end function

! Node type constants
integer, parameter :: NODE_PROGRAM = 1
integer, parameter :: NODE_FUNCTION_DEF = 2
integer, parameter :: NODE_ASSIGNMENT = 3
! ... etc for all node types
```

## 2. Type System Enhancements

### Detailed Type Information
```fortran
type :: type_info_t
    integer :: base_type        ! TINT, TREAL, etc.
    integer :: bit_width        ! 32, 64, etc.
    logical :: is_signed        ! For integers
    integer :: array_rank       ! 0 for scalars
    integer, allocatable :: array_dims(:)  ! Shape if known
    logical :: is_allocatable
    logical :: is_pointer
    character(len=:), allocatable :: derived_type_name
end type type_info_t

! Get detailed type information for a node
function get_node_type_info(ctx, arena, node_index) result(type_info)
    type(semantic_context_t), intent(in) :: ctx
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: node_index
    type(type_info_t) :: type_info
end function
```

### Type Conversion Utilities
```fortran
! Convert fortfront type to MLIR type descriptor
function get_mlir_type_descriptor(type_info) result(descriptor)
    type(type_info_t), intent(in) :: type_info
    character(len=:), allocatable :: descriptor  ! e.g., "i32", "f64", "!fir.array<10xi32>"
end function
```

## 3. Symbol Table Access

### Direct Symbol Lookup
```fortran
type :: symbol_info_t
    character(len=:), allocatable :: name
    integer :: declaration_node_index
    integer :: scope_level
    type(type_info_t) :: type_info
    logical :: is_parameter
    logical :: is_module_variable
    logical :: is_function
    logical :: is_subroutine
end type symbol_info_t

! Get symbol information by name
function lookup_symbol(ctx, name, scope_node_index) result(symbol)
    type(semantic_context_t), intent(in) :: ctx
    character(len=*), intent(in) :: name
    integer, intent(in) :: scope_node_index
    type(symbol_info_t) :: symbol
end function

! Get all symbols in a scope
function get_scope_symbols(ctx, scope_node_index) result(symbols)
    type(semantic_context_t), intent(in) :: ctx
    integer, intent(in) :: scope_node_index
    type(symbol_info_t), allocatable :: symbols(:)
end function
```

## 4. Module and Use Statement Information

### Module Dependencies
```fortran
type :: module_info_t
    character(len=:), allocatable :: name
    character(len=:), allocatable :: file_path
    integer :: module_node_index
    character(len=:), allocatable :: used_symbols(:)
end type module_info_t

! Get information about used modules
function get_used_modules(arena, prog_index) result(modules)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: prog_index
    type(module_info_t), allocatable :: modules(:)
end function
```

## 5. Intrinsic Function Information

### Intrinsic Recognition
```fortran
! Check if a function call is to an intrinsic
function is_intrinsic_call(name) result(is_intrinsic)
    character(len=*), intent(in) :: name
    logical :: is_intrinsic
end function

! Get intrinsic function signature
function get_intrinsic_signature(name) result(signature)
    character(len=*), intent(in) :: name
    type(function_signature_t) :: signature
end function

type :: function_signature_t
    type(type_info_t), allocatable :: param_types(:)
    type(type_info_t) :: return_type
    logical :: is_elemental
    logical :: is_pure
end type function_signature_t
```

## 6. Literal Value Access

### Direct Literal Value Extraction
```fortran
! Get integer literal value
function get_integer_literal_value(arena, node_index) result(value)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: node_index
    integer(kind=8) :: value
end function

! Get real literal value
function get_real_literal_value(arena, node_index) result(value)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: node_index
    real(kind=8) :: value
end function

! Get string literal value without quotes
function get_string_literal_value(arena, node_index) result(value)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: node_index
    character(len=:), allocatable :: value
end function
```

## 7. Control Flow Analysis

### Loop Information
```fortran
type :: loop_info_t
    integer :: loop_variable_index
    integer :: start_expr_index
    integer :: end_expr_index
    integer :: step_expr_index
    logical :: is_do_while
    logical :: is_do_concurrent
    character(len=:), allocatable :: label
end type loop_info_t

function get_loop_info(arena, do_loop_index) result(info)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: do_loop_index
    type(loop_info_t) :: info
end function
```

## 8. Array Operations

### Array Access Information
```fortran
type :: array_access_info_t
    integer :: array_var_index
    integer, allocatable :: subscript_indices(:)
    logical :: is_section  ! Array section vs element
    integer, allocatable :: section_start(:)
    integer, allocatable :: section_end(:)
    integer, allocatable :: section_stride(:)
end type array_access_info_t

function get_array_access_info(arena, subscript_index) result(info)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: subscript_index
    type(array_access_info_t) :: info
end function
```

## 9. Procedure Call Information

### Call Site Analysis
```fortran
type :: call_info_t
    character(len=:), allocatable :: callee_name
    integer :: callee_node_index  ! -1 if external
    integer, allocatable :: argument_indices(:)
    logical :: is_intrinsic
    logical :: is_elemental_call
    type(type_info_t) :: result_type
end type call_info_t

function get_call_info(ctx, arena, call_index) result(info)
    type(semantic_context_t), intent(in) :: ctx
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: call_index
    type(call_info_t) :: info
end function
```

## 10. C Interoperability Support

### ISO C Binding Information
```fortran
! Check if a procedure has BIND(C)
function has_c_binding(arena, proc_index) result(has_binding)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: proc_index
    logical :: has_binding
end function

! Get C binding name if specified
function get_c_binding_name(arena, proc_index) result(name)
    type(ast_arena_t), intent(in) :: arena
    integer, intent(in) :: proc_index
    character(len=:), allocatable :: name
end function
```

## Implementation Priority for fortfc

1. **Critical** - Node type queries and direct typed access
2. **Critical** - Enhanced type information with MLIR descriptors
3. **Critical** - Symbol table access for variable resolution
4. **High** - Literal value extraction
5. **High** - Control flow analysis
6. **Medium** - Array operation information
7. **Medium** - Procedure call analysis
8. **Low** - Module dependency tracking
9. **Low** - C interoperability features

These extensions will enable fortfc to:
- Build MLIR operations programmatically using C bindings
- Generate accurate type information for MLIR
- Properly resolve symbols and scopes
- Handle all Fortran constructs without text generation