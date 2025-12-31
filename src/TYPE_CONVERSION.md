# fortfc Type Conversion Design

## Overview

This document describes how fortfc converts fortfront's type information into MLIR type descriptors. All MLIR-specific logic resides here in fortfc, keeping fortfront purely focused on Fortran semantics.

## Type Conversion Architecture

### Type Converter Module
```fortran
module fortfc_type_converter
    use fortfront  ! For type_info_t
    use mlir_c_api
    implicit none
    
    type :: mlir_type_converter_t
        type(mlir_context_t) :: context
    contains
        procedure :: convert_type
        procedure :: get_mlir_type_string
    end type
    
contains
    
    ! Convert fortfront type_info to MLIR type
    function convert_type(this, type_info) result(mlir_type)
        class(mlir_type_converter_t), intent(in) :: this
        type(type_info_t), intent(in) :: type_info
        type(mlir_type_t) :: mlir_type
        
        select case(type_info%base_type)
        case(TINT)
            mlir_type = create_integer_type(this%context, type_info%bit_width)
        case(TREAL)
            mlir_type = create_float_type(this%context, type_info%bit_width)
        case(TLOGICAL)
            mlir_type = create_integer_type(this%context, 1)  ! i1 for bool
        case(TCHAR)
            mlir_type = create_fir_char_type(this%context, type_info%bit_width/8)
        case(TARRAY)
            mlir_type = create_array_type(this, type_info)
        end select
    end function
    
    ! Get MLIR type descriptor string
    function get_mlir_type_string(this, type_info) result(descriptor)
        class(mlir_type_converter_t), intent(in) :: this
        type(type_info_t), intent(in) :: type_info
        character(len=:), allocatable :: descriptor
        
        select case(type_info%base_type)
        case(TINT)
            descriptor = "i" // int_to_str(type_info%bit_width)
        case(TREAL)
            descriptor = "f" // int_to_str(type_info%bit_width)
        case(TLOGICAL)
            descriptor = "i1"
        case(TCHAR)
            descriptor = "!fir.char<1," // int_to_str(type_info%bit_width/8) // ">"
        case(TARRAY)
            descriptor = get_array_descriptor(this, type_info)
        end select
    end function
    
end module
```

## Type Mapping Rules

### Basic Types

| Fortran Type | fortfront type_info | MLIR Type | Notes |
|--------------|-------------------|-----------|-------|
| INTEGER*1 | base_type=TINT, bit_width=8 | i8 | |
| INTEGER*2 | base_type=TINT, bit_width=16 | i16 | |
| INTEGER*4 | base_type=TINT, bit_width=32 | i32 | Default INTEGER |
| INTEGER*8 | base_type=TINT, bit_width=64 | i64 | |
| REAL*4 | base_type=TREAL, bit_width=32 | f32 | Default REAL |
| REAL*8 | base_type=TREAL, bit_width=64 | f64 | DOUBLE PRECISION |
| LOGICAL | base_type=TLOGICAL | i1 | |
| CHARACTER(n) | base_type=TCHAR, bit_width=n*8 | !fir.char<1,n> | |
| COMPLEX*8 | base_type=TCOMPLEX, bit_width=64 | !fir.complex<4> | |
| COMPLEX*16 | base_type=TCOMPLEX, bit_width=128 | !fir.complex<8> | |

### Array Types

Arrays are represented in HLFIR/FIR with shape information:

```fortran
! Fixed-size array
INTEGER, DIMENSION(10) :: arr
! fortfront: base_type=TARRAY, element_type=TINT(32), array_dims=[10]
! MLIR: !fir.array<10xi32>

! Multi-dimensional array
REAL, DIMENSION(5,10) :: matrix  
! fortfront: base_type=TARRAY, element_type=TREAL(32), array_dims=[5,10]
! MLIR: !fir.array<5x10xf32>

! Assumed-shape array
REAL, DIMENSION(:,:) :: dyn_matrix
! fortfront: base_type=TARRAY, element_type=TREAL(32), array_rank=2, array_dims=[-1,-1]
! MLIR: !fir.box<!fir.array<?x?xf32>>

! Allocatable array
REAL, ALLOCATABLE, DIMENSION(:) :: alloc_arr
! fortfront: base_type=TARRAY, is_allocatable=true, element_type=TREAL(32)
! MLIR: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
```

### Derived Types

```fortran
TYPE :: person
    CHARACTER(20) :: name
    INTEGER :: age
END TYPE

! fortfront: base_type=TDERIVED, derived_type_name="person"
! MLIR: !fir.type<_QTperson{name:!fir.char<1,20>,age:i32}>
```

### Pointer Types

```fortran
INTEGER, POINTER :: ptr
! fortfront: base_type=TINT, is_pointer=true
! MLIR: !fir.ref<!fir.box<!fir.ptr<i32>>>

REAL, POINTER, DIMENSION(:) :: ptr_array
! fortfront: base_type=TARRAY, is_pointer=true, element_type=TREAL(32)
! MLIR: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
```

### Function Types

```fortran
! Function returning INTEGER
FUNCTION add(a, b)
    INTEGER :: a, b
    INTEGER :: add
END FUNCTION

! fortfront function_signature_t:
!   param_types = [TINT(32), TINT(32)]
!   return_type = TINT(32)
! MLIR: (i32, i32) -> i32
```

## Reference Types and Memory

HLFIR/FIR uses different reference types:

1. **!fir.ref<T>** - Simple reference to memory containing T
2. **!fir.box<T>** - Descriptor for arrays/pointers with runtime info
3. **!fir.heap<T>** - Heap allocated memory
4. **!fir.ptr<T>** - Fortran pointer target

## Implementation Guidelines

1. All MLIR-specific type knowledge stays in fortfc
2. Use fortfront's type_info_t as the source of truth
3. Handle special cases (allocatable, pointer) with type wrappers
4. Cache converted types for performance
5. Validate type conversions match flang's conventions

## Testing Strategy

1. Create unit tests for each type conversion
2. Compare generated types with flang's output
3. Test edge cases (zero-size arrays, etc.)
4. Validate memory reference types
5. Test derived type conversions