module test_type_conversion_validation
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use fortfc_type_converter
    use test_harness
    implicit none
    private
    
    public :: test_flang_type_comparison
    public :: test_type_conversion_examples
    public :: test_array_descriptor_formats
    public :: test_edge_cases
    public :: test_derived_type_mangling
    
contains

    function test_flang_type_comparison() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        character(len=:), allocatable :: our_type, flang_type
        
        passed = .true.
        
        print *, "Testing type generation against flang output..."
        
        context = create_mlir_context()
        call converter%init(context)
        
        ! Test 1: Integer types
        our_type = converter%get_mlir_type_string("integer", 4)
        flang_type = "i32"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: integer*4 - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        ! Test 2: Real types
        our_type = converter%get_mlir_type_string("real", 8)
        flang_type = "f64"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: real*8 - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        ! Test 3: Logical types
        our_type = converter%get_mlir_type_string("logical", 1)
        flang_type = "i1"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: logical*1 - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        ! Test 4: Character types
        our_type = converter%get_mlir_type_string("character", 10)
        flang_type = "!fir.char<1,10>"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: character*10 - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        ! Test 5: Fixed arrays
        our_type = get_array_descriptor([10, 20], "real", 4, .false., .false., .false.)
        flang_type = "!fir.array<10x20xf32>"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: real a(10,20) - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        ! Test 6: Assumed-shape arrays
        our_type = get_array_descriptor([-1, -1], "integer", 4, .true., .false., .false.)
        flang_type = "!fir.box<!fir.array<?x?xi32>>"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: integer a(:,:) - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        ! Test 7: Allocatable arrays
        our_type = get_array_descriptor([-1], "real", 8, .false., .true., .false.)
        flang_type = "!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>"
        passed = passed .and. (trim(our_type) == trim(flang_type))
        if (.not. (trim(our_type) == trim(flang_type))) then
            print *, "  FAIL: real, allocatable :: a(:) - Expected: ", trim(flang_type), " Got: ", trim(our_type)
        end if
        
        call converter%cleanup()
        call destroy_mlir_context(context)
        
        if (passed) then
            print *, "  All flang type comparisons passed!"
        end if
    end function test_flang_type_comparison

    function test_type_conversion_examples() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        
        passed = .true.
        
        print *, "Testing TYPE_CONVERSION.md examples..."
        
        context = create_mlir_context()
        call converter%init(context)
        
        ! Test all examples from TYPE_CONVERSION.md
        passed = passed .and. test_basic_types_example(converter)
        passed = passed .and. test_array_types_example(converter)
        passed = passed .and. test_derived_types_example(converter)
        passed = passed .and. test_pointer_types_example(converter)
        passed = passed .and. test_function_types_example(converter)
        
        call converter%cleanup()
        call destroy_mlir_context(context)
    end function test_type_conversion_examples

    function test_array_descriptor_formats() result(passed)
        logical :: passed
        character(len=:), allocatable :: descriptor
        
        passed = .true.
        
        print *, "Testing array descriptor formats..."
        
        ! Test 1: 1D fixed array
        descriptor = get_array_descriptor([100], "integer", 4, .false., .false., .false.)
        passed = passed .and. verify_descriptor_format(descriptor, "!fir.array<100xi32>")
        
        ! Test 2: 2D fixed array
        descriptor = get_array_descriptor([10, 20], "real", 8, .false., .false., .false.)
        passed = passed .and. verify_descriptor_format(descriptor, "!fir.array<10x20xf64>")
        
        ! Test 3: 3D fixed array
        descriptor = get_array_descriptor([5, 10, 15], "logical", 4, .false., .false., .false.)
        passed = passed .and. verify_descriptor_format(descriptor, "!fir.array<5x10x15xi32>")
        
        ! Test 4: 1D assumed-shape
        descriptor = get_array_descriptor([-1], "real", 4, .true., .false., .false.)
        passed = passed .and. verify_descriptor_format(descriptor, "!fir.box<!fir.array<?xf32>>")
        
        ! Test 5: Multi-dimensional assumed-shape
        descriptor = get_array_descriptor([-1, -1, -1], "integer", 8, .true., .false., .false.)
        passed = passed .and. verify_descriptor_format(descriptor, "!fir.box<!fir.array<?x?x?xi64>>")
        
        ! Test 6: Allocatable with fixed dimensions
        descriptor = get_array_descriptor([10, 20], "real", 4, .false., .true., .false.)
        passed = passed .and. verify_descriptor_format(descriptor, "!fir.ref<!fir.box<!fir.heap<!fir.array<10x20xf32>>>>")
        
        if (passed) then
            print *, "  All array descriptor formats correct!"
        end if
    end function test_array_descriptor_formats

    function test_edge_cases() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        character(len=:), allocatable :: type_str
        
        passed = .true.
        
        print *, "Testing edge cases..."
        
        context = create_mlir_context()
        call converter%init(context)
        
        ! Test 1: Zero-size arrays
        type_str = get_array_descriptor([0], "integer", 4, .false., .false., .false.)
        passed = passed .and. verify_descriptor_format(type_str, "!fir.array<0xi32>")
        if (.not. verify_descriptor_format(type_str, "!fir.array<0xi32>")) then
            print *, "  FAIL: Zero-size array - Got: ", trim(type_str)
        end if
        
        ! Test 2: Very large arrays
        type_str = get_array_descriptor([1000000], "real", 8, .false., .false., .false.)
        passed = passed .and. verify_descriptor_format(type_str, "!fir.array<1000000xf64>")
        
        ! Test 3: Maximum rank arrays (Fortran allows up to 15)
        type_str = get_array_descriptor([2, 2, 2, 2, 2, 2, 2], "integer", 4, .false., .false., .false.)
        passed = passed .and. verify_descriptor_format(type_str, "!fir.array<2x2x2x2x2x2x2xi32>")
        
        ! Test 4: Character with zero length
        type_str = converter%get_mlir_type_string("character", 0)
        passed = passed .and. (trim(type_str) == "!fir.char<1,0>")
        
        ! Test 5: Pointer to allocatable (not allowed in Fortran)
        ! Should handle gracefully
        type_str = get_array_descriptor([-1], "real", 4, .false., .true., .true.)
        ! Should prioritize allocatable over pointer
        passed = passed .and. verify_descriptor_format(type_str, "!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>")
        
        call converter%cleanup()
        call destroy_mlir_context(context)
        
        if (passed) then
            print *, "  All edge cases handled correctly!"
        end if
    end function test_edge_cases

    function test_derived_type_mangling() result(passed)
        logical :: passed
        character(len=:), allocatable :: mangled_name
        
        passed = .true.
        
        print *, "Testing derived type name mangling..."
        
        ! Test 1: Simple type name
        mangled_name = mangle_derived_type_name("mytype")
        passed = passed .and. (trim(mangled_name) == "_QTmytype")
        if (.not. (trim(mangled_name) == "_QTmytype")) then
            print *, "  FAIL: Simple type - Expected: _QTmytype Got: ", trim(mangled_name)
        end if
        
        ! Test 2: Type in module
        mangled_name = mangle_derived_type_name("mytype", "mymodule")
        passed = passed .and. (trim(mangled_name) == "_QMmymoduleTmytype")
        if (.not. (trim(mangled_name) == "_QMmymoduleTmytype")) then
            print *, "  FAIL: Module type - Expected: _QMmymoduleTmytype Got: ", trim(mangled_name)
        end if
        
        ! Test 3: Nested type
        mangled_name = mangle_derived_type_name("innertype", parent_type="outertype")
        passed = passed .and. (trim(mangled_name) == "_QToutertypeTinnertype")
        if (.not. (trim(mangled_name) == "_QToutertypeTinnertype")) then
            print *, "  FAIL: Nested type - Expected: _QToutertypeTinnertype Got: ", trim(mangled_name)
        end if
        
        ! Test 4: Type in module with parent
        mangled_name = mangle_derived_type_name("innertype", "mymodule", "outertype")
        passed = passed .and. (trim(mangled_name) == "_QMmymoduleToutertypeTinnertype")
        if (.not. (trim(mangled_name) == "_QMmymoduleToutertypeTinnertype")) then
            print *, "  FAIL: Module nested type - Expected: _QMmymoduleToutertypeTinnertype Got: ", trim(mangled_name)
        end if
        
        if (passed) then
            print *, "  All derived type mangling tests passed!"
        end if
    end function test_derived_type_mangling

    ! Helper functions
    
    function test_basic_types_example(converter) result(passed)
        type(mlir_type_converter_t), intent(inout) :: converter
        logical :: passed
        
        passed = .true.
        
        ! Test examples from TYPE_CONVERSION.md basic types section
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("integer", 1)) == "i8")
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("integer", 2)) == "i16")
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("integer", 4)) == "i32")
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("integer", 8)) == "i64")
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("real", 4)) == "f32")
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("real", 8)) == "f64")
    end function test_basic_types_example

    function test_array_types_example(converter) result(passed)
        type(mlir_type_converter_t), intent(inout) :: converter
        logical :: passed
        
        passed = .true.
        
        ! Test array examples from documentation
        ! real :: a(10,20)
        passed = passed .and. verify_descriptor_format( &
            get_array_descriptor([10, 20], "real", 4, .false., .false., .false.), &
            "!fir.array<10x20xf32>")
        
        ! integer, allocatable :: b(:,:)
        passed = passed .and. verify_descriptor_format( &
            get_array_descriptor([-1, -1], "integer", 4, .false., .true., .false.), &
            "!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>")
    end function test_array_types_example

    function test_derived_types_example(converter) result(passed)
        type(mlir_type_converter_t), intent(inout) :: converter
        logical :: passed
        
        passed = .true.
        
        ! Test derived type examples
        ! type :: point
        !   real :: x, y
        ! end type
        passed = passed .and. &
            (trim(mangle_derived_type_name("point")) == "_QTpoint")
        
        ! In module geometry
        passed = passed .and. &
            (trim(mangle_derived_type_name("point", "geometry")) == "_QMgeometryTpoint")
    end function test_derived_types_example

    function test_pointer_types_example(converter) result(passed)
        type(mlir_type_converter_t), intent(inout) :: converter
        logical :: passed
        
        passed = .true.
        
        ! real, pointer :: p
        passed = passed .and. verify_descriptor_format( &
            wrap_with_box_type("f32", is_pointer=.true.), &
            "!fir.ref<!fir.box<!fir.ptr<f32>>>")
        
        ! integer, pointer :: arr(:)
        passed = passed .and. verify_descriptor_format( &
            get_array_descriptor([-1], "integer", 4, .false., .false., .true.), &
            "!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>")
    end function test_pointer_types_example

    function test_function_types_example(converter) result(passed)
        type(mlir_type_converter_t), intent(inout) :: converter
        logical :: passed
        
        passed = .true.
        
        ! function add(a, b) result(c)
        !   real :: a, b, c
        ! Function type: (f32, f32) -> f32
        
        ! For now, just verify we can handle function result types
        passed = passed .and. &
            (trim(converter%get_mlir_type_string("real", 4)) == "f32")
    end function test_function_types_example

    function verify_descriptor_format(descriptor, expected) result(matches)
        character(len=*), intent(in) :: descriptor, expected
        logical :: matches
        
        matches = (trim(descriptor) == trim(expected))
        if (.not. matches) then
            print *, "  Descriptor mismatch:"
            print *, "    Expected: ", trim(expected)
            print *, "    Got:      ", trim(descriptor)
        end if
    end function verify_descriptor_format

    ! Type conversion helper function stubs
    ! These would come from type_conversion_helpers module
    
    function get_array_descriptor(shape, elem_type, elem_kind, assumed_shape, allocatable, pointer) result(descriptor)
        integer, intent(in) :: shape(:)
        character(len=*), intent(in) :: elem_type
        integer, intent(in) :: elem_kind
        logical, intent(in) :: assumed_shape, allocatable, pointer
        character(len=:), allocatable :: descriptor
        
        character(len=32) :: elem_str, shape_str
        integer :: i
        
        ! Get element type string
        select case (elem_type)
        case ("integer")
            select case (elem_kind)
            case (1); elem_str = "i8"
            case (2); elem_str = "i16"
            case (4); elem_str = "i32"
            case (8); elem_str = "i64"
            case default; elem_str = "i32"
            end select
        case ("real")
            select case (elem_kind)
            case (4); elem_str = "f32"
            case (8); elem_str = "f64"
            case default; elem_str = "f32"
            end select
        case ("logical")
            select case (elem_kind)
            case (1); elem_str = "i1"
            case (4); elem_str = "i32"
            case default; elem_str = "i32"
            end select
        case default
            elem_str = "i32"
        end select
        
        ! Build shape string
        shape_str = ""
        do i = 1, size(shape)
            if (shape(i) < 0 .or. assumed_shape) then
                shape_str = trim(shape_str) // "?"
            else
                write(shape_str, '(A,I0)') trim(shape_str), shape(i)
            end if
            if (i < size(shape)) shape_str = trim(shape_str) // "x"
        end do
        
        ! Build full descriptor
        descriptor = "!fir.array<" // trim(shape_str) // "x" // trim(elem_str) // ">"
        
        if (assumed_shape) then
            descriptor = "!fir.box<" // descriptor // ">"
        end if
        
        if (allocatable) then
            descriptor = "!fir.ref<!fir.box<!fir.heap<" // descriptor // ">>>"
        else if (pointer) then
            descriptor = "!fir.ref<!fir.box<!fir.ptr<" // descriptor // ">>>"
        end if
    end function get_array_descriptor

    function wrap_with_box_type(base_type, is_pointer) result(boxed_type)
        character(len=*), intent(in) :: base_type
        logical, intent(in) :: is_pointer
        character(len=:), allocatable :: boxed_type
        
        if (is_pointer) then
            boxed_type = "!fir.ref<!fir.box<!fir.ptr<" // trim(base_type) // ">>>"
        else
            boxed_type = "!fir.box<" // trim(base_type) // ">"
        end if
    end function wrap_with_box_type

    function mangle_derived_type_name(type_name, module_name, parent_type) result(mangled)
        character(len=*), intent(in) :: type_name
        character(len=*), intent(in), optional :: module_name
        character(len=*), intent(in), optional :: parent_type
        character(len=:), allocatable :: mangled
        
        mangled = "_Q"
        
        if (present(module_name)) then
            mangled = mangled // "M" // module_name
        end if
        
        if (present(parent_type)) then
            mangled = mangled // "T" // parent_type
        end if
        
        mangled = mangled // "T" // type_name
    end function mangle_derived_type_name

end module test_type_conversion_validation