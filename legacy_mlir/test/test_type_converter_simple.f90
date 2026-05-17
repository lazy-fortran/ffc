program test_type_converter_simple
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use fortfc_type_converter
    implicit none

    logical :: all_tests_passed

    print *, "=== Simple Type Converter Tests ==="
    print *

    all_tests_passed = .true.

    ! Run basic tests - these will fail initially (RED phase)
    if (.not. test_basic_integer_types()) all_tests_passed = .false.
    if (.not. test_basic_float_types()) all_tests_passed = .false.
    if (.not. test_type_string_generation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All simple type converter tests passed!"
        stop 0
    else
        print *, "Some simple type converter tests failed!"
        stop 1
    end if

contains

    function test_basic_integer_types() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(mlir_type_t) :: i8_type, i16_type, i32_type, i64_type
        
        passed = .true.
        
        ! Create context and converter
        context = create_mlir_context()
        converter = create_type_converter(context)
        passed = passed .and. converter%is_valid()
        
        ! Test basic integer type creation
        i8_type = converter%create_integer_type(8)
        passed = passed .and. i8_type%is_valid()
        
        i16_type = converter%create_integer_type(16)
        passed = passed .and. i16_type%is_valid()
        
        i32_type = converter%create_integer_type(32)
        passed = passed .and. i32_type%is_valid()
        
        i64_type = converter%create_integer_type(64)
        passed = passed .and. i64_type%is_valid()
        
        if (passed) then
            print *, "PASS: test_basic_integer_types"
        else
            print *, "FAIL: test_basic_integer_types"
        end if
        
        ! Cleanup
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_basic_integer_types

    function test_basic_float_types() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(mlir_type_t) :: f32_type, f64_type
        
        passed = .true.
        
        ! Create context and converter
        context = create_mlir_context()
        converter = create_type_converter(context)
        
        ! Test basic float type creation
        f32_type = converter%create_float_type(32)
        passed = passed .and. f32_type%is_valid()
        
        f64_type = converter%create_float_type(64)
        passed = passed .and. f64_type%is_valid()
        
        if (passed) then
            print *, "PASS: test_basic_float_types"
        else
            print *, "FAIL: test_basic_float_types"
        end if
        
        ! Cleanup
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_basic_float_types

    function test_type_string_generation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        character(len=:), allocatable :: type_str
        
        passed = .true.
        
        ! Create context and converter
        context = create_mlir_context()
        converter = create_type_converter(context)
        
        ! Test integer type string generation
        type_str = converter%get_integer_type_string(8)
        passed = passed .and. (type_str == "i8")
        
        type_str = converter%get_integer_type_string(32)
        passed = passed .and. (type_str == "i32")
        
        ! Test float type string generation
        type_str = converter%get_float_type_string(32)
        passed = passed .and. (type_str == "f32")
        
        type_str = converter%get_float_type_string(64)
        passed = passed .and. (type_str == "f64")
        
        ! Test logical type
        type_str = converter%get_logical_type_string()
        passed = passed .and. (type_str == "i1")
        
        if (passed) then
            print *, "PASS: test_type_string_generation"
        else
            print *, "FAIL: test_type_string_generation"
        end if
        
        ! Cleanup
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_type_string_generation

end program test_type_converter_simple