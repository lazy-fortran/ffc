program test_mlir_c_attributes
    use iso_c_binding, only: c_int64_t, c_double
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Attribute System Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_integer_attribute_creation()) all_tests_passed = .false.
    if (.not. test_float_attribute_creation()) all_tests_passed = .false.
    if (.not. test_string_attribute_creation()) all_tests_passed = .false.
    if (.not. test_array_attribute_creation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API attribute system tests passed!"
        stop 0
    else
        print *, "Some MLIR C API attribute system tests failed!"
        stop 1
    end if

contains

    function test_integer_attribute_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: i32_type, i64_type
        type(mlir_attribute_t) :: attr32, attr64
        integer(c_int64_t) :: value, retrieved
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        i64_type = create_integer_type(context, 64)
        
        ! Test i32 attribute
        value = 42_c_int64_t
        attr32 = create_integer_attribute(context, i32_type, value)
        passed = passed .and. attr32%is_valid() .and. is_integer_attribute(attr32)
        
        ! Test value retrieval
        retrieved = get_integer_from_attribute(attr32)
        passed = passed .and. (retrieved == 42_c_int64_t)
        
        ! Test i64 attribute
        value = 1234567890_c_int64_t
        attr64 = create_integer_attribute(context, i64_type, value)
        passed = passed .and. attr64%is_valid() .and. is_integer_attribute(attr64)
        
        if (passed) then
            print *, "PASS: test_integer_attribute_creation"
        else
            print *, "FAIL: test_integer_attribute_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_integer_attribute_creation

    function test_float_attribute_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: f32_type, f64_type
        type(mlir_attribute_t) :: attr32, attr64
        real(c_double) :: value, retrieved
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create types
        f32_type = create_float_type(context, 32)
        f64_type = create_float_type(context, 64)
        
        ! Test f32 attribute
        value = 3.14_c_double
        attr32 = create_float_attribute(context, f32_type, value)
        passed = passed .and. attr32%is_valid() .and. is_float_attribute(attr32)
        
        ! Test value retrieval
        retrieved = get_float_from_attribute(attr32)
        passed = passed .and. (abs(retrieved - 3.14_c_double) < 0.001_c_double)
        
        ! Test f64 attribute
        value = 2.71828_c_double
        attr64 = create_float_attribute(context, f64_type, value)
        passed = passed .and. attr64%is_valid() .and. is_float_attribute(attr64)
        
        if (passed) then
            print *, "PASS: test_float_attribute_creation"
        else
            print *, "FAIL: test_float_attribute_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_float_attribute_creation

    function test_string_attribute_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_attribute_t) :: attr
        character(len=:), allocatable :: retrieved
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Test string attribute
        attr = create_string_attribute(context, "test_string")
        passed = passed .and. attr%is_valid() .and. is_string_attribute(attr)
        
        ! Test value retrieval
        retrieved = get_string_from_attribute(attr)
        passed = passed .and. (retrieved == "test_string")
        
        ! Test empty string
        attr = create_string_attribute(context, "")
        passed = passed .and. attr%is_valid() .and. is_string_attribute(attr)
        
        if (passed) then
            print *, "PASS: test_string_attribute_creation"
        else
            print *, "FAIL: test_string_attribute_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_string_attribute_creation

    function test_array_attribute_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: i32_type
        type(mlir_attribute_t) :: array_attr
        type(mlir_attribute_t), dimension(3) :: elements
        integer :: i, array_size
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create type
        i32_type = create_integer_type(context, 32)
        
        ! Create element attributes
        do i = 1, 3
            elements(i) = create_integer_attribute(context, i32_type, int(i*10, c_int64_t))
        end do
        
        ! Create array attribute
        array_attr = create_array_attribute(context, elements)
        passed = passed .and. array_attr%is_valid() .and. is_array_attribute(array_attr)
        
        ! Test array size
        array_size = get_array_size(array_attr)
        passed = passed .and. (array_size == 3)
        
        if (passed) then
            print *, "PASS: test_array_attribute_creation"
        else
            print *, "FAIL: test_array_attribute_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_array_attribute_creation

end program test_mlir_c_attributes