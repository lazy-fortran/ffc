program test_mlir_c_types
    use iso_c_binding, only: c_int64_t
    use mlir_c_core
    use mlir_c_types
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Type System Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_integer_type_creation()) all_tests_passed = .false.
    if (.not. test_float_type_creation()) all_tests_passed = .false.
    if (.not. test_array_type_creation()) all_tests_passed = .false.
    if (.not. test_reference_type_creation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API type system tests passed!"
        stop 0
    else
        print *, "Some MLIR C API type system tests failed!"
        stop 1
    end if

contains

    function test_integer_type_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: i1_type, i8_type, i32_type, i64_type
        type(mlir_type_t) :: signed_type, unsigned_type
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Test various integer widths
        i1_type = create_integer_type(context, 1)
        passed = passed .and. i1_type%is_valid() .and. is_integer_type(i1_type)
        
        i8_type = create_integer_type(context, 8)
        passed = passed .and. i8_type%is_valid() .and. is_integer_type(i8_type)
        
        i32_type = create_integer_type(context, 32)
        passed = passed .and. i32_type%is_valid() .and. is_integer_type(i32_type)
        
        i64_type = create_integer_type(context, 64)
        passed = passed .and. i64_type%is_valid() .and. is_integer_type(i64_type)
        
        ! Test signed/unsigned
        signed_type = create_integer_type(context, 32, signed=.true.)
        passed = passed .and. signed_type%is_valid()
        
        unsigned_type = create_integer_type(context, 32, signed=.false.)
        passed = passed .and. unsigned_type%is_valid()
        
        ! Test width query
        passed = passed .and. (get_integer_width(i32_type) == 32)
        
        if (passed) then
            print *, "PASS: test_integer_type_creation"
        else
            print *, "FAIL: test_integer_type_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_integer_type_creation

    function test_float_type_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: f32_type, f64_type
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Test float types
        f32_type = create_float_type(context, 32)
        passed = passed .and. f32_type%is_valid() .and. is_float_type(f32_type)
        
        f64_type = create_float_type(context, 64)
        passed = passed .and. f64_type%is_valid() .and. is_float_type(f64_type)
        
        ! Test type kind
        passed = passed .and. (get_type_kind(f32_type) == TYPE_FLOAT)
        passed = passed .and. (get_type_kind(f64_type) == TYPE_FLOAT)
        
        if (passed) then
            print *, "PASS: test_float_type_creation"
        else
            print *, "FAIL: test_float_type_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_float_type_creation

    function test_array_type_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: element_type, array_type
        integer(c_int64_t), dimension(2) :: shape
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create element type
        element_type = create_integer_type(context, 32)
        
        ! Test 1D array
        shape = [10_c_int64_t, 0_c_int64_t]
        array_type = create_array_type(context, element_type, shape(1:1))
        passed = passed .and. array_type%is_valid() .and. is_array_type(array_type)
        
        ! Test 2D array
        shape = [10_c_int64_t, 20_c_int64_t]
        array_type = create_array_type(context, element_type, shape)
        passed = passed .and. array_type%is_valid() .and. is_array_type(array_type)
        
        ! Test type kind
        passed = passed .and. (get_type_kind(array_type) == TYPE_ARRAY)
        
        if (passed) then
            print *, "PASS: test_array_type_creation"
        else
            print *, "FAIL: test_array_type_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_array_type_creation

    function test_reference_type_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: base_type, ref_type
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create base types
        base_type = create_integer_type(context, 32)
        
        ! Test reference type creation
        ref_type = create_reference_type(context, base_type)
        passed = passed .and. ref_type%is_valid()
        
        ! Test with float base type
        base_type = create_float_type(context, 64)
        ref_type = create_reference_type(context, base_type)
        passed = passed .and. ref_type%is_valid()
        
        if (passed) then
            print *, "PASS: test_reference_type_creation"
        else
            print *, "FAIL: test_reference_type_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_reference_type_creation

end program test_mlir_c_types