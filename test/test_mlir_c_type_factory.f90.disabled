program test_mlir_c_type_factory
    use iso_c_binding, only: c_int64_t
    use mlir_c_core
    use mlir_c_types
    use mlir_c_type_factory
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Type Factory Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_type_caching()) all_tests_passed = .false.
    if (.not. test_validation()) all_tests_passed = .false.
    if (.not. test_cache_growth()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API type factory tests passed!"
        stop 0
    else
        print *, "Some MLIR C API type factory tests failed!"
        stop 1
    end if

contains

    function test_type_caching() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(type_factory_t) :: factory
        type(mlir_type_t) :: type1, type2
        
        passed = .true.
        
        ! Create context and factory
        context = create_mlir_context()
        call factory%init(context)
        
        ! Get same type twice - should return cached version
        type1 = factory%get_integer_type(32)
        type2 = factory%get_integer_type(32)
        
        ! Both should be valid and equal
        passed = passed .and. type1%is_valid() .and. type2%is_valid()
        
        ! Test different widths
        type1 = factory%get_integer_type(64)
        type2 = factory%get_integer_type(64)
        passed = passed .and. type1%is_valid() .and. type2%is_valid()
        
        ! Test float caching
        type1 = factory%get_float_type(32)
        type2 = factory%get_float_type(32)
        passed = passed .and. type1%is_valid() .and. type2%is_valid()
        
        if (passed) then
            print *, "PASS: test_type_caching"
        else
            print *, "FAIL: test_type_caching"
        end if
        
        ! Cleanup
        call factory%finalize()
        call destroy_mlir_context(context)
    end function test_type_caching

    function test_validation() result(passed)
        logical :: passed
        
        passed = .true.
        
        ! Test integer width validation
        passed = passed .and. validate_integer_width(1)    ! i1
        passed = passed .and. validate_integer_width(8)    ! i8
        passed = passed .and. validate_integer_width(16)   ! i16
        passed = passed .and. validate_integer_width(32)   ! i32
        passed = passed .and. validate_integer_width(64)   ! i64
        passed = passed .and. validate_integer_width(128)  ! i128
        
        ! Invalid widths
        passed = passed .and. .not. validate_integer_width(7)
        passed = passed .and. .not. validate_integer_width(33)
        
        ! Test float width validation
        passed = passed .and. validate_float_width(32)     ! f32
        passed = passed .and. validate_float_width(64)     ! f64
        
        ! Invalid float widths
        passed = passed .and. .not. validate_float_width(16)
        passed = passed .and. .not. validate_float_width(128)
        
        if (passed) then
            print *, "PASS: test_validation"
        else
            print *, "FAIL: test_validation"
        end if
    end function test_validation

    function test_cache_growth() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(type_factory_t) :: factory
        type(mlir_type_t) :: type
        integer :: i
        
        passed = .true.
        
        ! Create context and factory
        context = create_mlir_context()
        call factory%init(context)
        
        ! Add many types to trigger cache growth
        do i = 1, 20
            type = factory%get_integer_type(i, signed=.true.)
            passed = passed .and. type%is_valid()
            
            type = factory%get_integer_type(i, signed=.false.)
            passed = passed .and. type%is_valid()
        end do
        
        if (passed) then
            print *, "PASS: test_cache_growth"
        else
            print *, "FAIL: test_cache_growth"
        end if
        
        ! Cleanup
        call factory%finalize()
        call destroy_mlir_context(context)
    end function test_cache_growth

end program test_mlir_c_type_factory