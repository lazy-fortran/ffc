program test_dialect_shared
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use dialect_base
    use fir_dialect
    use hlfir_dialect
    implicit none

    logical :: all_tests_passed

    print *, "=== Dialect Shared Functionality Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_shared_memory_operations()) all_tests_passed = .false.
    if (.not. test_shared_control_flow()) all_tests_passed = .false.
    if (.not. test_shared_type_creation()) all_tests_passed = .false.
    if (.not. test_fortran_attributes()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All dialect shared functionality tests passed!"
        stop 0
    else
        print *, "Some dialect shared functionality tests failed!"
        stop 1
    end if

contains

    function test_shared_memory_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: fir_load, hlfir_load
        type(mlir_type_t) :: i32_type
        type(mlir_value_t) :: memref
        
        passed = .true.
        
        ! Create context and register dialects
        context = create_mlir_context()
        call register_fir_dialect(context)
        call register_hlfir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        
        ! Create dummy memref
        memref = create_dummy_value(context)
        
        ! Test shared memory operation creation
        fir_load = create_memory_operation(context, "fir.load", [memref], [i32_type])
        passed = passed .and. fir_load%is_valid()
        
        hlfir_load = create_memory_operation(context, "hlfir.load", [memref], [i32_type])
        passed = passed .and. hlfir_load%is_valid()
        
        if (passed) then
            print *, "PASS: test_shared_memory_operations"
        else
            print *, "FAIL: test_shared_memory_operations"
        end if
        
        ! Cleanup
        call destroy_operation(fir_load)
        call destroy_operation(hlfir_load)
        call destroy_mlir_context(context)
    end function test_shared_memory_operations

    function test_shared_control_flow() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: if_op, loop_op
        type(mlir_value_t) :: condition, lower, upper, step
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create dummy values
        condition = create_dummy_value(context)
        lower = create_dummy_value(context)
        upper = create_dummy_value(context)
        step = create_dummy_value(context)
        
        ! Test control flow operations
        if_op = create_control_flow_operation(context, "fir.if", [condition])
        passed = passed .and. if_op%is_valid()
        
        loop_op = create_control_flow_operation(context, "fir.do_loop", [lower, upper, step])
        passed = passed .and. loop_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_shared_control_flow"
        else
            print *, "FAIL: test_shared_control_flow"
        end if
        
        ! Cleanup
        call destroy_operation(if_op)
        call destroy_operation(loop_op)
        call destroy_mlir_context(context)
    end function test_shared_control_flow

    function test_shared_type_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: i32_type, array_2d, array_dyn
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create base type
        i32_type = create_integer_type(context, 32)
        
        ! Test array type creation with rank
        array_2d = create_array_expr_type(context, i32_type, [2], .true.)
        passed = passed .and. array_2d%is_valid()
        
        ! Test array type creation with shape
        array_dyn = create_array_expr_type(context, i32_type, [10, 20], .false.)
        passed = passed .and. array_dyn%is_valid()
        
        if (passed) then
            print *, "PASS: test_shared_type_creation"
        else
            print *, "FAIL: test_shared_type_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_shared_type_creation

    function test_fortran_attributes() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_attribute_t) :: attrs1, attrs2, attrs3
        character(len=:), allocatable :: attr_str
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Test different attribute combinations
        attrs1 = create_fortran_attributes(context, [.true., .false., .false.])  ! contiguous
        ! For stub, get_string_from_attribute returns "test_string"
        passed = passed .and. attrs1%is_valid()
        
        attrs2 = create_fortran_attributes(context, [.false., .true., .true.])  ! target, optional
        passed = passed .and. attrs2%is_valid()
        
        attrs3 = create_fortran_attributes(context, [.true., .true., .false., .true.])  ! contiguous, target, allocatable
        passed = passed .and. attrs3%is_valid()
        
        if (passed) then
            print *, "PASS: test_fortran_attributes"
        else
            print *, "FAIL: test_fortran_attributes"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_fortran_attributes

end program test_dialect_shared