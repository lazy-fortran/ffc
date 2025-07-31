program test_fir_dialect
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use fir_dialect
    implicit none

    logical :: all_tests_passed

    print *, "=== FIR Dialect Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_fir_dialect_registration()) all_tests_passed = .false.
    if (.not. test_fir_declare_operation()) all_tests_passed = .false.
    if (.not. test_fir_load_operation()) all_tests_passed = .false.
    if (.not. test_fir_store_operation()) all_tests_passed = .false.
    if (.not. test_fir_alloca_operation()) all_tests_passed = .false.
    if (.not. test_fir_do_loop_operation()) all_tests_passed = .false.
    if (.not. test_fir_if_operation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All FIR dialect tests passed!"
        stop 0
    else
        print *, "Some FIR dialect tests failed!"
        stop 1
    end if

contains

    function test_fir_dialect_registration() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Register FIR dialect
        call register_fir_dialect(context)
        
        ! Check if dialect is registered
        passed = passed .and. is_fir_dialect_registered(context)
        
        if (passed) then
            print *, "PASS: test_fir_dialect_registration"
        else
            print *, "FAIL: test_fir_dialect_registration"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_fir_dialect_registration

    function test_fir_declare_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: declare_op
        type(mlir_type_t) :: i32_type, ref_type
        type(mlir_value_t) :: memref
        type(mlir_attribute_t) :: name_attr
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! Create dummy memref value
        memref = create_dummy_value(context)
        
        ! Create name attribute
        name_attr = create_string_attribute(context, "x")
        
        ! Create fir.declare operation
        declare_op = create_fir_declare(context, memref, name_attr, ref_type)
        passed = passed .and. declare_op%is_valid()
        passed = passed .and. verify_operation(declare_op)
        
        if (passed) then
            print *, "PASS: test_fir_declare_operation"
        else
            print *, "FAIL: test_fir_declare_operation"
        end if
        
        ! Cleanup
        call destroy_operation(declare_op)
        call destroy_mlir_context(context)
    end function test_fir_declare_operation

    function test_fir_load_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: load_op
        type(mlir_type_t) :: i32_type, ref_type
        type(mlir_value_t) :: memref
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! Create dummy memref value
        memref = create_dummy_value(context)
        
        ! Create fir.load operation
        load_op = create_fir_load(context, memref, i32_type)
        passed = passed .and. load_op%is_valid()
        passed = passed .and. verify_operation(load_op)
        
        if (passed) then
            print *, "PASS: test_fir_load_operation"
        else
            print *, "FAIL: test_fir_load_operation"
        end if
        
        ! Cleanup
        call destroy_operation(load_op)
        call destroy_mlir_context(context)
    end function test_fir_load_operation

    function test_fir_store_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: store_op
        type(mlir_type_t) :: i32_type, ref_type
        type(mlir_value_t) :: value, memref
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! Create dummy values
        value = create_dummy_value(context)
        memref = create_dummy_value(context)
        
        ! Create fir.store operation
        store_op = create_fir_store(context, value, memref)
        passed = passed .and. store_op%is_valid()
        passed = passed .and. verify_operation(store_op)
        
        if (passed) then
            print *, "PASS: test_fir_store_operation"
        else
            print *, "FAIL: test_fir_store_operation"
        end if
        
        ! Cleanup
        call destroy_operation(store_op)
        call destroy_mlir_context(context)
    end function test_fir_store_operation

    function test_fir_alloca_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: alloca_op
        type(mlir_type_t) :: i32_type, ref_type
        type(mlir_attribute_t) :: size_attr
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! Create size attribute (optional)
        size_attr = create_integer_attribute(context, i32_type, 1_c_int64_t)
        
        ! Create fir.alloca operation
        alloca_op = create_fir_alloca(context, i32_type, ref_type)
        passed = passed .and. alloca_op%is_valid()
        passed = passed .and. verify_operation(alloca_op)
        
        if (passed) then
            print *, "PASS: test_fir_alloca_operation"
        else
            print *, "FAIL: test_fir_alloca_operation"
        end if
        
        ! Cleanup
        call destroy_operation(alloca_op)
        call destroy_mlir_context(context)
    end function test_fir_alloca_operation

    function test_fir_do_loop_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: do_loop_op
        type(mlir_type_t) :: i32_type
        type(mlir_value_t) :: lower, upper, step
        type(mlir_region_t) :: body_region
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        
        ! Create dummy values for bounds
        lower = create_dummy_value(context)
        upper = create_dummy_value(context)
        step = create_dummy_value(context)
        
        ! Create empty region for loop body
        body_region = create_empty_region(context)
        
        ! Create fir.do_loop operation
        do_loop_op = create_fir_do_loop(context, lower, upper, step, body_region)
        passed = passed .and. do_loop_op%is_valid()
        passed = passed .and. verify_operation(do_loop_op)
        
        if (passed) then
            print *, "PASS: test_fir_do_loop_operation"
        else
            print *, "FAIL: test_fir_do_loop_operation"
        end if
        
        ! Cleanup
        call destroy_operation(do_loop_op)
        call destroy_mlir_context(context)
    end function test_fir_do_loop_operation

    function test_fir_if_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: if_op
        type(mlir_type_t) :: i1_type
        type(mlir_value_t) :: condition
        type(mlir_region_t) :: then_region, else_region
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create types
        i1_type = create_integer_type(context, 1)
        
        ! Create dummy condition value
        condition = create_dummy_value(context)
        
        ! Create empty regions for then/else branches
        then_region = create_empty_region(context)
        else_region = create_empty_region(context)
        
        ! Create fir.if operation
        if_op = create_fir_if(context, condition, then_region, else_region)
        passed = passed .and. if_op%is_valid()
        passed = passed .and. verify_operation(if_op)
        
        if (passed) then
            print *, "PASS: test_fir_if_operation"
        else
            print *, "FAIL: test_fir_if_operation"
        end if
        
        ! Cleanup
        call destroy_operation(if_op)
        call destroy_mlir_context(context)
    end function test_fir_if_operation

end program test_fir_dialect