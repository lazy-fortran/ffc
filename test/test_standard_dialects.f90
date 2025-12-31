program test_standard_dialects
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use standard_dialects
    implicit none

    logical :: all_tests_passed

    print *, "=== Standard Dialects Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_func_dialect_operations()) all_tests_passed = .false.
    if (.not. test_arith_dialect_operations()) all_tests_passed = .false.
    if (.not. test_scf_dialect_operations()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All standard dialects tests passed!"
        stop 0
    else
        print *, "Some standard dialects tests failed!"
        stop 1
    end if

contains

    function test_func_dialect_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: func_op, return_op, call_op
        type(mlir_type_t) :: i32_type, func_type
        type(mlir_value_t) :: arg_val, result_val
        type(mlir_region_t) :: body_region
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_func_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        func_type = create_function_type(context, [i32_type], [i32_type])
        
        ! Create dummy values
        arg_val = create_dummy_value(context)
        result_val = create_dummy_value(context)
        
        ! Create empty region for function body
        body_region = create_empty_region(context)
        
        ! Test func.func operation
        func_op = create_func_func(context, "test_func", func_type, body_region)
        passed = passed .and. func_op%is_valid()
        passed = passed .and. verify_operation(func_op)
        
        ! Test func.return operation
        return_op = create_func_return(context, [result_val])
        passed = passed .and. return_op%is_valid()
        
        ! Test func.call operation
        call_op = create_func_call(context, "test_func", [arg_val], [i32_type])
        passed = passed .and. call_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_func_dialect_operations"
        else
            print *, "FAIL: test_func_dialect_operations"
        end if
        
        ! Cleanup
        call destroy_operation(func_op)
        call destroy_operation(return_op)
        call destroy_operation(call_op)
        call destroy_mlir_context(context)
    end function test_func_dialect_operations

    function test_arith_dialect_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: addi_op, muli_op, cmpf_op, const_op
        type(mlir_type_t) :: i32_type, f64_type, i1_type
        type(mlir_value_t) :: lhs, rhs, f_lhs, f_rhs
        type(mlir_attribute_t) :: const_val
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_arith_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        f64_type = create_float_type(context, 64)
        i1_type = create_integer_type(context, 1)
        
        ! Create dummy values
        lhs = create_dummy_value(context)
        rhs = create_dummy_value(context)
        f_lhs = create_dummy_value(context)
        f_rhs = create_dummy_value(context)
        
        ! Test arith.addi operation
        addi_op = create_arith_addi(context, lhs, rhs, i32_type)
        passed = passed .and. addi_op%is_valid()
        passed = passed .and. verify_operation(addi_op)
        
        ! Test arith.muli operation
        muli_op = create_arith_muli(context, lhs, rhs, i32_type)
        passed = passed .and. muli_op%is_valid()
        
        ! Test arith.cmpf operation
        cmpf_op = create_arith_cmpf(context, "oeq", f_lhs, f_rhs, i1_type)
        passed = passed .and. cmpf_op%is_valid()
        
        ! Test arith.constant operation
        const_val = create_integer_attribute(context, i32_type, 42_c_int64_t)
        const_op = create_arith_constant(context, const_val, i32_type)
        passed = passed .and. const_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_arith_dialect_operations"
        else
            print *, "FAIL: test_arith_dialect_operations"
        end if
        
        ! Cleanup
        call destroy_operation(addi_op)
        call destroy_operation(muli_op)
        call destroy_operation(cmpf_op)
        call destroy_operation(const_op)
        call destroy_mlir_context(context)
    end function test_arith_dialect_operations

    function test_scf_dialect_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: if_op, for_op, while_op, yield_op
        type(mlir_type_t) :: i32_type, i1_type
        type(mlir_value_t) :: condition, lower, upper, step, init_val
        type(mlir_region_t) :: then_region, else_region, body_region, before_region, after_region
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_scf_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        i1_type = create_integer_type(context, 1)
        
        ! Create dummy values
        condition = create_dummy_value(context)
        lower = create_dummy_value(context)
        upper = create_dummy_value(context)
        step = create_dummy_value(context)
        init_val = create_dummy_value(context)
        
        ! Create empty regions
        then_region = create_empty_region(context)
        else_region = create_empty_region(context)
        body_region = create_empty_region(context)
        before_region = create_empty_region(context)
        after_region = create_empty_region(context)
        
        ! Test scf.if operation
        if_op = create_scf_if(context, condition, [i32_type], then_region, else_region)
        passed = passed .and. if_op%is_valid()
        passed = passed .and. verify_operation(if_op)
        
        ! Test scf.for operation
        for_op = create_scf_for(context, lower, upper, step, [init_val], body_region)
        passed = passed .and. for_op%is_valid()
        
        ! Test scf.while operation
        while_op = create_scf_while(context, [init_val], before_region, after_region)
        passed = passed .and. while_op%is_valid()
        
        ! Test scf.yield operation
        yield_op = create_scf_yield(context, [init_val])
        passed = passed .and. yield_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_scf_dialect_operations"
        else
            print *, "FAIL: test_scf_dialect_operations"
        end if
        
        ! Cleanup
        call destroy_operation(if_op)
        call destroy_operation(for_op)
        call destroy_operation(while_op)
        call destroy_operation(yield_op)
        call destroy_mlir_context(context)
    end function test_scf_dialect_operations

end program test_standard_dialects