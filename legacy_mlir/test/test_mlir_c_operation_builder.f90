program test_mlir_c_operation_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Operation Builder Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_fluent_builder()) all_tests_passed = .false.
    if (.not. test_operation_templates()) all_tests_passed = .false.
    if (.not. test_chained_building()) all_tests_passed = .false.
    if (.not. test_error_handling()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API operation builder tests passed!"
        stop 0
    else
        print *, "Some MLIR C API operation builder tests failed!"
        stop 1
    end if

contains

    function test_fluent_builder() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(operation_builder_t) :: builder
        type(mlir_operation_t) :: op
        type(mlir_type_t) :: i32_type, f64_type
        type(mlir_value_t) :: val1, val2
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        f64_type = create_float_type(context, 64)
        
        ! Create dummy values
        val1 = create_dummy_value(context)
        val2 = create_dummy_value(context)
        
        ! Test fluent builder
        call builder%init(context, "test.add")
        call builder%operand(val1)
        call builder%operand(val2)
        call builder%result(i32_type)
        call builder%attr("fastmath", create_string_attribute(context, "fast"))
        
        op = builder%build()
        passed = passed .and. op%is_valid()
        passed = passed .and. verify_operation(op)
        
        if (passed) then
            print *, "PASS: test_fluent_builder"
        else
            print *, "FAIL: test_fluent_builder"
        end if
        
        ! Cleanup
        call destroy_operation(op)
        call destroy_mlir_context(context)
    end function test_fluent_builder

    function test_operation_templates() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: add_op, mul_op, const_op
        type(mlir_type_t) :: i32_type
        type(mlir_value_t) :: lhs, rhs
        type(mlir_attribute_t) :: const_val
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create type
        i32_type = create_integer_type(context, 32)
        
        ! Create dummy values
        lhs = create_dummy_value(context)
        rhs = create_dummy_value(context)
        
        ! Test binary operation templates
        add_op = build_binary_op(context, "arith.addi", lhs, rhs, i32_type)
        passed = passed .and. add_op%is_valid()
        
        mul_op = build_binary_op(context, "arith.muli", lhs, rhs, i32_type)
        passed = passed .and. mul_op%is_valid()
        
        ! Test constant operation template
        const_val = create_integer_attribute(context, i32_type, 42_c_int64_t)
        const_op = build_constant_op(context, i32_type, const_val)
        passed = passed .and. const_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_operation_templates"
        else
            print *, "FAIL: test_operation_templates"
        end if
        
        ! Cleanup
        call destroy_operation(add_op)
        call destroy_operation(mul_op)
        call destroy_operation(const_op)
        call destroy_mlir_context(context)
    end function test_operation_templates

    function test_chained_building() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(operation_builder_t) :: builder
        type(mlir_operation_t) :: op
        type(mlir_type_t) :: i32_type, i64_type
        type(mlir_value_t), dimension(3) :: values
        type(mlir_attribute_t) :: attr1, attr2
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        i64_type = create_integer_type(context, 64)
        
        ! Create dummy values
        values(1) = create_dummy_value(context)
        values(2) = create_dummy_value(context)
        values(3) = create_dummy_value(context)
        
        ! Create attributes
        attr1 = create_integer_attribute(context, i32_type, 10_c_int64_t)
        attr2 = create_string_attribute(context, "test")
        
        ! Test complex chained building
        call builder%init(context, "test.complex")
        call builder%operands(values)
        call builder%result(i32_type)
        call builder%result(i64_type)
        call builder%attr("param1", attr1)
        call builder%attr("param2", attr2)
        call builder%location(create_unknown_location(context))
        
        op = builder%build()
        passed = passed .and. op%is_valid()
        passed = passed .and. (get_operation_num_results(op) == 2)
        
        if (passed) then
            print *, "PASS: test_chained_building"
        else
            print *, "FAIL: test_chained_building"
        end if
        
        ! Cleanup
        call destroy_operation(op)
        call destroy_mlir_context(context)
    end function test_chained_building

    function test_error_handling() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(operation_builder_t) :: builder
        type(mlir_operation_t) :: op
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Test building without required elements
        call builder%init(context, "test.incomplete")
        ! No operands or results added
        
        op = builder%build()
        ! Should still create a valid operation (some ops have no operands/results)
        passed = passed .and. op%is_valid()
        
        ! Test invalid operation name
        call builder%init(context, "")
        op = builder%build()
        passed = passed .and. .not. op%is_valid()
        
        if (passed) then
            print *, "PASS: test_error_handling"
        else
            print *, "FAIL: test_error_handling"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_error_handling

end program test_mlir_c_operation_builder