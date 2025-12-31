program test_hlfir_expression_generation
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    use expression_gen
    implicit none

    logical :: all_tests_passed

    print *, "=== HLFIR Expression Generation Tests (C API) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_literal_expressions()) all_tests_passed = .false.
    if (.not. test_variable_references()) all_tests_passed = .false.
    if (.not. test_binary_operations()) all_tests_passed = .false.
    if (.not. test_unary_operations()) all_tests_passed = .false.
    if (.not. test_function_calls()) all_tests_passed = .false.
    if (.not. test_array_subscripts()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All HLFIR expression generation tests passed!"
        stop 0
    else
        print *, "Some HLFIR expression generation tests failed!"
        stop 1
    end if

contains

    function test_literal_expressions() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: int_literal, real_literal, bool_literal, char_literal
        type(mlir_type_t) :: int_type, real_type, bool_type, char_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create types
        int_type = create_integer_type(context, 32)
        real_type = create_float_type(context, 64)
        bool_type = create_integer_type(context, 1)
        char_type = create_integer_type(context, 8)  ! Simplified char as i8
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. real_type%is_valid()
        passed = passed .and. bool_type%is_valid()
        passed = passed .and. char_type%is_valid()
        
        ! Test: Generate literal expressions
        int_literal = generate_integer_literal(builder, int_type, 42)
        real_literal = generate_real_literal(builder, real_type, 3.14d0)
        bool_literal = generate_boolean_literal(builder, bool_type, .true.)
        char_literal = generate_character_literal(builder, char_type, "A")
        
        passed = passed .and. int_literal%is_valid()
        passed = passed .and. real_literal%is_valid()
        passed = passed .and. bool_literal%is_valid()
        passed = passed .and. char_literal%is_valid()
        
        ! Verify literal values
        passed = passed .and. is_literal_operation(int_literal)
        passed = passed .and. literal_has_value(int_literal, 42)
        passed = passed .and. literal_has_type(int_literal, int_type)
        
        if (passed) then
            print *, "PASS: test_literal_expressions"
        else
            print *, "FAIL: test_literal_expressions"
        end if
        
        ! Cleanup
        call destroy_operation(char_literal)
        call destroy_operation(bool_literal)
        call destroy_operation(real_literal)
        call destroy_operation(int_literal)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_literal_expressions

    function test_variable_references() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: var_ref, global_ref
        type(mlir_value_t) :: var_value
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create variable type
        int_type = create_integer_type(context, 32)
        var_value = create_dummy_value(context)
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. var_value%is_valid()
        
        ! Test: Generate variable reference expressions
        var_ref = generate_variable_reference(builder, "local_var", int_type)
        global_ref = generate_global_reference(builder, "global_var", int_type)
        
        passed = passed .and. var_ref%is_valid()
        passed = passed .and. global_ref%is_valid()
        
        ! Verify variable references
        passed = passed .and. is_variable_reference(var_ref)
        passed = passed .and. reference_has_name(var_ref, "local_var")
        passed = passed .and. reference_has_type(var_ref, int_type)
        
        if (passed) then
            print *, "PASS: test_variable_references"
        else
            print *, "FAIL: test_variable_references"
        end if
        
        ! Cleanup
        call destroy_operation(global_ref)
        call destroy_operation(var_ref)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_variable_references

    function test_binary_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: add_op, sub_op, mul_op, div_op, cmp_op
        type(mlir_value_t) :: lhs, rhs
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create operands
        int_type = create_integer_type(context, 32)
        lhs = create_dummy_value(context)
        rhs = create_dummy_value(context)
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. lhs%is_valid()
        passed = passed .and. rhs%is_valid()
        
        ! Test: Generate binary operations
        add_op = generate_binary_operation(builder, "add", lhs, rhs)
        sub_op = generate_binary_operation(builder, "sub", lhs, rhs)
        mul_op = generate_binary_operation(builder, "mul", lhs, rhs)
        div_op = generate_binary_operation(builder, "div", lhs, rhs)
        cmp_op = generate_comparison_operation(builder, "eq", lhs, rhs)
        
        passed = passed .and. add_op%is_valid()
        passed = passed .and. sub_op%is_valid()
        passed = passed .and. mul_op%is_valid()
        passed = passed .and. div_op%is_valid()
        passed = passed .and. cmp_op%is_valid()
        
        ! Verify binary operations
        passed = passed .and. is_binary_operation(add_op)
        passed = passed .and. binary_has_operands(add_op, lhs, rhs)
        passed = passed .and. binary_has_operator(add_op, "add")
        
        if (passed) then
            print *, "PASS: test_binary_operations"
        else
            print *, "FAIL: test_binary_operations"
        end if
        
        ! Cleanup
        call destroy_operation(cmp_op)
        call destroy_operation(div_op)
        call destroy_operation(mul_op)
        call destroy_operation(sub_op)
        call destroy_operation(add_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_binary_operations

    function test_unary_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: neg_op, not_op, abs_op
        type(mlir_value_t) :: operand
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create operand
        int_type = create_integer_type(context, 32)
        operand = create_dummy_value(context)
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. operand%is_valid()
        
        ! Test: Generate unary operations
        neg_op = generate_unary_operation(builder, "neg", operand)
        not_op = generate_unary_operation(builder, "not", operand)
        abs_op = generate_intrinsic_operation(builder, "abs", operand)
        
        passed = passed .and. neg_op%is_valid()
        passed = passed .and. not_op%is_valid()
        passed = passed .and. abs_op%is_valid()
        
        ! Verify unary operations
        passed = passed .and. is_unary_operation(neg_op)
        passed = passed .and. unary_has_operand(neg_op, operand)
        passed = passed .and. unary_has_operator(neg_op, "neg")
        
        if (passed) then
            print *, "PASS: test_unary_operations"
        else
            print *, "FAIL: test_unary_operations"
        end if
        
        ! Cleanup
        call destroy_operation(abs_op)
        call destroy_operation(not_op)
        call destroy_operation(neg_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_unary_operations

    function test_function_calls() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: call_op
        type(mlir_value_t), dimension(2) :: args
        type(mlir_type_t) :: int_type, return_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create function arguments
        int_type = create_integer_type(context, 32)
        return_type = int_type
        args(1) = create_dummy_value(context)
        args(2) = create_dummy_value(context)
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. return_type%is_valid()
        passed = passed .and. args(1)%is_valid()
        passed = passed .and. args(2)%is_valid()
        
        ! Test: Generate function call expression
        call_op = generate_function_call(builder, "add_func", args, return_type)
        passed = passed .and. call_op%is_valid()
        
        ! Verify function call
        passed = passed .and. is_function_call(call_op)
        passed = passed .and. call_has_name(call_op, "add_func")
        passed = passed .and. call_has_arguments(call_op, args)
        passed = passed .and. call_has_return_type(call_op, return_type)
        
        if (passed) then
            print *, "PASS: test_function_calls"
        else
            print *, "FAIL: test_function_calls"
        end if
        
        ! Cleanup
        call destroy_operation(call_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_function_calls

    function test_array_subscripts() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: subscript_op, slice_op
        type(mlir_value_t) :: array_base, index1, index2
        type(mlir_value_t), dimension(2) :: indices
        type(mlir_type_t) :: int_type, array_type, element_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create array and index types
        int_type = create_integer_type(context, 32)
        element_type = int_type
        array_type = create_array_type(context, element_type, [10_c_int64_t, 20_c_int64_t])
        array_base = create_dummy_value(context)
        index1 = create_dummy_value(context)
        index2 = create_dummy_value(context)
        indices = [index1, index2]
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. array_type%is_valid()
        passed = passed .and. array_base%is_valid()
        passed = passed .and. index1%is_valid()
        passed = passed .and. index2%is_valid()
        
        ! Test: Generate array subscript expressions
        subscript_op = generate_array_subscript(builder, array_base, indices)
        slice_op = generate_array_slice(builder, array_base, index1, index2)
        
        passed = passed .and. subscript_op%is_valid()
        passed = passed .and. slice_op%is_valid()
        
        ! Verify array subscripts
        passed = passed .and. is_array_subscript(subscript_op)
        passed = passed .and. subscript_has_base(subscript_op, array_base)
        passed = passed .and. subscript_has_indices(subscript_op, indices)
        
        if (passed) then
            print *, "PASS: test_array_subscripts"
        else
            print *, "FAIL: test_array_subscripts"
        end if
        
        ! Cleanup
        call destroy_operation(slice_op)
        call destroy_operation(subscript_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_array_subscripts

end program test_hlfir_expression_generation