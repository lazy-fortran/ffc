program test_hlfir_statement_generation
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    use statement_gen
    implicit none

    logical :: all_tests_passed

    print *, "=== HLFIR Statement Generation Tests (C API) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_assignment_statements()) all_tests_passed = .false.
    if (.not. test_if_then_else_statements()) all_tests_passed = .false.
    if (.not. test_do_loop_statements()) all_tests_passed = .false.
    if (.not. test_while_loop_statements()) all_tests_passed = .false.
    if (.not. test_select_case_statements()) all_tests_passed = .false.
    if (.not. test_print_statements()) all_tests_passed = .false.
    if (.not. test_read_statements()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All HLFIR statement generation tests passed!"
        stop 0
    else
        print *, "Some HLFIR statement generation tests failed!"
        stop 1
    end if

contains

    function test_assignment_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: assign_op
        type(mlir_value_t) :: lhs, rhs
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create integer type and values
        int_type = create_integer_type(context, 32)
        lhs = create_dummy_value(int_type)
        rhs = create_dummy_value(int_type)
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. lhs%is_valid()
        passed = passed .and. rhs%is_valid()
        
        ! Test: Generate assignment statement using HLFIR
        assign_op = generate_assignment_statement(builder, lhs, rhs)
        passed = passed .and. assign_op%is_valid()
        
        ! Verify assignment operation was created properly
        passed = passed .and. is_assignment_operation(assign_op)
        passed = passed .and. assignment_has_operands(assign_op, lhs, rhs)
        
        if (passed) then
            print *, "PASS: test_assignment_statements"
        else
            print *, "FAIL: test_assignment_statements"
        end if
        
        ! Cleanup
        call destroy_operation(assign_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_assignment_statements

    function test_if_then_else_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: if_op, then_block, else_block
        type(mlir_value_t) :: condition
        type(mlir_type_t) :: bool_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create boolean condition
        bool_type = create_integer_type(context, 1)  ! i1 for boolean
        condition = create_dummy_value(bool_type)
        
        passed = passed .and. bool_type%is_valid()
        passed = passed .and. condition%is_valid()
        
        ! Test: Generate if-then-else statement using SCF dialect
        if_op = generate_if_then_else_statement(builder, condition)
        passed = passed .and. if_op%is_valid()
        
        ! Test: Create then and else blocks
        then_block = create_then_block(builder, if_op)
        else_block = create_else_block(builder, if_op)
        passed = passed .and. then_block%is_valid()
        passed = passed .and. else_block%is_valid()
        
        ! Verify if operation structure
        passed = passed .and. is_if_operation(if_op)
        passed = passed .and. if_has_condition(if_op, condition)
        
        if (passed) then
            print *, "PASS: test_if_then_else_statements"
        else
            print *, "FAIL: test_if_then_else_statements"
        end if
        
        ! Cleanup
        call destroy_operation(else_block)
        call destroy_operation(then_block)
        call destroy_operation(if_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_if_then_else_statements

    function test_do_loop_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: loop_op, loop_body
        type(mlir_value_t) :: lower_bound, upper_bound, step
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create loop bounds
        int_type = create_integer_type(context, 32)
        lower_bound = create_dummy_value(int_type)
        upper_bound = create_dummy_value(int_type)
        step = create_dummy_value(int_type)
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. lower_bound%is_valid()
        passed = passed .and. upper_bound%is_valid()
        passed = passed .and. step%is_valid()
        
        ! Test: Generate do loop statement using SCF dialect
        loop_op = generate_do_loop_statement(builder, lower_bound, upper_bound, step)
        passed = passed .and. loop_op%is_valid()
        
        ! Test: Create loop body
        loop_body = create_loop_body(builder, loop_op)
        passed = passed .and. loop_body%is_valid()
        
        ! Verify loop operation structure
        passed = passed .and. is_loop_operation(loop_op)
        passed = passed .and. loop_has_bounds(loop_op, lower_bound, upper_bound, step)
        
        if (passed) then
            print *, "PASS: test_do_loop_statements"
        else
            print *, "FAIL: test_do_loop_statements"
        end if
        
        ! Cleanup
        call destroy_operation(loop_body)
        call destroy_operation(loop_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_do_loop_statements

    function test_while_loop_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: while_op, while_body
        type(mlir_value_t) :: condition
        type(mlir_type_t) :: bool_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create while condition
        bool_type = create_integer_type(context, 1)  ! i1 for boolean
        condition = create_dummy_value(bool_type)
        
        passed = passed .and. bool_type%is_valid()
        passed = passed .and. condition%is_valid()
        
        ! Test: Generate while loop statement using SCF dialect
        while_op = generate_while_loop_statement(builder, condition)
        passed = passed .and. while_op%is_valid()
        
        ! Test: Create while body
        while_body = create_while_body(builder, while_op)
        passed = passed .and. while_body%is_valid()
        
        ! Verify while operation structure
        passed = passed .and. is_while_operation(while_op)
        passed = passed .and. while_has_condition(while_op, condition)
        
        if (passed) then
            print *, "PASS: test_while_loop_statements"
        else
            print *, "FAIL: test_while_loop_statements"
        end if
        
        ! Cleanup
        call destroy_operation(while_body)
        call destroy_operation(while_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_while_loop_statements

    function test_select_case_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: select_op
        type(mlir_value_t) :: selector
        type(mlir_type_t) :: int_type
        integer, dimension(3) :: case_values
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create selector and case values
        int_type = create_integer_type(context, 32)
        selector = create_dummy_value(int_type)
        case_values = [1, 2, 3]
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. selector%is_valid()
        
        ! Test: Generate select case statement
        select_op = generate_select_case_statement(builder, selector, case_values)
        passed = passed .and. select_op%is_valid()
        
        ! Verify select operation structure
        passed = passed .and. is_select_operation(select_op)
        passed = passed .and. select_has_cases(select_op, case_values)
        
        if (passed) then
            print *, "PASS: test_select_case_statements"
        else
            print *, "FAIL: test_select_case_statements"
        end if
        
        ! Cleanup
        call destroy_operation(select_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_select_case_statements

    function test_print_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: print_op
        type(mlir_value_t) :: print_value
        type(mlir_type_t) :: int_type
        character(len=32) :: format_string
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create print value
        int_type = create_integer_type(context, 32)
        print_value = create_dummy_value(int_type)
        format_string = "*, i0"
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. print_value%is_valid()
        
        ! Test: Generate print statement using FIR I/O operations
        print_op = generate_print_statement(builder, format_string, [print_value])
        passed = passed .and. print_op%is_valid()
        
        ! Verify print operation structure
        passed = passed .and. is_print_operation(print_op)
        passed = passed .and. print_has_format(print_op, format_string)
        
        if (passed) then
            print *, "PASS: test_print_statements"
        else
            print *, "FAIL: test_print_statements"
        end if
        
        ! Cleanup
        call destroy_operation(print_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_print_statements

    function test_read_statements() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: read_op
        type(mlir_value_t) :: read_target
        type(mlir_type_t) :: real_type
        character(len=32) :: format_string
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create read target
        real_type = create_float_type(context, 64)
        read_target = create_dummy_value(real_type)
        format_string = "*, f0.0"
        
        passed = passed .and. real_type%is_valid()
        passed = passed .and. read_target%is_valid()
        
        ! Test: Generate read statement using FIR I/O operations
        read_op = generate_read_statement(builder, format_string, [read_target])
        passed = passed .and. read_op%is_valid()
        
        ! Verify read operation structure
        passed = passed .and. is_read_operation(read_op)
        passed = passed .and. read_has_targets(read_op, [read_target])
        
        if (passed) then
            print *, "PASS: test_read_statements"
        else
            print *, "FAIL: test_read_statements"
        end if
        
        ! Cleanup
        call destroy_operation(read_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_read_statements

end program test_hlfir_statement_generation