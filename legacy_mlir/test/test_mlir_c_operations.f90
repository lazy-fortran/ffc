program test_mlir_c_operations
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Operation Builder Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_operation_state_creation()) all_tests_passed = .false.
    if (.not. test_operand_addition()) all_tests_passed = .false.
    if (.not. test_result_type_specification()) all_tests_passed = .false.
    if (.not. test_attribute_attachment()) all_tests_passed = .false.
    if (.not. test_operation_creation_and_verification()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API operation builder tests passed!"
        stop 0
    else
        print *, "Some MLIR C API operation builder tests failed!"
        stop 1
    end if

contains

    function test_operation_state_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: op_state
        type(mlir_string_ref_t) :: name
        
        passed = .true.
        
        ! Create context and location
        context = create_mlir_context()
        loc = create_unknown_location(context)
        
        ! Test operation state creation
        name = create_string_ref("test.op")
        op_state = create_operation_state(name, loc)
        passed = passed .and. op_state%is_valid()
        
        ! Test state cleanup
        call destroy_operation_state(op_state)
        passed = passed .and. .not. op_state%is_valid()
        
        if (passed) then
            print *, "PASS: test_operation_state_creation"
        else
            print *, "FAIL: test_operation_state_creation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_operation_state_creation

    function test_operand_addition() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: op_state
        type(mlir_value_t) :: operand1, operand2
        type(mlir_value_t), dimension(2) :: operands
        type(mlir_string_ref_t) :: name
        
        passed = .true.
        
        ! Create context and location
        context = create_mlir_context()
        loc = create_unknown_location(context)
        
        ! Create operation state
        name = create_string_ref("test.binary_op")
        op_state = create_operation_state(name, loc)
        
        ! Create dummy operands (in real code these would be SSA values)
        operand1 = create_dummy_value(context)
        operand2 = create_dummy_value(context)
        
        ! Test single operand addition
        call add_operand(op_state, operand1)
        passed = passed .and. (get_num_operands(op_state) == 1)
        
        call add_operand(op_state, operand2)
        passed = passed .and. (get_num_operands(op_state) == 2)
        
        ! Test bulk operand addition
        call destroy_operation_state(op_state)
        op_state = create_operation_state(name, loc)
        
        operands(1) = operand1
        operands(2) = operand2
        call add_operands(op_state, operands)
        passed = passed .and. (get_num_operands(op_state) == 2)
        
        if (passed) then
            print *, "PASS: test_operand_addition"
        else
            print *, "FAIL: test_operand_addition"
        end if
        
        ! Cleanup
        call destroy_operation_state(op_state)
        call destroy_mlir_context(context)
    end function test_operand_addition

    function test_result_type_specification() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: op_state
        type(mlir_type_t) :: i32_type, f64_type
        type(mlir_type_t), dimension(2) :: result_types
        type(mlir_string_ref_t) :: name
        
        passed = .true.
        
        ! Create context and location
        context = create_mlir_context()
        loc = create_unknown_location(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        f64_type = create_float_type(context, 64)
        
        ! Create operation state
        name = create_string_ref("test.multi_result_op")
        op_state = create_operation_state(name, loc)
        
        ! Test single result type
        call add_result(op_state, i32_type)
        passed = passed .and. (get_num_results(op_state) == 1)
        
        call add_result(op_state, f64_type)
        passed = passed .and. (get_num_results(op_state) == 2)
        
        ! Test bulk result addition
        call destroy_operation_state(op_state)
        op_state = create_operation_state(name, loc)
        
        result_types(1) = i32_type
        result_types(2) = f64_type
        call add_results(op_state, result_types)
        passed = passed .and. (get_num_results(op_state) == 2)
        
        if (passed) then
            print *, "PASS: test_result_type_specification"
        else
            print *, "FAIL: test_result_type_specification"
        end if
        
        ! Cleanup
        call destroy_operation_state(op_state)
        call destroy_mlir_context(context)
    end function test_result_type_specification

    function test_attribute_attachment() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: op_state
        type(mlir_type_t) :: i32_type
        type(mlir_attribute_t) :: int_attr, str_attr
        type(mlir_string_ref_t) :: op_name, attr_name1, attr_name2
        
        passed = .true.
        
        ! Create context and location
        context = create_mlir_context()
        loc = create_unknown_location(context)
        
        ! Create operation state
        op_name = create_string_ref("test.attributed_op")
        op_state = create_operation_state(op_name, loc)
        
        ! Create attributes
        i32_type = create_integer_type(context, 32)
        int_attr = create_integer_attribute(context, i32_type, 42_c_int64_t)
        str_attr = create_string_attribute(context, "test_value")
        
        ! Test attribute attachment
        attr_name1 = create_string_ref("count")
        call add_attribute(op_state, attr_name1, int_attr)
        passed = passed .and. (get_num_attributes(op_state) == 1)
        
        attr_name2 = create_string_ref("name")
        call add_attribute(op_state, attr_name2, str_attr)
        passed = passed .and. (get_num_attributes(op_state) == 2)
        
        if (passed) then
            print *, "PASS: test_attribute_attachment"
        else
            print *, "FAIL: test_attribute_attachment"
        end if
        
        ! Cleanup
        call destroy_operation_state(op_state)
        call destroy_mlir_context(context)
    end function test_attribute_attachment

    function test_operation_creation_and_verification() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: loc
        type(mlir_operation_state_t) :: op_state
        type(mlir_operation_t) :: operation
        type(mlir_type_t) :: i32_type
        type(mlir_string_ref_t) :: name
        
        passed = .true.
        
        ! Create context and location
        context = create_mlir_context()
        loc = create_unknown_location(context)
        
        ! Create operation state with result
        name = create_string_ref("test.complete_op")
        op_state = create_operation_state(name, loc)
        
        i32_type = create_integer_type(context, 32)
        call add_result(op_state, i32_type)
        
        ! Create operation from state
        operation = create_operation(op_state)
        passed = passed .and. operation%is_valid()
        
        ! Verify operation
        passed = passed .and. verify_operation(operation)
        
        ! Test operation properties
        passed = passed .and. (get_operation_num_results(operation) == 1)
        passed = passed .and. (get_operation_name(operation) == "test.complete_op")
        
        if (passed) then
            print *, "PASS: test_operation_creation_and_verification"
        else
            print *, "FAIL: test_operation_creation_and_verification"
        end if
        
        ! Cleanup
        call destroy_operation(operation)
        call destroy_mlir_context(context)
    end function test_operation_creation_and_verification

end program test_mlir_c_operations