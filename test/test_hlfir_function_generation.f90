program test_hlfir_function_generation
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    use function_gen
    implicit none

    logical :: all_tests_passed

    print *, "=== HLFIR Function Generation Tests (C API) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_function_signature_generation()) all_tests_passed = .false.
    if (.not. test_parameter_handling()) all_tests_passed = .false.
    if (.not. test_local_variable_declarations()) all_tests_passed = .false.
    if (.not. test_return_value_handling()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All HLFIR function generation tests passed!"
        stop 0
    else
        print *, "Some HLFIR function generation tests failed!"
        stop 1
    end if

contains

    function test_function_signature_generation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: func_op
        type(mlir_type_t) :: int_type, real_type
        type(mlir_type_t), dimension(2) :: param_types
        type(mlir_type_t) :: return_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create parameter and return types
        int_type = create_integer_type(context, 32)
        real_type = create_float_type(context, 64)
        param_types = [int_type, real_type]
        return_type = int_type
        
        passed = passed .and. int_type%is_valid()
        passed = passed .and. real_type%is_valid()
        
        ! Test: Generate function signature using C API
        func_op = generate_function_signature(builder, "calculate", param_types, return_type)
        passed = passed .and. func_op%is_valid()
        
        ! Verify function signature was created properly
        passed = passed .and. function_has_correct_signature(func_op, "calculate", param_types, return_type)
        
        if (passed) then
            print *, "PASS: test_function_signature_generation"
        else
            print *, "FAIL: test_function_signature_generation"
        end if
        
        ! Cleanup
        call destroy_operation(func_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_function_signature_generation

    function test_parameter_handling() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: func_op
        type(mlir_value_t), dimension(:), allocatable :: params
        type(mlir_type_t) :: int_type, ref_type
        type(mlir_type_t), dimension(2) :: param_types
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create parameter types
        int_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, int_type)
        param_types = [int_type, ref_type]
        
        ! Test: Generate function with parameter handling
        func_op = generate_function_with_parameters(builder, "process", &
            param_types, ["value ", "result"])
        passed = passed .and. func_op%is_valid()
        
        ! Test: Extract function parameters
        params = extract_function_parameters(func_op)
        passed = passed .and. size(params) == 2
        
        ! Test: Verify parameter names and types
        passed = passed .and. parameter_has_name(params(1), "value")
        passed = passed .and. parameter_has_name(params(2), "result")
        
        if (passed) then
            print *, "PASS: test_parameter_handling"
        else
            print *, "FAIL: test_parameter_handling"
        end if
        
        ! Cleanup
        if (allocated(params)) deallocate(params)
        call destroy_operation(func_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_parameter_handling

    function test_local_variable_declarations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: func_op, declare_op
        type(mlir_type_t) :: real_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create variable type
        real_type = create_float_type(context, 64)
        passed = passed .and. real_type%is_valid()
        
        ! Test: Generate function with local variable declarations
        func_op = create_function_with_locals(builder, "compute", &
            [mlir_type_t ::], real_type)
        passed = passed .and. func_op%is_valid()
        
        ! Test: Add local variable declaration within function
        declare_op = declare_local_variable(builder, real_type, "temp")
        passed = passed .and. declare_op%is_valid()
        
        ! Test: Verify variable was declared properly
        passed = passed .and. variable_is_local(declare_op)
        passed = passed .and. variable_has_type(declare_op, real_type)
        
        if (passed) then
            print *, "PASS: test_local_variable_declarations"
        else
            print *, "FAIL: test_local_variable_declarations"
        end if
        
        ! Cleanup
        call destroy_operation(declare_op)
        call destroy_operation(func_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_local_variable_declarations

    function test_return_value_handling() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: func_op, return_op
        type(mlir_value_t) :: return_value
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create return type
        int_type = create_integer_type(context, 32)
        passed = passed .and. int_type%is_valid()
        
        ! Test: Generate function with return value
        func_op = create_function_with_return(builder, "get_value", &
            [mlir_type_t ::], int_type)
        passed = passed .and. func_op%is_valid()
        
        ! Test: Create return value
        return_value = create_constant_value(builder, int_type, 42)
        passed = passed .and. return_value%is_valid()
        
        ! Test: Generate return operation
        return_op = generate_return_with_value(builder, return_value)
        passed = passed .and. return_op%is_valid()
        
        ! Test: Verify return operation has correct value
        passed = passed .and. return_has_value(return_op, return_value)
        
        if (passed) then
            print *, "PASS: test_return_value_handling"
        else
            print *, "FAIL: test_return_value_handling"
        end if
        
        ! Cleanup
        call destroy_operation(return_op)
        call destroy_operation(func_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_return_value_handling

end program test_hlfir_function_generation