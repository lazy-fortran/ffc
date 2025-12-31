program test_hlfir_program_generation
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    use program_gen
    implicit none

    logical :: all_tests_passed

    print *, "=== HLFIR Program Generation Tests (C API) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_empty_program_hlfir_generation()) all_tests_passed = .false.
    if (.not. test_module_with_hlfir_functions()) all_tests_passed = .false.
    if (.not. test_module_hlfir_variables()) all_tests_passed = .false.
    if (.not. test_program_with_dependencies()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All HLFIR program generation tests passed!"
        stop 0
    else
        print *, "Some HLFIR program generation tests failed!"
        stop 1
    end if

contains

    function test_empty_program_hlfir_generation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: main_func, return_op
        type(mlir_module_t) :: module
        type(mlir_location_t) :: location
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        location = create_unknown_location(context)
        module = create_empty_module(location)
        passed = passed .and. builder%is_valid()
        passed = passed .and. module%is_valid()
        
        ! Test: Generate empty program using C API operations
        ! This should create a func.func @main() with func.return
        main_func = generate_empty_main_function(builder)
        passed = passed .and. main_func%is_valid()
        
        ! Verify the function was created properly
        passed = passed .and. operation_has_name(main_func, "@main")
        
        ! Test return operation creation
        return_op = create_return_operation(builder)
        passed = passed .and. return_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_empty_program_hlfir_generation"
        else
            print *, "FAIL: test_empty_program_hlfir_generation"
        end if
        
        ! Cleanup (module is cleaned up with context)
        call destroy_operation(return_op)
        call destroy_operation(main_func)
        call destroy_mlir_builder(builder)  
        call destroy_mlir_context(context)
    end function test_empty_program_hlfir_generation

    function test_module_with_hlfir_functions() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: func_op, declare_op
        type(mlir_type_t) :: int_type, ref_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create types for function parameters
        int_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, int_type)
        passed = passed .and. int_type%is_valid()
        passed = passed .and. ref_type%is_valid()
        
        ! Test: Generate function with HLFIR operations
        func_op = generate_hlfir_function(builder, "add", [ref_type, ref_type], ref_type)
        passed = passed .and. func_op%is_valid()
        
        ! Test: Create HLFIR declare operation for variables
        declare_op = create_hlfir_declare_operation(builder, ref_type, "a")
        passed = passed .and. declare_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_module_with_hlfir_functions"
        else
            print *, "FAIL: test_module_with_hlfir_functions"
        end if
        
        ! Cleanup
        call destroy_operation(declare_op)
        call destroy_operation(func_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_module_with_hlfir_functions

    function test_module_hlfir_variables() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: global_op, declare_op
        type(mlir_type_t) :: int_type
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create integer type
        int_type = create_integer_type(context, 32)
        passed = passed .and. int_type%is_valid()
        
        ! Test: Create global variable using C API
        global_op = create_global_variable_operation(builder, int_type, "PI_APPROX")
        passed = passed .and. global_op%is_valid()
        
        ! Test: Create HLFIR declare for global access
        declare_op = create_hlfir_declare_global(builder, int_type, "global_var")
        passed = passed .and. declare_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_module_hlfir_variables"
        else
            print *, "FAIL: test_module_hlfir_variables"
        end if
        
        ! Cleanup
        call destroy_operation(declare_op)
        call destroy_operation(global_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_module_hlfir_variables

    function test_program_with_dependencies() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_operation_t) :: main_func
        logical :: has_dependencies
        
        passed = .true.
        
        ! Create MLIR context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Test: Generate program with module dependencies
        main_func = generate_main_with_dependencies(builder, ["iso_fortran_env", "math_utils     "])
        passed = passed .and. main_func%is_valid()
        
        ! Test: Check dependency tracking
        has_dependencies = operation_has_dependencies(main_func)
        passed = passed .and. has_dependencies
        
        if (passed) then
            print *, "PASS: test_program_with_dependencies"
        else
            print *, "FAIL: test_program_with_dependencies"
        end if
        
        ! Cleanup
        call destroy_operation(main_func)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_program_with_dependencies


end program test_hlfir_program_generation