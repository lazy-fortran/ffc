program test_mlir_pass_manager
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use standard_dialects
    use ffc_pass_manager
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR Pass Manager Tests (C API) ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_pass_manager_creation()) all_tests_passed = .false.
    if (.not. test_pass_pipeline_configuration()) all_tests_passed = .false.
    if (.not. test_pass_execution()) all_tests_passed = .false.
    if (.not. test_pass_verification()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR pass manager tests passed!"
        stop 0
    else
        print *, "Some MLIR pass manager tests failed!"
        stop 1
    end if

contains

    function test_pass_manager_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_pass_manager_t) :: pass_manager
        
        passed = .true.
        
        ! Create MLIR context
        context = create_mlir_context()
        passed = passed .and. context%is_valid()
        
        ! Test: Create pass manager using MLIR C API
        pass_manager = create_pass_manager(context)
        passed = passed .and. pass_manager%is_valid()
        
        ! Verify pass manager properties
        passed = passed .and. pass_manager_has_context(pass_manager, context)
        passed = passed .and. pass_manager_is_empty(pass_manager)
        
        ! Test pass manager configuration
        call configure_pass_manager(pass_manager, "builtin.module")
        passed = passed .and. pass_manager_has_anchor(pass_manager, "builtin.module")
        
        if (passed) then
            print *, "PASS: test_pass_manager_creation"
        else
            print *, "FAIL: test_pass_manager_creation"
        end if
        
        ! Cleanup
        call destroy_pass_manager(pass_manager)
        call destroy_mlir_context(context)
    end function test_pass_manager_creation

    function test_pass_pipeline_configuration() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_pass_manager_t) :: pass_manager
        type(mlir_pass_pipeline_t) :: pipeline
        character(len=256), dimension(3) :: pass_names
        
        passed = .true.
        
        ! Create MLIR context and pass manager
        context = create_mlir_context()
        pass_manager = create_pass_manager(context)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. pass_manager%is_valid()
        
        ! Test: Configure pass pipeline
        pass_names = ["canonicalize      ", "cse               ", "sccp              "]
        pipeline = create_pass_pipeline(pass_manager, pass_names)
        passed = passed .and. pipeline%is_valid()
        
        ! Verify pipeline configuration
        passed = passed .and. pipeline_has_passes(pipeline, pass_names)
        passed = passed .and. pipeline_pass_count(pipeline) == 3
        
        ! Test: Add individual passes to pipeline
        call add_pass_to_pipeline(pipeline, "inline")
        passed = passed .and. pipeline_pass_count(pipeline) == 4
        passed = passed .and. pipeline_has_pass(pipeline, "inline")
        
        ! Test: Parse pass pipeline from string
        call parse_pass_pipeline(pass_manager, "builtin.module(canonicalize,cse)")
        passed = passed .and. pass_manager_has_passes(pass_manager)
        
        if (passed) then
            print *, "PASS: test_pass_pipeline_configuration"
        else
            print *, "FAIL: test_pass_pipeline_configuration"
        end if
        
        ! Cleanup
        call destroy_pass_pipeline(pipeline)
        call destroy_pass_manager(pass_manager)
        call destroy_mlir_context(context)
    end function test_pass_pipeline_configuration

    function test_pass_execution() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_pass_manager_t) :: pass_manager
        type(mlir_module_t) :: module
        type(mlir_operation_t) :: func_op
        type(mlir_location_t) :: location
        logical :: success
        
        passed = .true.
        
        ! Create MLIR context, builder, and pass manager
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        pass_manager = create_pass_manager(context)
        location = create_unknown_location(context)
        module = create_empty_module(location)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. builder%is_valid()
        passed = passed .and. pass_manager%is_valid()
        passed = passed .and. module%is_valid()
        
        ! Create a simple function for testing
        func_op = create_test_function(builder)
        passed = passed .and. func_op%is_valid()
        
        ! Test: Configure and run passes
        call parse_pass_pipeline(pass_manager, "builtin.module(canonicalize)")
        
        ! Execute passes on module
        success = run_passes(pass_manager, module)
        passed = passed .and. success
        
        ! Verify pass execution effects
        passed = passed .and. module_was_transformed(module)
        passed = passed .and. pass_manager_succeeded(pass_manager)
        
        ! Test: Run passes with verification
        success = run_passes_with_verification(pass_manager, module)
        passed = passed .and. success
        
        if (passed) then
            print *, "PASS: test_pass_execution"
        else
            print *, "FAIL: test_pass_execution"
        end if
        
        ! Cleanup
        call destroy_operation(func_op)
        call destroy_pass_manager(pass_manager)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_pass_execution

    function test_pass_verification() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_pass_manager_t) :: pass_manager
        type(mlir_module_t) :: module, invalid_module
        type(mlir_location_t) :: location
        logical :: verification_result
        
        passed = .true.
        
        ! Create MLIR context and pass manager
        context = create_mlir_context()
        pass_manager = create_pass_manager(context)
        location = create_unknown_location(context)
        
        passed = passed .and. context%is_valid()
        passed = passed .and. pass_manager%is_valid()
        
        ! Test: Verify valid module
        module = create_valid_test_module(location)
        passed = passed .and. module%is_valid()
        
        verification_result = verify_module(pass_manager, module)
        passed = passed .and. verification_result
        
        ! Test: Verify invalid module (should fail)
        invalid_module = create_invalid_test_module(location)
        passed = passed .and. invalid_module%is_valid()
        
        verification_result = verify_module(pass_manager, invalid_module)
        passed = passed .and. (.not. verification_result)  ! Should fail
        
        ! Test: Enable/disable verification in pass manager
        call enable_pass_verification(pass_manager)
        passed = passed .and. pass_manager_has_verification_enabled(pass_manager)
        
        call disable_pass_verification(pass_manager)
        passed = passed .and. (.not. pass_manager_has_verification_enabled(pass_manager))
        
        ! Test: Get verification diagnostics
        call enable_pass_verification(pass_manager)
        verification_result = verify_module(pass_manager, invalid_module)
        passed = passed .and. pass_manager_has_diagnostics(pass_manager)
        
        if (passed) then
            print *, "PASS: test_pass_verification"
        else
            print *, "FAIL: test_pass_verification"
        end if
        
        ! Cleanup
        call destroy_pass_manager(pass_manager)
        call destroy_mlir_context(context)
    end function test_pass_verification


end program test_mlir_pass_manager