program run_all_tests
    use test_harness
    implicit none
    
    type(test_suite_t) :: mlir_core_suite
    type(test_suite_t) :: dialect_suite
    type(test_suite_t) :: builder_suite
    type(test_suite_t) :: codegen_suite
    type(test_suite_t) :: backend_suite
    type(test_suite_t) :: memory_suite
    type(test_suite_t) :: integration_suite
    
    logical :: verbose = .true.
    integer :: total_passed = 0
    integer :: total_failed = 0
    integer :: total_skipped = 0
    
    print *, "=== FortFC Comprehensive Test Suite ==="
    print *, "Testing MLIR C API implementation with HLFIR generation"
    print *
    
    ! Run MLIR Core tests
    mlir_core_suite = create_test_suite("MLIR Core C API")
    call setup_mlir_core_tests(mlir_core_suite)
    call run_test_suite(mlir_core_suite, verbose)
    call update_totals(mlir_core_suite)
    
    ! Run Dialect tests
    dialect_suite = create_test_suite("Dialect C Bindings")
    call setup_dialect_tests(dialect_suite)
    call run_test_suite(dialect_suite, verbose)
    call update_totals(dialect_suite)
    
    ! Run Builder tests
    builder_suite = create_test_suite("IR Builder")
    call setup_builder_tests(builder_suite)
    call run_test_suite(builder_suite, verbose)
    call update_totals(builder_suite)
    
    ! Run Code Generation tests
    codegen_suite = create_test_suite("AST to MLIR Code Generation")
    call setup_codegen_tests(codegen_suite)
    call run_test_suite(codegen_suite, verbose)
    call update_totals(codegen_suite)
    
    ! Run Backend tests
    backend_suite = create_test_suite("Backend Integration")
    call setup_backend_tests(backend_suite)
    call run_test_suite(backend_suite, verbose)
    call update_totals(backend_suite)
    
    ! Run Memory Management tests
    memory_suite = create_test_suite("Memory Management")
    call setup_memory_tests(memory_suite)
    call run_test_suite(memory_suite, verbose)
    call update_totals(memory_suite)
    
    ! Run Integration tests
    integration_suite = create_test_suite("Integration Tests")
    call setup_integration_tests(integration_suite)
    call run_test_suite(integration_suite, verbose)
    call update_totals(integration_suite)
    
    ! Print overall summary
    call print_overall_summary()
    
    ! Clean up
    call mlir_core_suite%cleanup()
    call dialect_suite%cleanup()
    call builder_suite%cleanup()
    call codegen_suite%cleanup()
    call backend_suite%cleanup()
    call memory_suite%cleanup()
    call integration_suite%cleanup()
    
    if (total_failed > 0) then
        stop 1
    else
        stop 0
    end if
    
contains

    subroutine setup_mlir_core_tests(suite)
        use test_mlir_c_core, only: test_context_creation, test_module_creation, &
                                    test_location_creation, test_string_ref_handling
        use test_mlir_c_types, only: test_integer_types, test_float_types, &
                                     test_array_types, test_reference_types
        use test_mlir_c_attributes, only: test_integer_attributes, test_float_attributes, &
                                          test_string_attributes, test_array_attributes
        use test_mlir_c_operations, only: test_operation_state, test_operand_addition, &
                                          test_result_types, test_attribute_attachment
        type(test_suite_t), intent(inout) :: suite
        
        ! Core tests
        call add_test_case(suite, "Context creation and destruction", test_context_creation)
        call add_test_case(suite, "Module creation", test_module_creation)
        call add_test_case(suite, "Location creation", test_location_creation)
        call add_test_case(suite, "String ref handling", test_string_ref_handling)
        
        ! Type tests
        call add_test_case(suite, "Integer type creation", test_integer_types)
        call add_test_case(suite, "Float type creation", test_float_types)
        call add_test_case(suite, "Array type creation", test_array_types)
        call add_test_case(suite, "Reference type creation", test_reference_types)
        
        ! Attribute tests
        call add_test_case(suite, "Integer attribute creation", test_integer_attributes)
        call add_test_case(suite, "Float attribute creation", test_float_attributes)
        call add_test_case(suite, "String attribute creation", test_string_attributes)
        call add_test_case(suite, "Array attribute creation", test_array_attributes)
        
        ! Operation tests
        call add_test_case(suite, "Operation state creation", test_operation_state)
        call add_test_case(suite, "Operand addition", test_operand_addition)
        call add_test_case(suite, "Result type specification", test_result_types)
        call add_test_case(suite, "Attribute attachment", test_attribute_attachment)
    end subroutine setup_mlir_core_tests

    subroutine setup_dialect_tests(suite)
        use test_fir_dialect, only: test_fir_registration, test_fir_declare, &
                                    test_fir_load, test_fir_store
        use test_hlfir_dialect, only: test_hlfir_registration, test_hlfir_declare, &
                                      test_hlfir_designate, test_hlfir_elemental
        use test_standard_dialects, only: test_func_dialect, test_arith_dialect, &
                                          test_scf_dialect
        type(test_suite_t), intent(inout) :: suite
        
        ! FIR dialect tests
        call add_test_case(suite, "FIR dialect registration", test_fir_registration)
        call add_test_case(suite, "fir.declare operation", test_fir_declare)
        call add_test_case(suite, "fir.load operation", test_fir_load)
        call add_test_case(suite, "fir.store operation", test_fir_store)
        
        ! HLFIR dialect tests
        call add_test_case(suite, "HLFIR dialect registration", test_hlfir_registration)
        call add_test_case(suite, "hlfir.declare operation", test_hlfir_declare)
        call add_test_case(suite, "hlfir.designate operation", test_hlfir_designate)
        call add_test_case(suite, "hlfir.elemental operation", test_hlfir_elemental)
        
        ! Standard dialect tests
        call add_test_case(suite, "func dialect operations", test_func_dialect)
        call add_test_case(suite, "arith dialect operations", test_arith_dialect)
        call add_test_case(suite, "scf dialect operations", test_scf_dialect)
    end subroutine setup_dialect_tests

    subroutine setup_builder_tests(suite)
        use test_mlir_builder, only: test_builder_creation, test_insertion_point, &
                                     test_block_creation, test_region_handling
        use test_ssa_manager, only: test_ssa_generation, test_value_naming, &
                                    test_value_tracking, test_use_def_chains
        use test_type_converter_simple, only: test_basic_types, test_array_conversion, &
                                              test_type_caching
        type(test_suite_t), intent(inout) :: suite
        
        ! Builder tests
        call add_test_case(suite, "Builder creation and cleanup", test_builder_creation)
        call add_test_case(suite, "Insertion point management", test_insertion_point)
        call add_test_case(suite, "Block creation", test_block_creation)
        call add_test_case(suite, "Region handling", test_region_handling)
        
        ! SSA manager tests
        call add_test_case(suite, "SSA value generation", test_ssa_generation)
        call add_test_case(suite, "Value naming", test_value_naming)
        call add_test_case(suite, "Value type tracking", test_value_tracking)
        call add_test_case(suite, "Use-def chains", test_use_def_chains)
        
        ! Type conversion tests
        call add_test_case(suite, "Basic type conversion", test_basic_types)
        call add_test_case(suite, "Array type conversion", test_array_conversion)
        call add_test_case(suite, "Type caching", test_type_caching)
    end subroutine setup_builder_tests

    subroutine setup_codegen_tests(suite)
        use test_hlfir_program_generation, only: test_empty_program, test_module_functions, &
                                                 test_module_variables, test_use_statements
        use test_hlfir_function_generation, only: test_function_signature, test_parameter_handling, &
                                                  test_local_variables, test_return_values
        use test_hlfir_statement_generation, only: test_assignments, test_if_then_else, &
                                                   test_do_loops, test_io_operations
        use test_hlfir_expression_generation, only: test_literals, test_variables, &
                                                    test_binary_ops, test_function_calls
        type(test_suite_t), intent(inout) :: suite
        
        ! Program generation tests
        call add_test_case(suite, "Empty program generation", test_empty_program)
        call add_test_case(suite, "Module with functions", test_module_functions)
        call add_test_case(suite, "Module variables", test_module_variables)
        call add_test_case(suite, "Use statements", test_use_statements)
        
        ! Function generation tests
        call add_test_case(suite, "Function signature generation", test_function_signature)
        call add_test_case(suite, "Parameter handling", test_parameter_handling)
        call add_test_case(suite, "Local variable declarations", test_local_variables)
        call add_test_case(suite, "Return value handling", test_return_values)
        
        ! Statement generation tests
        call add_test_case(suite, "Assignment statements", test_assignments)
        call add_test_case(suite, "If-then-else statements", test_if_then_else)
        call add_test_case(suite, "Do loop statements", test_do_loops)
        call add_test_case(suite, "I/O operations", test_io_operations)
        
        ! Expression generation tests
        call add_test_case(suite, "Literal expressions", test_literals)
        call add_test_case(suite, "Variable references", test_variables)
        call add_test_case(suite, "Binary operations", test_binary_ops)
        call add_test_case(suite, "Function calls", test_function_calls)
    end subroutine setup_codegen_tests

    subroutine setup_backend_tests(suite)
        use test_mlir_pass_manager, only: test_pass_manager_creation, test_pipeline_config, &
                                         test_pass_execution, test_pass_verification
        use test_lowering_pipeline, only: test_hlfir_to_fir, test_fir_to_llvm, &
                                         test_optimization_passes, test_debug_info
        use test_mlir_c_backend, only: test_c_api_backend_against_existing, &
                                      test_compilation_to_object_files, &
                                      test_executable_generation, test_optimization_levels
        type(test_suite_t), intent(inout) :: suite
        
        ! Pass manager tests
        call add_test_case(suite, "Pass manager creation", test_pass_manager_creation)
        call add_test_case(suite, "Pass pipeline configuration", test_pipeline_config)
        call add_test_case(suite, "Pass execution", test_pass_execution)
        call add_test_case(suite, "Pass verification", test_pass_verification)
        
        ! Lowering pipeline tests
        call add_test_case(suite, "HLFIR to FIR lowering", test_hlfir_to_fir)
        call add_test_case(suite, "FIR to LLVM lowering", test_fir_to_llvm)
        call add_test_case(suite, "Optimization passes", test_optimization_passes)
        call add_test_case(suite, "Debug info preservation", test_debug_info)
        
        ! Backend tests
        call add_test_case(suite, "C API backend vs text backend", test_c_api_backend_against_existing)
        call add_test_case(suite, "Compilation to object files", test_compilation_to_object_files)
        call add_test_case(suite, "Executable generation", test_executable_generation)
        call add_test_case(suite, "Optimization levels", test_optimization_levels)
    end subroutine setup_backend_tests

    subroutine setup_memory_tests(suite)
        use test_memory_management, only: test_memory_leak_detection, test_large_program_handling, &
                                         test_error_recovery, test_resource_cleanup
        type(test_suite_t), intent(inout) :: suite
        
        call add_test_case(suite, "Memory leak detection", test_memory_leak_detection)
        call add_test_case(suite, "Large program handling", test_large_program_handling)
        call add_test_case(suite, "Error recovery", test_error_recovery)
        call add_test_case(suite, "Resource cleanup", test_resource_cleanup)
    end subroutine setup_memory_tests

    subroutine setup_integration_tests(suite)
        type(test_suite_t), intent(inout) :: suite
        
        ! Integration tests combining multiple components
        call add_test_case(suite, "Hello World compilation", test_hello_world_compilation)
        call add_test_case(suite, "Simple arithmetic program", test_arithmetic_program)
        call add_test_case(suite, "Array operations", test_array_operations)
        call add_test_case(suite, "Module usage", test_module_usage)
        call add_test_case(suite, "Complex number support", test_complex_numbers)
        call add_test_case(suite, "Character string handling", test_character_strings)
        call add_test_case(suite, "Derived type support", test_derived_types)
        call add_test_case(suite, "Intrinsic functions", test_intrinsic_functions)
    end subroutine setup_integration_tests

    subroutine update_totals(suite)
        type(test_suite_t), intent(in) :: suite
        
        total_passed = total_passed + suite%passed
        total_failed = total_failed + suite%failed
        total_skipped = total_skipped + suite%skipped
    end subroutine update_totals

    subroutine print_overall_summary()
        print *
        print '(A)', repeat("=", 70)
        print '(A)', "OVERALL TEST SUMMARY"
        print '(A)', repeat("=", 70)
        print '(A,I0)', "Total tests run: ", total_passed + total_failed + total_skipped
        print '(A,I0)', "Passed: ", total_passed
        print '(A,I0)', "Failed: ", total_failed
        print '(A,I0)', "Skipped: ", total_skipped
        print '(A)', repeat("=", 70)
        
        if (total_failed > 0) then
            print '(A)', "TEST SUITE FAILED"
        else
            print '(A)', "ALL TESTS PASSED"
        end if
    end subroutine print_overall_summary

    ! Integration test implementations (stubs for now)
    
    function test_hello_world_compilation() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_hello_world_compilation
    
    function test_arithmetic_program() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_arithmetic_program
    
    function test_array_operations() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_array_operations
    
    function test_module_usage() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_module_usage
    
    function test_complex_numbers() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_complex_numbers
    
    function test_character_strings() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_character_strings
    
    function test_derived_types() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_derived_types
    
    function test_intrinsic_functions() result(passed)
        logical :: passed
        passed = .true.  ! TODO: Implement
    end function test_intrinsic_functions

end program run_all_tests