program test_dialect_registry
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use dialect_registry
    use fir_dialect
    use hlfir_dialect
    use standard_dialects
    implicit none

    logical :: all_tests_passed

    print *, "=== Dialect Registry Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_dialect_set_creation()) all_tests_passed = .false.
    if (.not. test_preset_configurations()) all_tests_passed = .false.
    if (.not. test_selective_registration()) all_tests_passed = .false.
    if (.not. test_convenience_functions()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All dialect registry tests passed!"
        stop 0
    else
        print *, "Some dialect registry tests failed!"
        stop 1
    end if

contains

    function test_dialect_set_creation() result(passed)
        logical :: passed
        type(dialect_set_t) :: dialects
        
        passed = .true.
        
        ! Test empty set
        dialects = create_dialect_set()
        passed = passed .and. .not. dialects%fir
        passed = passed .and. .not. dialects%hlfir
        passed = passed .and. .not. dialects%func
        
        ! Test adding dialects
        call dialects%add("fir")
        call dialects%add("hlfir")
        passed = passed .and. dialects%fir
        passed = passed .and. dialects%hlfir
        passed = passed .and. .not. dialects%func
        
        ! Test clearing
        call dialects%clear()
        passed = passed .and. .not. dialects%fir
        passed = passed .and. .not. dialects%hlfir
        
        if (passed) then
            print *, "PASS: test_dialect_set_creation"
        else
            print *, "FAIL: test_dialect_set_creation"
        end if
    end function test_dialect_set_creation

    function test_preset_configurations() result(passed)
        logical :: passed
        type(dialect_set_t) :: fortran_set, minimal_set, opt_set
        
        passed = .true.
        
        ! Test Fortran preset
        fortran_set = create_dialect_set("fortran")
        passed = passed .and. fortran_set%fir
        passed = passed .and. fortran_set%hlfir
        passed = passed .and. fortran_set%arith
        passed = passed .and. fortran_set%scf
        passed = passed .and. fortran_set%func
        
        ! Test minimal preset
        minimal_set = create_dialect_set("minimal")
        passed = passed .and. .not. minimal_set%fir
        passed = passed .and. .not. minimal_set%hlfir
        passed = passed .and. minimal_set%func
        passed = passed .and. minimal_set%arith
        
        ! Test optimization preset
        opt_set = create_dialect_set("optimization")
        passed = passed .and. opt_set%fir
        passed = passed .and. .not. opt_set%hlfir
        passed = passed .and. opt_set%arith
        passed = passed .and. opt_set%scf
        
        if (passed) then
            print *, "PASS: test_preset_configurations"
        else
            print *, "FAIL: test_preset_configurations"
        end if
    end function test_preset_configurations

    function test_selective_registration() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(dialect_set_t) :: dialects
        type(mlir_operation_t) :: fir_op, arith_op
        type(mlir_type_t) :: i32_type
        type(mlir_value_t) :: val1, val2
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Register only selected dialects
        dialects = create_dialect_set()
        call dialects%add("fir")
        call dialects%add("arith")
        call register_dialects(context, dialects)
        
        ! Test that registered dialects work
        i32_type = create_integer_type(context, 32)
        val1 = create_dummy_value(context)
        val2 = create_dummy_value(context)
        
        fir_op = create_fir_load(context, val1, i32_type)
        passed = passed .and. fir_op%is_valid()
        
        arith_op = create_arith_addi(context, val1, val2, i32_type)
        passed = passed .and. arith_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_selective_registration"
        else
            print *, "FAIL: test_selective_registration"
        end if
        
        ! Cleanup
        call destroy_operation(fir_op)
        call destroy_operation(arith_op)
        call destroy_mlir_context(context)
    end function test_selective_registration

    function test_convenience_functions() result(passed)
        logical :: passed
        type(mlir_context_t) :: context1, context2
        type(mlir_operation_t) :: hlfir_op, func_op
        type(mlir_type_t) :: i32_type, func_type
        type(mlir_value_t) :: val
        type(mlir_region_t) :: region
        
        passed = .true.
        
        ! Test register_all_fortran_dialects
        context1 = create_mlir_context()
        call register_all_fortran_dialects(context1)
        
        i32_type = create_integer_type(context1, 32)
        val = create_dummy_value(context1)
        
        hlfir_op = create_hlfir_declare(context1, val, &
            create_string_attribute(context1, "x"), i32_type, &
            create_string_attribute(context1, "attrs"))
        passed = passed .and. hlfir_op%is_valid()
        
        ! Test register_all_standard_dialects
        context2 = create_mlir_context()
        call register_all_standard_dialects(context2)
        
        func_type = create_function_type(context2, [i32_type], [i32_type])
        region = create_empty_region(context2)
        
        func_op = create_func_func(context2, "test", func_type, region)
        passed = passed .and. func_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_convenience_functions"
        else
            print *, "FAIL: test_convenience_functions"
        end if
        
        ! Cleanup
        call destroy_operation(hlfir_op)
        call destroy_operation(func_op)
        call destroy_mlir_context(context1)
        call destroy_mlir_context(context2)
    end function test_convenience_functions

end program test_dialect_registry