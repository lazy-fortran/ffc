program test_hlfir_dialect
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use hlfir_dialect
    implicit none

    logical :: all_tests_passed

    print *, "=== HLFIR Dialect Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_hlfir_dialect_registration()) all_tests_passed = .false.
    if (.not. test_hlfir_declare_operation()) all_tests_passed = .false.
    if (.not. test_hlfir_designate_operation()) all_tests_passed = .false.
    if (.not. test_hlfir_elemental_operation()) all_tests_passed = .false.
    if (.not. test_hlfir_associate_operation()) all_tests_passed = .false.
    if (.not. test_hlfir_end_associate_operation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All HLFIR dialect tests passed!"
        stop 0
    else
        print *, "Some HLFIR dialect tests failed!"
        stop 1
    end if

contains

    function test_hlfir_dialect_registration() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        
        passed = .true.
        
        ! Create context
        context = create_mlir_context()
        
        ! Register HLFIR dialect
        call register_hlfir_dialect(context)
        
        ! Check if dialect is registered
        passed = passed .and. is_hlfir_dialect_registered(context)
        
        if (passed) then
            print *, "PASS: test_hlfir_dialect_registration"
        else
            print *, "FAIL: test_hlfir_dialect_registration"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_dialect_registration

    function test_hlfir_declare_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: declare_op
        type(mlir_type_t) :: i32_type, expr_type
        type(mlir_value_t) :: memref
        type(mlir_attribute_t) :: name_attr, fortran_attrs
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_hlfir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        expr_type = create_hlfir_expr_type(context, i32_type, 0)  ! scalar
        
        ! Create dummy memref value
        memref = create_dummy_value(context)
        
        ! Create attributes
        name_attr = create_string_attribute(context, "x")
        fortran_attrs = create_hlfir_fortran_attrs(context, .true., .false., .false.)  ! contiguous
        
        ! Create hlfir.declare operation
        declare_op = create_hlfir_declare(context, memref, name_attr, expr_type, fortran_attrs)
        passed = passed .and. declare_op%is_valid()
        passed = passed .and. verify_operation(declare_op)
        
        if (passed) then
            print *, "PASS: test_hlfir_declare_operation"
        else
            print *, "FAIL: test_hlfir_declare_operation"
        end if
        
        ! Cleanup
        call destroy_operation(declare_op)
        call destroy_mlir_context(context)
    end function test_hlfir_declare_operation

    function test_hlfir_designate_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: designate_op
        type(mlir_type_t) :: i32_type, array_type, expr_type
        type(mlir_value_t) :: array_val, index_val
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_hlfir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        array_type = create_hlfir_expr_type(context, i32_type, 1)  ! 1D array
        expr_type = create_hlfir_expr_type(context, i32_type, 0)   ! scalar result
        
        ! Create dummy values
        array_val = create_dummy_value(context)
        index_val = create_dummy_value(context)
        
        ! Create hlfir.designate operation (array indexing)
        designate_op = create_hlfir_designate(context, array_val, [index_val], expr_type)
        passed = passed .and. designate_op%is_valid()
        passed = passed .and. verify_operation(designate_op)
        
        if (passed) then
            print *, "PASS: test_hlfir_designate_operation"
        else
            print *, "FAIL: test_hlfir_designate_operation"
        end if
        
        ! Cleanup
        call destroy_operation(designate_op)
        call destroy_mlir_context(context)
    end function test_hlfir_designate_operation

    function test_hlfir_elemental_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: elemental_op
        type(mlir_type_t) :: i32_type, array_type
        type(mlir_value_t), dimension(2) :: shape_vals
        type(mlir_region_t) :: body_region
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_hlfir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        array_type = create_hlfir_expr_type(context, i32_type, 2)  ! 2D array
        
        ! Create shape values
        shape_vals(1) = create_dummy_value(context)
        shape_vals(2) = create_dummy_value(context)
        
        ! Create empty region for elemental body
        body_region = create_empty_region(context)
        
        ! Create hlfir.elemental operation
        elemental_op = create_hlfir_elemental(context, shape_vals, array_type, body_region)
        passed = passed .and. elemental_op%is_valid()
        passed = passed .and. verify_operation(elemental_op)
        
        if (passed) then
            print *, "PASS: test_hlfir_elemental_operation"
        else
            print *, "FAIL: test_hlfir_elemental_operation"
        end if
        
        ! Cleanup
        call destroy_operation(elemental_op)
        call destroy_mlir_context(context)
    end function test_hlfir_elemental_operation

    function test_hlfir_associate_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: associate_op
        type(mlir_type_t) :: i32_type, expr_type, var_type
        type(mlir_value_t) :: expr_val
        type(mlir_region_t) :: body_region
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_hlfir_dialect(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        expr_type = create_hlfir_expr_type(context, i32_type, 0)
        var_type = create_hlfir_var_type(context, i32_type)
        
        ! Create dummy expression value
        expr_val = create_dummy_value(context)
        
        ! Create empty region for associate body
        body_region = create_empty_region(context)
        
        ! Create hlfir.associate operation
        associate_op = create_hlfir_associate(context, expr_val, var_type, body_region)
        passed = passed .and. associate_op%is_valid()
        passed = passed .and. verify_operation(associate_op)
        
        if (passed) then
            print *, "PASS: test_hlfir_associate_operation"
        else
            print *, "FAIL: test_hlfir_associate_operation"
        end if
        
        ! Cleanup
        call destroy_operation(associate_op)
        call destroy_mlir_context(context)
    end function test_hlfir_associate_operation

    function test_hlfir_end_associate_operation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_operation_t) :: end_associate_op
        type(mlir_value_t) :: var_val
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_hlfir_dialect(context)
        
        ! Create dummy variable value
        var_val = create_dummy_value(context)
        
        ! Create hlfir.end_associate operation
        end_associate_op = create_hlfir_end_associate(context, var_val)
        passed = passed .and. end_associate_op%is_valid()
        passed = passed .and. verify_operation(end_associate_op)
        
        if (passed) then
            print *, "PASS: test_hlfir_end_associate_operation"
        else
            print *, "FAIL: test_hlfir_end_associate_operation"
        end if
        
        ! Cleanup
        call destroy_operation(end_associate_op)
        call destroy_mlir_context(context)
    end function test_hlfir_end_associate_operation

end program test_hlfir_dialect