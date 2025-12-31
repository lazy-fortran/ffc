program test_fir_dialect_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use fir_dialect
    use fir_dialect_builder
    implicit none

    logical :: all_tests_passed

    print *, "=== FIR Dialect Builder Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_fir_builder_templates()) all_tests_passed = .false.
    if (.not. test_fir_type_builders()) all_tests_passed = .false.
    if (.not. test_fir_composite_operations()) all_tests_passed = .false.
    if (.not. test_fir_builder_validation()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All FIR dialect builder tests passed!"
        stop 0
    else
        print *, "Some FIR dialect builder tests failed!"
        stop 1
    end if

contains

    function test_fir_builder_templates() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(fir_builder_t) :: builder
        type(mlir_operation_t) :: op
        type(mlir_type_t) :: i32_type, ref_type
        type(mlir_value_t) :: memref
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Initialize builder
        call builder%init(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        ref_type = create_fir_reference_type(context, i32_type)
        
        ! Create dummy memref
        memref = create_dummy_value(context)
        
        ! Test load template
        op = builder%build_load(memref, i32_type)
        passed = passed .and. op%is_valid()
        call destroy_operation(op)
        
        ! Test store template
        op = builder%build_store(memref, memref)
        passed = passed .and. op%is_valid()
        call destroy_operation(op)
        
        if (passed) then
            print *, "PASS: test_fir_builder_templates"
        else
            print *, "FAIL: test_fir_builder_templates"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_fir_builder_templates

    function test_fir_type_builders() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_t) :: base_type, ref_type, array_type, box_type
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Create base type
        base_type = create_integer_type(context, 32)
        
        ! Test FIR reference type
        ref_type = create_fir_reference_type(context, base_type)
        passed = passed .and. ref_type%is_valid()
        
        ! Test FIR array type
        array_type = create_fir_array_type(context, base_type, [10, 20])
        passed = passed .and. array_type%is_valid()
        
        ! Test FIR box type
        box_type = create_fir_box_type(context, base_type)
        passed = passed .and. box_type%is_valid()
        
        if (passed) then
            print *, "PASS: test_fir_type_builders"
        else
            print *, "FAIL: test_fir_type_builders"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_fir_type_builders

    function test_fir_composite_operations() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(fir_builder_t) :: builder
        type(mlir_operation_t) :: alloca_op, declare_op
        type(mlir_type_t) :: i32_type, ref_type
        type(mlir_value_t) :: memref
        character(len=32) :: var_name
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Initialize builder
        call builder%init(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        ref_type = create_fir_reference_type(context, i32_type)
        
        ! Test composite operation: alloca + declare
        var_name = "local_var"
        call builder%build_local_variable(var_name, i32_type, alloca_op, declare_op)
        
        passed = passed .and. alloca_op%is_valid()
        passed = passed .and. declare_op%is_valid()
        
        if (passed) then
            print *, "PASS: test_fir_composite_operations"
        else
            print *, "FAIL: test_fir_composite_operations"
        end if
        
        ! Cleanup
        call destroy_operation(alloca_op)
        call destroy_operation(declare_op)
        call destroy_mlir_context(context)
    end function test_fir_composite_operations

    function test_fir_builder_validation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(fir_builder_t) :: builder
        type(mlir_operation_t) :: op
        type(mlir_type_t) :: i32_type, f64_type, ref_i32, ref_f64
        type(mlir_value_t) :: i32_ref, f64_value
        
        passed = .true.
        
        ! Create context and register dialect
        context = create_mlir_context()
        call register_fir_dialect(context)
        
        ! Initialize builder
        call builder%init(context)
        
        ! Create types
        i32_type = create_integer_type(context, 32)
        f64_type = create_float_type(context, 64)
        ref_i32 = create_fir_reference_type(context, i32_type)
        ref_f64 = create_fir_reference_type(context, f64_type)
        
        ! Create dummy values
        i32_ref = create_dummy_value(context)
        f64_value = create_dummy_value(context)
        
        ! Test type validation in store operation
        ! This should validate that stored value type matches reference element type
        op = builder%build_store_validated(f64_value, i32_ref, f64_type, ref_i32)
        passed = passed .and. .not. op%is_valid()  ! Should fail validation
        
        if (passed) then
            print *, "PASS: test_fir_builder_validation"
        else
            print *, "FAIL: test_fir_builder_validation"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_fir_builder_validation

end program test_fir_dialect_builder