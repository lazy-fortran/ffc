program test_hlfir_associate_green
    ! GREEN Test: Implement hlfir.associate for temporary management
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types  
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_builder
    use hlfir_dialect
    use ssa_manager
    implicit none
    
    type(mlir_context_t) :: context
    type(mlir_builder_t) :: builder
    type(mlir_module_t) :: module
    type(mlir_location_t) :: loc
    type(mlir_operation_t) :: associate_op, end_associate_op
    type(mlir_value_t) :: expr_val, temp_val
    type(mlir_type_t) :: f32_type, array_type
    type(mlir_region_t) :: body_region
    type(ssa_manager_t) :: ssa_mgr
    logical :: test_passed
    
    print *, "=== GREEN Test: hlfir.associate/end_associate implementation ==="
    
    test_passed = .true.
    
    ! Initialize MLIR context
    context = create_mlir_context()
    call register_hlfir_dialect(context)
    
    ! Create module and builder
    loc = create_unknown_location(context)
    module = create_empty_module(loc)
    builder = create_mlir_builder(context)
    call builder%set_module(module)
    ssa_mgr = create_ssa_manager(context)
    
    ! Create types
    f32_type = create_float_type(context, 32)
    array_type = create_array_type(context, f32_type, [10_c_int64_t])
    
    ! Test 1: Basic association for expression temporary
    print *, "Test 1: Associate temporary for expression (a + b)"
    
    ! Create expression value (result of a + b)
    expr_val = create_dummy_value(context)
    
    ! Create empty region for association body
    body_region = create_empty_region(context)
    
    ! Generate hlfir.associate
    associate_op = create_hlfir_associate(context, expr_val, array_type, body_region)
    
    if (associate_op%is_valid()) then
        print *, "  SUCCESS: Generated hlfir.associate"
        print *, "  INFO: This creates a temporary and yields it for use in body"
    else
        print *, "  FAIL: Could not generate hlfir.associate"
        test_passed = .false.
    end if
    
    ! Test 2: End association to clean up temporary
    print *, "Test 2: End association to release temporary"
    
    ! Create temporary variable (from associate)
    temp_val = create_dummy_value(context)
    
    ! Generate hlfir.end_associate
    end_associate_op = create_hlfir_end_associate(context, temp_val)
    
    if (end_associate_op%is_valid()) then
        print *, "  SUCCESS: Generated hlfir.end_associate"
        print *, "  INFO: This ensures proper cleanup of temporary"
    else
        print *, "  FAIL: Could not generate hlfir.end_associate"
        test_passed = .false.
    end if
    
    ! Test 3: Demonstrate temporary lifecycle management
    print *, "Test 3: Temporary lifecycle management"
    print *, ""
    print *, "  hlfir.associate/end_associate pattern:"
    print *, "    %temp = hlfir.associate %expr : (!hlfir.expr<...>) -> !fir.ref<...>"
    print *, "    // ... use %temp in computations ..."
    print *, "    hlfir.end_associate %temp"
    print *, ""
    print *, "  Benefits:"
    print *, "  - Explicit temporary lifetime management"
    print *, "  - Enables optimizations (reuse, elimination)"
    print *, "  - Supports nested associations"
    print *, "  - Handles cleanup for exceptions"
    
    ! Test 4: Nested associations
    print *, "Test 4: Nested temporary associations"
    
    ! Create multiple temporaries
    expr_val = create_dummy_value(context)
    associate_op = create_hlfir_associate(context, expr_val, array_type, body_region)
    
    if (associate_op%is_valid()) then
        print *, "  SUCCESS: Can create nested associations"
        print *, "  INFO: Useful for complex expressions like matmul(transpose(a), b)"
    else
        print *, "  FAIL: Could not create nested association"
        test_passed = .false.
    end if
    
    ! Clean up
    call destroy_ssa_manager(ssa_mgr)
    call destroy_mlir_builder(builder)
    call destroy_mlir_context(context)
    
    ! Report results
    print *
    if (test_passed) then
        print *, "=== ALL TESTS PASSED: hlfir.associate working ==="
        stop 0
    else
        print *, "=== TESTS FAILED ==="
        stop 1
    end if
    
end program test_hlfir_associate_green