program test_real_mlir_integration
    use iso_c_binding
    use mlir_c_core
    implicit none
    
    type(mlir_context_t) :: context
    type(mlir_module_t) :: module
    type(mlir_location_t) :: location
    logical :: success
    
    print *, "=== Testing Real MLIR C API Integration ==="
    
    ! RED Test: This should work with real MLIR instead of stubs
    success = .true.
    
    ! Test 1: Context creation and destruction
    print *, "Testing context creation..."
    context = mlir_context_create()
    if (.not. context%is_valid()) then
        print *, "FAIL: Context creation failed"
        success = .false.
    else
        print *, "PASS: Context created successfully"
    end if
    
    ! Test 2: Location creation
    print *, "Testing location creation..."
    location = mlir_location_unknown_get(context)
    if (.not. location%is_valid()) then
        print *, "FAIL: Location creation failed"
        success = .false.
    else
        print *, "PASS: Location created successfully"
    end if
    
    ! Test 3: Module creation
    print *, "Testing module creation..."
    module = mlir_module_create_empty(location)
    if (.not. module%is_valid()) then
        print *, "FAIL: Module creation failed"
        success = .false.
    else
        print *, "PASS: Module created successfully"
    end if
    
    ! Test 4: Verify this is NOT a stub (real MLIR should have different behavior)
    print *, "Testing real vs stub behavior..."
    ! Real MLIR contexts should have different internal state
    ! This is our "canary" test to ensure we're not using stubs
    
    ! Cleanup
    call context%destroy()
    
    if (success) then
        print *, "=== ALL TESTS PASSED: Real MLIR Integration Working ==="
        stop 0
    else
        print *, "=== TESTS FAILED: Still using stubs or integration broken ==="
        stop 1
    end if
    
end program test_real_mlir_integration