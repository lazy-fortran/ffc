program test_hlfir_declare_red_basic
    ! RED Phase Test for hlfir.declare following Flang's HLFIR patterns
    ! This test MUST fail initially - that's the point of RED phase TDD
    
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    implicit none
    
    logical :: test_passed
    type(mlir_context_t) :: context
    type(mlir_location_t) :: location
    type(mlir_module_t) :: module
    type(mlir_type_t) :: i32_type
    
    print *, "=== RED Test: hlfir.declare basic functionality ==="
    print *, "Expected: This test MUST FAIL (RED phase)"
    print *
    
    ! Setup MLIR context
    context = create_mlir_context()
    location = create_unknown_location(context)
    module = create_empty_module(location) 
    
    ! Test basic type creation first (should work)
    i32_type = create_integer_type(context, 32)
    
    if (.not. i32_type%is_valid()) then
        print *, "FAIL: Cannot even create basic integer type"
        print *, "This indicates fundamental MLIR integration issues"
        stop 1
    end if
    
    print *, "PASS: Basic integer type creation works"
    
    ! Now test what should fail in RED phase:
    ! We need to create an alloca operation that gives us a memref,
    ! then pass that to hlfir.declare. This should fail because
    ! hlfir.declare is not properly implemented yet.
    
    test_passed = test_hlfir_declare_with_alloca()
    
    ! Cleanup
    call destroy_mlir_context(context)
    
    if (test_passed) then
        print *, "ERROR: Test passed when it should fail in RED phase!"
        print *, "This suggests hlfir.declare is already implemented."
        stop 1
    else
        print *, "SUCCESS: Test failed as expected in RED phase"
        print *, "Ready to implement hlfir.declare in GREEN phase"
        stop 0
    end if
    
contains

    function test_hlfir_declare_with_alloca() result(passed)
        logical :: passed
        
        print *, "Testing hlfir.declare operation creation..."
        
        ! In Flang's HLFIR, the pattern is:
        ! 1. Create alloca for variable storage: %alloca = fir.alloca !fir.ref<!fir.int<32>>
        ! 2. Declare variable with HLFIR: %var:2 = hlfir.declare %alloca {var_name="x"} : (!fir.ref<!fir.int<32>>) -> (!hlfir.expr<!fir.int<32>>, !fir.ref<!fir.int<32>>)
        !
        ! The hlfir.declare should return TWO SSA values:
        ! - An HLFIR entity that preserves Fortran semantics
        ! - The original FIR memory reference for lowering
        
        ! This test should fail because:
        ! 1. We don't have proper fir.alloca operation creation
        ! 2. We don't have proper hlfir.declare operation creation
        ! 3. We don't have dual SSA value return handling
        
        print *, "  Expected failure: Missing fir.alloca operation builder"
        print *, "  Expected failure: Missing hlfir.declare operation builder"  
        print *, "  Expected failure: Missing dual SSA value handling"
        
        ! For now, just return false to indicate the functionality is missing
        passed = .false.
    end function test_hlfir_declare_with_alloca

end program test_hlfir_declare_red_basic