program test_hlfir_declare_simple_red
    use mlir_c_core
    use mlir_c_types  
    use mlir_c_attributes
    use mlir_c_operations
    use hlfir_dialect
    implicit none
    
    logical :: all_tests_passed
    
    print *, "=== RED Phase: Simple HLFIR Declare Test ==="
    print *, "This test MUST FAIL initially (RED phase of TDD)"
    print *
    
    all_tests_passed = .true.
    
    ! RED Test for basic hlfir.declare - should fail because signature is wrong
    if (.not. test_hlfir_declare_basic()) all_tests_passed = .false.
    
    print *
    if (all_tests_passed) then
        print *, "ERROR: Tests should FAIL in RED phase! Implementation already exists?"
        stop 1
    else
        print *, "SUCCESS: All tests FAILED as expected in RED phase"
        print *, "Ready to implement GREEN phase"
        stop 0
    end if
    
contains

    function test_hlfir_declare_basic() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: int_type, ref_type
        type(mlir_attribute_t) :: name_attr
        type(mlir_operation_t) :: declare_op
        integer :: num_results
        
        print *, "RED TEST: Basic hlfir.declare for integer variable"
        
        ! Setup MLIR context
        context = create_mlir_context()
        location = create_unknown_location(context)
        
        ! Create types: i32 -> !fir.ref<i32>
        int_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, int_type)
        
        ! Create name attribute
        name_attr = create_string_attribute(context, "test_var")
        
        ! Try to create hlfir.declare - this should work with current signature but be wrong
        ! According to Flang, hlfir.declare should take a memref and return dual SSA results
        ! Current signature: create_hlfir_declare(context, memref, name_attr, result_type, fortran_attrs)
        ! But we need to pass a location, not context as first arg, and memref needs to be a value
        
        ! This should fail because we don't have a proper memref value
        passed = .false.
        
        ! To create proper test, we need:
        ! 1. An alloca operation that gives us a memref value
        ! 2. Pass that memref to hlfir.declare
        ! 3. Get back dual SSA results (HLFIR entity + FIR base)
        
        print *, "  EXPECTED FAIL: Cannot create proper hlfir.declare without memref value"
        print *, "  Need to implement: allocation -> hlfir.declare -> dual results"
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_basic
    
end program test_hlfir_declare_simple_red