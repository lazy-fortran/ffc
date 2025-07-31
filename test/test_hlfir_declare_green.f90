program test_hlfir_declare_green
    ! GREEN Phase Test for hlfir.declare implementation
    ! This test should PASS after implementing the functionality
    
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use mlir_c_operation_builder
    implicit none
    
    logical :: all_tests_passed
    
    print *, "=== GREEN Test: hlfir.declare implementation ==="
    print *, "Expected: This test should PASS after implementation"
    print *
    
    all_tests_passed = .true.
    
    ! Test individual components
    if (.not. test_fir_alloca_creation()) all_tests_passed = .false.
    if (.not. test_hlfir_declare_creation()) all_tests_passed = .false.
    if (.not. test_dual_ssa_results()) all_tests_passed = .false.
    
    print *
    if (all_tests_passed) then
        print *, "SUCCESS: All GREEN phase tests passed!"
        print *, "hlfir.declare implementation complete"
        stop 0
    else
        print *, "FAILURE: GREEN phase tests failed"
        print *, "Implementation incomplete"
        stop 1
    end if
    
contains

    function test_fir_alloca_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: i32_type, ref_type
        type(operation_builder_t) :: builder
        type(mlir_operation_t) :: alloca_op
        
        print *, "TEST: Creating fir.alloca operation"
        
        ! Setup
        context = create_mlir_context()
        location = create_unknown_location(context)
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! Try to create fir.alloca operation
        ! %alloca = fir.alloca !fir.ref<!fir.int<32>>
        call builder%init(context, "fir.alloca")
        call builder%result(ref_type)
        alloca_op = builder%build()
        
        passed = alloca_op%is_valid()
        
        if (passed) then
            print *, "  PASS: fir.alloca operation created successfully"
        else
            print *, "  FAIL: fir.alloca operation creation failed"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_fir_alloca_creation
    
    function test_hlfir_declare_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: i32_type, ref_type
        type(operation_builder_t) :: alloca_builder, declare_builder
        type(mlir_operation_t) :: alloca_op, declare_op
        type(mlir_value_t) :: alloca_result
        
        print *, "TEST: Creating hlfir.declare operation"
        
        ! Setup
        context = create_mlir_context()
        location = create_unknown_location(context)
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! First create fir.alloca to get a memory reference
        call alloca_builder%init(context, "fir.alloca")
        call alloca_builder%result(ref_type)
        alloca_op = alloca_builder%build()
        
        if (.not. alloca_op%is_valid()) then
            print *, "  FAIL: Cannot create alloca for testing hlfir.declare"
            passed = .false.
            call destroy_mlir_context(context)
            return
        end if
        
        ! Get the result from allocas (this is the memory reference we pass to hlfir.declare)
        alloca_result = get_operation_result(alloca_op, 0)
        
        ! Now create hlfir.declare
        ! %var:2 = hlfir.declare %alloca {var_name="x"} : (!fir.ref<!fir.int<32>>) -> (!hlfir.expr<!fir.int<32>>, !fir.ref<!fir.int<32>>)
        call declare_builder%init(context, "hlfir.declare")
        call declare_builder%operand(alloca_result)
        call declare_builder%result(i32_type)  ! HLFIR entity result
        call declare_builder%result(ref_type)  ! FIR reference result
        declare_op = declare_builder%build()
        
        passed = declare_op%is_valid()
        
        if (passed) then
            print *, "  PASS: hlfir.declare operation created successfully"
        else
            print *, "  FAIL: hlfir.declare operation creation failed"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_creation
    
    function test_dual_ssa_results() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: i32_type, ref_type
        type(operation_builder_t) :: alloca_builder, declare_builder
        type(mlir_operation_t) :: alloca_op, declare_op
        type(mlir_value_t) :: alloca_result, hlfir_entity, fir_ref
        integer :: num_results
        
        print *, "TEST: hlfir.declare dual SSA results"
        
        ! Setup (same as previous test)
        context = create_mlir_context()
        location = create_unknown_location(context)
        i32_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, i32_type)
        
        ! Create alloca
        call alloca_builder%init(context, "fir.alloca")
        call alloca_builder%result(ref_type)
        alloca_op = alloca_builder%build()
        
        if (.not. alloca_op%is_valid()) then
            passed = .false.
            call destroy_mlir_context(context)
            return
        end if
        
        alloca_result = get_operation_result(alloca_op, 0)
        
        ! Create hlfir.declare with dual results
        call declare_builder%init(context, "hlfir.declare")
        call declare_builder%operand(alloca_result)
        call declare_builder%result(i32_type)  ! HLFIR entity
        call declare_builder%result(ref_type)  ! FIR reference
        declare_op = declare_builder%build()
        
        if (.not. declare_op%is_valid()) then
            print *, "  FAIL: hlfir.declare creation failed"
            passed = .false.
            call destroy_mlir_context(context)
            return
        end if
        
        ! Check that we have exactly 2 results
        num_results = get_operation_num_results(declare_op)
        if (num_results /= 2) then
            print *, "  FAIL: Expected 2 results, got ", num_results
            passed = .false.
            call destroy_mlir_context(context)
            return
        end if
        
        ! Extract both results
        hlfir_entity = get_operation_result(declare_op, 0)  ! HLFIR entity
        fir_ref = get_operation_result(declare_op, 1)       ! FIR reference
        
        passed = hlfir_entity%is_valid() .and. fir_ref%is_valid()
        
        if (passed) then
            print *, "  PASS: hlfir.declare returns dual SSA values correctly"
        else
            print *, "  FAIL: hlfir.declare dual SSA values invalid"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_dual_ssa_results

end program test_hlfir_declare_green