program test_hlfir_declare_red
    use mlir_c_core
    use mlir_c_types  
    use mlir_c_attributes
    use mlir_c_operations
    use hlfir_dialect
    implicit none
    
    logical :: all_tests_passed
    
    print *, "=== RED Phase: HLFIR Declare Operation Tests ==="
    print *, "These tests MUST FAIL initially (RED phase of TDD)"
    print *
    
    all_tests_passed = .true.
    
    ! RED Tests for hlfir.declare following Flang patterns
    if (.not. test_hlfir_declare_integer_variable()) all_tests_passed = .false.
    if (.not. test_hlfir_declare_array_variable()) all_tests_passed = .false.
    if (.not. test_hlfir_declare_character_variable()) all_tests_passed = .false.
    if (.not. test_hlfir_declare_dual_ssa_results()) all_tests_passed = .false.
    if (.not. test_hlfir_declare_shape_attributes()) all_tests_passed = .false.
    
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

    function test_hlfir_declare_integer_variable() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: int_type, ref_type
        type(mlir_attribute_t) :: name_attr
        type(mlir_operation_t) :: declare_op
        integer :: num_results
        
        print *, "RED TEST: hlfir.declare for integer variable"
        
        ! Setup MLIR context
        context = create_mlir_context()
        location = create_unknown_location(context)
        
        ! Create types: !fir.ref<!fir.int<4>>
        int_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, int_type)
        
        ! Create name attribute
        name_attr = create_string_attribute(context, "test_var")
        
        ! Try to create hlfir.declare operation (should fail - not implemented yet)
        declare_op = create_hlfir_declare(context, location, ref_type, name_attr)
        
        ! Check if operation was created successfully
        passed = declare_op%is_valid()
        
        if (passed) then
            ! Additional checks for dual SSA results (HLFIR + FIR base)
            num_results = get_operation_num_results(declare_op)
            passed = (num_results == 2)  ! Should return both hlfir and fir.ref results
        end if
        
        if (passed) then
            print *, "  UNEXPECTED PASS: hlfir.declare created (should fail in RED phase)"
        else
            print *, "  EXPECTED FAIL: hlfir.declare not implemented yet"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_integer_variable
    
    function test_hlfir_declare_array_variable() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: int_type, array_type, ref_type
        type(mlir_attribute_t) :: name_attr, shape_attr
        type(mlir_operation_t) :: declare_op
        
        print *, "RED TEST: hlfir.declare for array variable"
        
        ! Setup MLIR context
        context = create_mlir_context()
        location = create_unknown_location(context)
        
        ! Create types: !fir.ref<!fir.array<10x!fir.int<4>>>
        int_type = create_integer_type(context, 32)
        array_type = create_array_type(context, int_type, [10])
        ref_type = create_reference_type(context, array_type)
        
        ! Create attributes
        name_attr = create_string_attribute(context, "test_array")
        shape_attr = create_shape_attribute(context, [10])
        
        ! Try to create hlfir.declare with shape (should fail)
        declare_op = create_hlfir_declare_with_shape(context, location, ref_type, &
                                                   name_attr, shape_attr)
        
        passed = declare_op%is_valid()
        
        if (passed) then
            print *, "  UNEXPECTED PASS: hlfir.declare with shape created (should fail in RED phase)"
        else
            print *, "  EXPECTED FAIL: hlfir.declare with shape not implemented yet"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_array_variable
    
    function test_hlfir_declare_character_variable() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: char_type, ref_type
        type(mlir_attribute_t) :: name_attr, len_attr
        type(mlir_operation_t) :: declare_op
        
        print *, "RED TEST: hlfir.declare for character variable"
        
        ! Setup MLIR context
        context = create_mlir_context()
        location = create_unknown_location(context)
        
        ! Create types: !fir.ref<!fir.char<1,10>>
        char_type = create_character_type(context, 1, 10)
        ref_type = create_reference_type(context, char_type)
        
        ! Create attributes
        name_attr = create_string_attribute(context, "test_string")
        len_attr = create_integer_attribute(context, create_integer_type(context, 32), 10_8)
        
        ! Try to create hlfir.declare with character length (should fail)
        declare_op = create_hlfir_declare_with_length(context, location, ref_type, &
                                                    name_attr, len_attr)
        
        passed = declare_op%is_valid()
        
        if (passed) then
            print *, "  UNEXPECTED PASS: hlfir.declare with char length created (should fail in RED phase)"
        else
            print *, "  EXPECTED FAIL: hlfir.declare with char length not implemented yet"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_character_variable
    
    function test_hlfir_declare_dual_ssa_results() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: int_type, ref_type
        type(mlir_attribute_t) :: name_attr
        type(mlir_operation_t) :: declare_op
        type(mlir_value_t) :: hlfir_result, fir_result
        
        print *, "RED TEST: hlfir.declare dual SSA results (HLFIR + FIR base)"
        
        ! Setup MLIR context
        context = create_mlir_context()
        location = create_unknown_location(context)
        
        ! Create types
        int_type = create_integer_type(context, 32)
        ref_type = create_reference_type(context, int_type)
        name_attr = create_string_attribute(context, "test_var")
        
        ! Try to create hlfir.declare and extract dual results (should fail)
        declare_op = create_hlfir_declare(context, location, ref_type, name_attr)
        
        if (declare_op%is_valid()) then
            ! Try to get both results
            hlfir_result = get_operation_result(declare_op, 0)  ! HLFIR entity
            fir_result = get_operation_result(declare_op, 1)    ! FIR base address
            
            passed = hlfir_result%is_valid() .and. fir_result%is_valid()
        else
            passed = .false.
        end if
        
        if (passed) then
            print *, "  UNEXPECTED PASS: dual SSA results working (should fail in RED phase)"
        else
            print *, "  EXPECTED FAIL: dual SSA results not implemented yet"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_dual_ssa_results
    
    function test_hlfir_declare_shape_attributes() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        type(mlir_type_t) :: int_type, array_type, ref_type
        type(mlir_attribute_t) :: name_attr, shape_attr, type_params_attr
        type(mlir_operation_t) :: declare_op
        
        print *, "RED TEST: hlfir.declare with Fortran variable semantics (shape, type parameters)"
        
        ! Setup MLIR context
        context = create_mlir_context()
        location = create_unknown_location(context)
        
        ! Create types for assumed-shape array: !fir.ref<!fir.box<!fir.array<?x!fir.int<4>>>>
        int_type = create_integer_type(context, 32) 
        array_type = create_assumed_shape_array_type(context, int_type, 1)
        ref_type = create_reference_type(context, array_type)
        
        ! Create full Fortran variable semantics attributes
        name_attr = create_string_attribute(context, "assumed_array")
        shape_attr = create_assumed_shape_attribute(context, 1)
        type_params_attr = create_empty_array_attribute(context)
        
        ! Try to create hlfir.declare with full Fortran semantics (should fail)
        declare_op = create_hlfir_declare_with_fortran_semantics(context, location, ref_type, &
                                                               name_attr, shape_attr, type_params_attr)
        
        passed = declare_op%is_valid()
        
        if (passed) then
            print *, "  UNEXPECTED PASS: hlfir.declare with Fortran semantics created (should fail in RED phase)"
        else
            print *, "  EXPECTED FAIL: hlfir.declare with Fortran semantics not implemented yet"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_hlfir_declare_shape_attributes
    
end program test_hlfir_declare_red