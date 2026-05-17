program test_ssa_manager
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use ssa_manager
    use test_helpers
    implicit none

    logical :: all_tests_passed

    print *, "=== SSA Manager Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_ssa_value_generation()) all_tests_passed = .false.
    if (.not. test_value_naming()) all_tests_passed = .false.
    if (.not. test_value_type_tracking()) all_tests_passed = .false.
    if (.not. test_use_def_chains()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All SSA manager tests passed!"
        stop 0
    else
        print *, "Some SSA manager tests failed!"
        stop 1
    end if

contains

    function test_ssa_value_generation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(ssa_manager_t) :: manager
        type(mlir_type_t) :: i32_type, f64_type
        type(ssa_value_info_t) :: val1, val2, val3
        
        passed = .true.
        
        ! Create context and manager
        context = create_mlir_context()
        manager = create_ssa_manager(context)
        passed = passed .and. manager%is_valid()
        
        ! Create some types
        i32_type = create_integer_type(context, 32)
        f64_type = create_float_type(context, 64)
        
        ! Test generating SSA values
        val1 = manager%generate_value(i32_type)
        passed = passed .and. val1%is_valid()
        passed = passed .and. (val1%get_id() == 1)
        
        val2 = manager%generate_value(f64_type) 
        passed = passed .and. val2%is_valid()
        passed = passed .and. (val2%get_id() == 2)
        
        val3 = manager%generate_value(i32_type)
        passed = passed .and. val3%is_valid()
        passed = passed .and. (val3%get_id() == 3)
        
        ! Test that values are unique
        passed = passed .and. (.not. val1%equals(val2))
        passed = passed .and. (.not. val1%equals(val3))
        passed = passed .and. (.not. val2%equals(val3))
        
        if (passed) then
            print *, "PASS: test_ssa_value_generation"
        else
            print *, "FAIL: test_ssa_value_generation"
        end if
        
        ! Cleanup
        call destroy_ssa_manager(manager)
        call destroy_mlir_context(context)
    end function test_ssa_value_generation

    function test_value_naming() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(ssa_manager_t) :: manager
        type(mlir_type_t) :: i32_type
        type(ssa_value_info_t) :: val1, val2, val3
        character(len=:), allocatable :: name
        
        passed = .true.
        
        ! Create context and manager
        context = create_mlir_context()
        manager = create_ssa_manager(context)
        i32_type = create_integer_type(context, 32)
        
        ! Test automatic naming
        val1 = manager%generate_value(i32_type)
        name = val1%get_name()
        passed = passed .and. (name == "%0")
        
        val2 = manager%generate_value(i32_type)
        name = val2%get_name()
        passed = passed .and. (name == "%1")
        
        ! Test custom naming
        val3 = manager%generate_named_value("temp", i32_type)
        name = val3%get_name()
        passed = passed .and. (name == "%temp")
        
        ! Test name lookup
        passed = passed .and. manager%has_value("temp")
        passed = passed .and. (.not. manager%has_value("nonexistent"))
        
        val1 = manager%get_value_by_name("temp")
        passed = passed .and. val1%is_valid()
        passed = passed .and. val1%equals(val3)
        
        if (passed) then
            print *, "PASS: test_value_naming"
        else
            print *, "FAIL: test_value_naming"
        end if
        
        ! Cleanup
        call destroy_ssa_manager(manager)
        call destroy_mlir_context(context)
    end function test_value_naming

    function test_value_type_tracking() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(ssa_manager_t) :: manager
        type(mlir_type_t) :: i32_type, f64_type, i1_type
        type(ssa_value_info_t) :: int_val, float_val, bool_val
        type(mlir_type_t) :: retrieved_type
        
        passed = .true.
        
        ! Create context and manager
        context = create_mlir_context()
        manager = create_ssa_manager(context)
        
        ! Create different types
        i32_type = create_integer_type(context, 32)
        f64_type = create_float_type(context, 64)
        i1_type = create_integer_type(context, 1)
        
        ! Generate values with different types
        int_val = manager%generate_named_value("int_var", i32_type)
        float_val = manager%generate_named_value("float_var", f64_type)
        bool_val = manager%generate_named_value("bool_var", i1_type)
        
        ! Test type retrieval
        retrieved_type = int_val%get_type()
        passed = passed .and. manager%types_equal(retrieved_type, i32_type)
        
        retrieved_type = float_val%get_type()
        passed = passed .and. manager%types_equal(retrieved_type, f64_type)
        
        retrieved_type = bool_val%get_type()
        passed = passed .and. manager%types_equal(retrieved_type, i1_type)
        
        ! Test type queries
        passed = passed .and. manager%is_integer_type(int_val)
        passed = passed .and. (.not. manager%is_integer_type(float_val))
        passed = passed .and. manager%is_float_type(float_val)
        passed = passed .and. (.not. manager%is_float_type(int_val))
        
        if (passed) then
            print *, "PASS: test_value_type_tracking"
        else
            print *, "FAIL: test_value_type_tracking" 
        end if
        
        ! Cleanup
        call destroy_ssa_manager(manager)
        call destroy_mlir_context(context)
    end function test_value_type_tracking

    function test_use_def_chains() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(ssa_manager_t) :: manager
        type(mlir_type_t) :: i32_type
        type(ssa_value_info_t) :: def_val, use_val1, use_val2
        type(mlir_operation_t) :: def_op, use_op1, use_op2
        integer, allocatable :: use_list(:)
        
        passed = .true.
        
        ! Create context and manager
        context = create_mlir_context()
        manager = create_ssa_manager(context)
        i32_type = create_integer_type(context, 32)
        
        ! Create a definition (value produced by an operation)
        def_op = create_dummy_operation(context)
        def_val = manager%generate_named_value("def", i32_type)
        call manager%set_defining_op(def_val, def_op)
        
        ! Test definition tracking
        def_op = manager%get_defining_op(def_val)
        passed = passed .and. def_op%is_valid()
        
        ! Create uses (operations that consume the value)
        use_op1 = create_dummy_operation(context)
        use_op2 = create_dummy_operation(context)
        
        call manager%add_use(def_val, use_op1)
        call manager%add_use(def_val, use_op2)
        
        ! Test use counting
        passed = passed .and. (manager%get_use_count(def_val) == 2)
        
        ! Test use enumeration
        use_list = manager%get_uses(def_val)
        passed = passed .and. allocated(use_list)
        passed = passed .and. (size(use_list) == 2)
        
        ! Test use removal
        call manager%remove_use(def_val, use_op1)
        passed = passed .and. (manager%get_use_count(def_val) == 1)
        
        ! Test dead value detection
        call manager%remove_use(def_val, use_op2)
        passed = passed .and. (manager%get_use_count(def_val) == 0)
        passed = passed .and. manager%is_dead_value(def_val)
        
        if (passed) then
            print *, "PASS: test_use_def_chains"
        else
            print *, "FAIL: test_use_def_chains"
        end if
        
        ! Cleanup
        if (allocated(use_list)) deallocate(use_list)
        call destroy_ssa_manager(manager)
        call destroy_mlir_context(context)
    end function test_use_def_chains

end program test_ssa_manager