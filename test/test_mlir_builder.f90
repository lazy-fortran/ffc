program test_mlir_builder
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_builder
    use test_helpers
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR Builder Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - RED phase: Some should fail
    if (.not. test_builder_creation_and_cleanup()) all_tests_passed = .false.
    if (.not. test_insertion_point_management()) all_tests_passed = .false.
    if (.not. test_block_creation()) all_tests_passed = .false.
    if (.not. test_region_handling()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR builder tests passed!"
        stop 0
    else
        print *, "Some MLIR builder tests failed!"
        stop 1
    end if

contains

    function test_builder_creation_and_cleanup() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_module_t) :: module
        
        passed = .true.
        
        ! Create context and module
        context = create_mlir_context()
        module = create_empty_module(create_unknown_location(context))
        
        ! Test builder creation
        builder = create_mlir_builder(context)
        passed = passed .and. builder%is_valid()
        
        ! Test builder with module
        call builder%set_module(module)
        passed = passed .and. builder%has_module()
        
        ! Test builder state
        passed = passed .and. .not. builder%has_insertion_point()
        
        ! Test cleanup
        call destroy_mlir_builder(builder)
        passed = passed .and. .not. builder%is_valid()
        
        if (passed) then
            print *, "PASS: test_builder_creation_and_cleanup"
        else
            print *, "FAIL: test_builder_creation_and_cleanup"
        end if
        
        ! Cleanup
        call destroy_mlir_context(context)
    end function test_builder_creation_and_cleanup

    function test_insertion_point_management() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_block_t) :: block1, block2
        type(mlir_operation_t) :: op
        
        passed = .true.
        
        ! Create context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create blocks
        block1 = create_mlir_block()
        block2 = create_mlir_block()
        
        ! Test setting insertion point to block
        call builder%set_insertion_point_to_start(block1)
        passed = passed .and. builder%has_insertion_point()
        block2 = builder%get_insertion_block()
        passed = passed .and. block2%equals(block1)
        
        ! Test insertion point stack
        call builder%push_insertion_point()
        call builder%set_insertion_point_to_start(block2)
        block1 = builder%get_insertion_block()
        passed = passed .and. block1%equals(block2)
        
        call builder%pop_insertion_point()
        block1 = builder%get_insertion_block()
        block2 = create_mlir_block()
        call builder%set_insertion_point_to_start(block2)
        block1 = builder%get_insertion_block()
        passed = passed .and. block1%equals(block2)
        
        ! Test setting insertion point after operation
        op = create_dummy_operation(context)
        call builder%set_insertion_point_after(op)
        passed = passed .and. builder%has_insertion_point()
        
        if (passed) then
            print *, "PASS: test_insertion_point_management"
        else
            print *, "FAIL: test_insertion_point_management"
        end if
        
        ! Cleanup
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_insertion_point_management

    function test_block_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_region_t) :: region
        type(mlir_block_t) :: block, entry_block
        type(mlir_type_t), dimension(2) :: arg_types
        type(mlir_value_t), dimension(:), allocatable :: block_args
        
        passed = .true.
        
        ! Create context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create region
        region = create_empty_region(context)
        
        ! Test creating block without arguments
        block = builder%create_block(region)
        passed = passed .and. block%is_valid()
        passed = passed .and. (block_get_num_arguments(block) == 0)
        
        ! Test creating block with arguments
        arg_types(1) = create_integer_type(context, 32)
        arg_types(2) = create_float_type(context, 64)
        
        entry_block = builder%create_block_with_args(region, arg_types)
        passed = passed .and. entry_block%is_valid()
        passed = passed .and. (block_get_num_arguments(entry_block) == 2)
        
        ! Test getting block arguments
        block_args = block_get_arguments(entry_block)
        passed = passed .and. allocated(block_args)
        passed = passed .and. (size(block_args) == 2)
        
        if (passed) then
            print *, "PASS: test_block_creation"
        else
            print *, "FAIL: test_block_creation"
        end if
        
        ! Cleanup
        if (allocated(block_args)) deallocate(block_args)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_block_creation

    function test_region_handling() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        type(mlir_region_t) :: region
        type(mlir_block_t) :: block
        type(mlir_operation_t) :: parent_op
        type(builder_scope_t) :: scope
        
        passed = .true.
        
        ! Create context and builder
        context = create_mlir_context()
        builder = create_mlir_builder(context)
        
        ! Create parent operation with region
        parent_op = create_operation_with_region(context, "test.region_op")
        region = parent_op%get_region(0)
        
        ! Test entering region scope
        scope = builder%enter_region(region)
        passed = passed .and. scope%is_valid()
        
        ! Create block in region
        block = builder%create_block(region)
        call builder%set_insertion_point_to_start(block)
        passed = passed .and. builder%has_insertion_point()
        
        ! Test scope cleanup
        call builder%exit_scope(scope)
        passed = passed .and. .not. builder%has_insertion_point()
        
        ! Test automatic scope management
        call builder%with_region(region)
        block = builder%get_or_create_entry_block(region)
        passed = passed .and. block%is_valid()
        passed = passed .and. region_has_blocks(region)
        
        if (passed) then
            print *, "PASS: test_region_handling"
        else
            print *, "FAIL: test_region_handling"
        end if
        
        ! Cleanup
        call destroy_operation(parent_op)
        call destroy_mlir_builder(builder)
        call destroy_mlir_context(context)
    end function test_region_handling

end program test_mlir_builder