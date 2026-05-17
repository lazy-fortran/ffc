program test_mlir_c_builder
    use mlir_c_core
    use mlir_c_builder
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Builder Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_builder_creation()) all_tests_passed = .false.
    if (.not. test_builder_with_context()) all_tests_passed = .false.
    if (.not. test_builder_raii()) all_tests_passed = .false.
    if (.not. test_builder_getters()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API builder tests passed!"
        stop 0
    else
        print *, "Some MLIR C API builder tests failed!"
        stop 1
    end if

contains

    function test_builder_creation() result(passed)
        logical :: passed
        type(mlir_builder_t) :: builder
        
        ! Create builder using constructor
        builder = mlir_builder_t()
        
        ! Check that all components are valid
        block
            type(mlir_context_t) :: ctx
            type(mlir_module_t) :: mod
            type(mlir_location_t) :: loc
            
            ctx = builder%get_context()
            mod = builder%get_module()
            loc = builder%get_location()
            
            passed = ctx%is_valid() .and. mod%is_valid() .and. loc%is_valid()
        end block
        
        if (passed) then
            print *, "PASS: test_builder_creation"
        else
            print *, "FAIL: test_builder_creation - builder components not valid"
        end if
        
        ! Explicit cleanup
        call builder%finalize()
    end function test_builder_creation

    function test_builder_with_context() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_builder_t) :: builder
        
        ! Create context separately
        context = create_mlir_context()
        
        ! Create builder with existing context
        builder = mlir_builder_t(context)
        
        ! Check that builder module is valid (we can't directly compare contexts)
        block
            type(mlir_module_t) :: mod
            mod = builder%get_module()
            passed = mod%is_valid()
        end block
        
        if (passed) then
            print *, "PASS: test_builder_with_context"
        else
            print *, "FAIL: test_builder_with_context - context not shared"
        end if
        
        ! Explicit cleanup
        call builder%finalize()
        
        ! Context should still be valid
        passed = passed .and. context%is_valid()
        
        ! Clean up context
        call destroy_mlir_context(context)
    end function test_builder_with_context

    function test_builder_raii() result(passed)
        logical :: passed
        
        passed = .true.
        
        ! Test automatic cleanup in a block
        block
            type(mlir_builder_t) :: temp_builder
            temp_builder = mlir_builder_t()
            
            ! Builder should be valid in block
            block
                type(mlir_context_t) :: temp_ctx
                temp_ctx = temp_builder%get_context()
                passed = temp_ctx%is_valid()
            end block
        end block
        ! temp_builder should be automatically cleaned up
        
        if (passed) then
            print *, "PASS: test_builder_raii"
        else
            print *, "FAIL: test_builder_raii"
        end if
    end function test_builder_raii

    function test_builder_getters() result(passed)
        logical :: passed
        type(mlir_builder_t) :: builder
        type(mlir_context_t) :: ctx
        type(mlir_module_t) :: mod
        type(mlir_location_t) :: loc
        
        ! Create builder
        builder = mlir_builder_t()
        
        ! Get components
        ctx = builder%get_context()
        mod = builder%get_module()
        loc = builder%get_location()
        
        ! All should be valid
        passed = ctx%is_valid() .and. mod%is_valid() .and. loc%is_valid()
        
        if (passed) then
            print *, "PASS: test_builder_getters"
        else
            print *, "FAIL: test_builder_getters - not all components valid"
        end if
        
        ! Explicit cleanup
        call builder%finalize()
    end function test_builder_getters

end program test_mlir_c_builder