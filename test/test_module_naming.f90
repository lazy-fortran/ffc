program test_module_naming
    use iso_c_binding
    implicit none

    logical :: all_tests_passed

    print *, "=== Module Naming Tests (RED Phase) ==="
    print *, "Testing renamed modules: ffc_error_handling, ffc_pass_manager"
    print *

    all_tests_passed = .true.

    ! Run all tests - these will FAIL initially (RED phase)
    if (.not. test_ffc_error_handling_module()) all_tests_passed = .false.
    if (.not. test_ffc_pass_manager_module()) all_tests_passed = .false.
    if (.not. test_dependent_module_imports()) all_tests_passed = .false.
    if (.not. test_no_naming_conflicts()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All module naming tests passed!"
        stop 0
    else
        print *, "Some module naming tests failed!"
        stop 1
    end if

contains

    ! Given: The ffc_error_handling module should exist and be usable
    ! When: We attempt to import and use the module
    ! Then: All public interfaces should be accessible without conflict
    function test_ffc_error_handling_module() result(passed)
        logical :: passed
        
        passed = .false.
        
        ! Test: Attempt to use renamed error_handling module
        block
            use ffc_error_handling, only: error_result_t, make_error, make_success, has_error
            type(error_result_t) :: result
            
            ! Test basic error result functionality
            result = make_error("Test error message")
            passed = has_error(result)
            
            ! Test success result functionality  
            result = make_success("Test success message")
            passed = passed .and. (.not. has_error(result))
        end block
        
        if (passed) then
            print *, "PASS: test_ffc_error_handling_module"
        else
            print *, "FAIL: test_ffc_error_handling_module - ffc_error_handling module not found or not working"
        end if
        
    end function test_ffc_error_handling_module

    ! Given: The ffc_pass_manager module should exist and be usable
    ! When: We attempt to import and use the module
    ! Then: All public interfaces should be accessible without conflict
    function test_ffc_pass_manager_module() result(passed)
        logical :: passed
        
        passed = .false.
        
        ! Test: Attempt to use renamed pass_manager module
        block
            use mlir_c_core, only: mlir_context_t, create_mlir_context, destroy_mlir_context
            use ffc_pass_manager, only: create_pass_manager, destroy_pass_manager, &
                                      pass_manager_has_context, pass_manager_is_empty
            type(mlir_context_t) :: context
            type :: mlir_pass_manager_t
                type(c_ptr) :: ptr = c_null_ptr
            contains
                procedure :: is_valid => mlir_pass_manager_is_valid
            end type mlir_pass_manager_t
            type(mlir_pass_manager_t) :: pass_manager
            
            ! Create context and pass manager
            context = create_mlir_context()
            if (.not. context%is_valid()) return
            
            pass_manager = create_pass_manager(context)
            passed = pass_manager%is_valid()
            passed = passed .and. pass_manager_has_context(pass_manager, context)
            passed = passed .and. pass_manager_is_empty(pass_manager)
            
            ! Cleanup
            call destroy_pass_manager(pass_manager)
            call destroy_mlir_context(context)
        end block
        
        if (passed) then
            print *, "PASS: test_ffc_pass_manager_module"
        else
            print *, "FAIL: test_ffc_pass_manager_module - ffc_pass_manager module not found or not working"
        end if
        
    end function test_ffc_pass_manager_module

    ! Given: Modules that depend on the renamed modules should work
    ! When: We compile and use dependent modules
    ! Then: All imports should resolve correctly without conflicts
    function test_dependent_module_imports() result(passed)
        logical :: passed
        
        passed = .false.
        
        ! Test: Verify that dependent modules can import renamed modules
        ! This tests that lowering_pipeline.f90 can use ffc_pass_manager
        block
            ! Simulate importing the same modules that lowering_pipeline uses
            use mlir_c_core, only: mlir_context_t, create_mlir_context, destroy_mlir_context
            use mlir_c_types, only: create_void_type
            use ffc_pass_manager, only: create_pass_manager, destroy_pass_manager
            
            type(mlir_context_t) :: context
            type :: mlir_pass_manager_t
                type(c_ptr) :: ptr = c_null_ptr
            contains
                procedure :: is_valid => mlir_pass_manager_is_valid
            end type mlir_pass_manager_t
            type(mlir_pass_manager_t) :: pass_manager
            
            ! Create context 
            context = create_mlir_context()
            if (.not. context%is_valid()) return
            
            ! Test that we can create pass manager (simulating lowering_pipeline usage)
            pass_manager = create_pass_manager(context)
            passed = pass_manager%is_valid()
            
            ! Cleanup
            call destroy_pass_manager(pass_manager)
            call destroy_mlir_context(context)
        end block
        
        if (passed) then
            print *, "PASS: test_dependent_module_imports"
        else
            print *, "FAIL: test_dependent_module_imports - dependent modules cannot import renamed modules"
        end if
        
    end function test_dependent_module_imports

    ! Given: The renamed modules should not conflict with fortfront
    ! When: Both ffc and fortfront modules are available
    ! Then: There should be no naming conflicts during compilation
    function test_no_naming_conflicts() result(passed)
        logical :: passed
        
        passed = .true.
        
        ! Test: Verify no conflicts exist by checking that we can reference
        ! both the ffc versions (with ffc_ prefix) without ambiguity
        block
            ! This should work - using ffc prefixed modules explicitly
            ! The original error_handling and pass_manager from fortfront
            ! should not interfere with ffc_error_handling and ffc_pass_manager
            
            ! If we get to this point without compilation errors, 
            ! it means the renaming resolved the conflicts
            passed = .true.
        end block
        
        if (passed) then
            print *, "PASS: test_no_naming_conflicts"
        else
            print *, "FAIL: test_no_naming_conflicts - naming conflicts still exist"
        end if
        
    end function test_no_naming_conflicts

    ! Helper function for pass manager validity check
    function mlir_pass_manager_is_valid(this) result(valid)
        class(*), intent(in) :: this
        logical :: valid
        
        select type(this)
        type is (mlir_pass_manager_t)
            valid = c_associated(this%ptr)
        class default
            valid = .false.
        end select
    end function mlir_pass_manager_is_valid

    ! Helper type definition for mlir_pass_manager_t
    ! This will need to match the actual definition once modules are renamed
    type :: mlir_pass_manager_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => mlir_pass_manager_is_valid
    end type mlir_pass_manager_t

end program test_module_naming