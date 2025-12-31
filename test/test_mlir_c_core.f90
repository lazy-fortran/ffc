program test_mlir_c_core
    use mlir_c_core
    implicit none

    logical :: all_tests_passed

    print *, "=== MLIR C API Core Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests - these will fail initially (RED phase)
    if (.not. test_mlir_context_creation()) all_tests_passed = .false.
    if (.not. test_mlir_context_destruction()) all_tests_passed = .false.
    if (.not. test_mlir_module_creation()) all_tests_passed = .false.
    if (.not. test_mlir_location_creation()) all_tests_passed = .false.
    if (.not. test_mlir_string_ref_handling()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR C API core tests passed!"
        stop 0
    else
        print *, "Some MLIR C API core tests failed!"
        stop 1
    end if

contains

    function test_mlir_context_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        
        ! Create MLIR context
        context = create_mlir_context()
        passed = context%is_valid()
        
        if (passed) then
            print *, "PASS: test_mlir_context_creation"
        else
            print *, "FAIL: test_mlir_context_creation - context not valid"
        end if
        
        ! Clean up
        call destroy_mlir_context(context)
    end function test_mlir_context_creation

    function test_mlir_context_destruction() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        
        ! Create and destroy context
        context = create_mlir_context()
        call destroy_mlir_context(context)
        passed = .not. context%is_valid()
        
        if (passed) then
            print *, "PASS: test_mlir_context_destruction"
        else
            print *, "FAIL: test_mlir_context_destruction - context still valid after destruction"
        end if
    end function test_mlir_context_destruction

    function test_mlir_module_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_module_t) :: module
        type(mlir_location_t) :: location
        
        ! Create context, location, and module
        context = create_mlir_context()
        location = create_unknown_location(context)
        module = create_empty_module(location)
        passed = module%is_valid()
        
        if (passed) then
            print *, "PASS: test_mlir_module_creation"
        else
            print *, "FAIL: test_mlir_module_creation - module not valid"
        end if
        
        ! Clean up
        call destroy_mlir_context(context)
    end function test_mlir_module_creation

    function test_mlir_location_creation() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_location_t) :: location
        
        ! Create context and location
        context = create_mlir_context()
        location = create_unknown_location(context)
        passed = location%is_valid()
        
        if (passed) then
            print *, "PASS: test_mlir_location_creation"
        else
            print *, "FAIL: test_mlir_location_creation - location not valid"
        end if
        
        ! Clean up
        call destroy_mlir_context(context)
    end function test_mlir_location_creation

    function test_mlir_string_ref_handling() result(passed)
        logical :: passed
        type(mlir_string_ref_t) :: str_ref
        character(len=17), target :: test_string = "test_string_value"
        character(len=:), allocatable :: retrieved_string
        
        ! Create string ref and retrieve
        str_ref = create_string_ref(test_string)
        retrieved_string = get_string_from_ref(str_ref)
        passed = (retrieved_string == test_string)
        
        if (passed) then
            print *, "PASS: test_mlir_string_ref_handling"
        else
            print *, "FAIL: test_mlir_string_ref_handling - string mismatch"
            print *, "Expected: '", test_string, "'"
            print *, "Got: '", retrieved_string, "'"
        end if
    end function test_mlir_string_ref_handling

end program test_mlir_c_core