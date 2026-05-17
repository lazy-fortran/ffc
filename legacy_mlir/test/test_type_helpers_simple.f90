program test_type_helpers_simple
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use fortfc_type_converter
    use type_conversion_helpers
    implicit none

    logical :: all_tests_passed

    print *, "=== Simple Type Helpers Tests ==="
    print *

    all_tests_passed = .true.

    ! Run basic tests
    if (.not. test_basic_helper_functions()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All simple type helpers tests passed!"
        stop 0
    else
        print *, "Some simple type helpers tests failed!"
        stop 1
    end if

contains

    function test_basic_helper_functions() result(passed)
        logical :: passed
        type(mlir_context_t) :: context
        type(mlir_type_converter_t) :: converter
        type(type_helper_t) :: helpers
        character(len=:), allocatable :: result_str
        
        passed = .true.
        
        ! Create context and helpers
        context = create_mlir_context()
        converter = create_type_converter(context)
        helpers = create_type_helpers(converter)
        passed = passed .and. helpers%is_valid()
        
        ! Test basic string operations
        result_str = helpers%get_reference_type_string("i32")
        passed = passed .and. (result_str == "!fir.ref<i32>")
        
        result_str = helpers%wrap_with_box_type("!fir.array<?xi32>")
        passed = passed .and. (result_str == "!fir.box<!fir.array<?xi32>>")
        
        result_str = helpers%mangle_derived_type_name("person")
        passed = passed .and. (result_str == "_QTperson")
        
        if (passed) then
            print *, "PASS: test_basic_helper_functions"
        else
            print *, "FAIL: test_basic_helper_functions"
        end if
        
        ! Cleanup
        call destroy_type_helpers(helpers)
        call destroy_type_converter(converter)
        call destroy_mlir_context(context)
    end function test_basic_helper_functions

end program test_type_helpers_simple