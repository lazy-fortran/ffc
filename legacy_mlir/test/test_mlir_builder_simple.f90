program test_mlir_builder_simple
    use iso_c_binding
    use mlir_c_core
    implicit none

    logical :: all_tests_passed
    type(mlir_context_t) :: context

    print *, "=== Simple MLIR Builder Tests ==="
    print *

    all_tests_passed = .true.

    ! Simple test - can we create a context?
    context = create_mlir_context()
    if (.not. context%is_valid()) then
        print *, "FAIL: context creation"
        all_tests_passed = .false.
    else
        print *, "PASS: context creation"
        call destroy_mlir_context(context)
    end if

    print *
    if (all_tests_passed) then
        print *, "All simple MLIR builder tests passed!"
        stop 0
    else
        print *, "Some simple MLIR builder tests failed!"
        stop 1
    end if

end program test_mlir_builder_simple