program test_session_reject_boz_array_constructor_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== BOZ-in-array-constructor rejection compiler test ==='

    all_passed = .true.
    if (.not. test_boz_in_constructor_rejected()) all_passed = .false.
    if (.not. test_boz_assignment_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: BOZ literal in array constructor rejected, ' // &
        'scalar BOZ assignment still accepted'

contains

    logical function test_boz_in_constructor_rejected()
        ! A BOZ-literal-constant is never a valid ac-value: it is only
        ! permitted as a DATA-statement value, an actual argument to
        ! INT/REAL/DBLE/CMPLX, or the right side of a scalar intrinsic
        ! assignment (gfortran: "cannot appear in an array constructor").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, dimension(2) :: i'//new_line('a')// &
            "  i = (/Z'abcde', Z'abcde'/)"//new_line('a')// &
            '  print *, i'//new_line('a')// &
            'end program main'

        test_boz_in_constructor_rejected = expect_error_contains( &
            source, 'cannot appear in an array constructor', &
            '/tmp/ffc_session_boz_ctor_reject')
    end function test_boz_in_constructor_rejected

    logical function test_boz_assignment_accepted()
        ! The nearest valid form: a BOZ literal assigned directly to a
        ! scalar integer variable must still compile and run.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            "  i = int(Z'ff')"//new_line('a')// &
            '  print *, i'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_boz_assignment_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_boz_scalar_accept')
    end function test_boz_assignment_accepted

end program test_session_reject_boz_array_constructor_compiler
