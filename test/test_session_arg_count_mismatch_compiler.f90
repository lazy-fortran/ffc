program test_session_arg_count_mismatch_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== argument count mismatch compiler test ==='

    all_passed = .true.
    if (.not. test_too_many_actuals_rejected()) all_passed = .false.
    if (.not. test_exact_count_accepted()) all_passed = .false.
    if (.not. test_optional_omission_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: excess actuals rejected, exact and optional-omitted calls run'

contains

    logical function test_too_many_actuals_rejected()
        ! Passing more actual arguments than the callee declares dummies is
        ! always invalid, regardless of keyword or optional dummies.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call s(1, 2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_too_many_actuals_rejected = expect_error_contains( &
            source, 'More actual than formal arguments', &
            '/tmp/ffc_session_argc_excess')
    end function test_too_many_actuals_rejected

    logical function test_exact_count_accepted()
        ! The nearest valid form: exactly as many actuals as dummies runs.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call s(9)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_exact_count_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_argc_exact')
    end function test_exact_count_accepted

    logical function test_optional_omission_accepted()
        ! Omitting a trailing optional dummy is valid and must still run: the
        ! excess-only check must not misfire on fewer actuals than dummies.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call s(1)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x, y)'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer, optional :: y'//new_line('a')// &
            '    if (present(y)) then'//new_line('a')// &
            '      print *, x + y'//new_line('a')// &
            '    else'//new_line('a')// &
            '      print *, x'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_optional_omission_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_argc_optional')
    end function test_optional_omission_accepted

end program test_session_arg_count_mismatch_compiler
