program test_session_rank_mismatch_arg_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== rank mismatch argument compiler test ==='

    all_passed = .true.
    if (.not. test_scalar_var_to_array_dummy_rejected()) all_passed = .false.
    if (.not. test_literal_to_array_dummy_rejected()) all_passed = .false.
    if (.not. test_array_to_array_dummy_accepted()) all_passed = .false.
    if (.not. test_scalar_to_scalar_dummy_accepted()) all_passed = .false.
    if (.not. test_scalar_to_assumed_rank_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar-to-array-dummy rank mismatch rejected, valid calls run'

contains

    logical function test_scalar_var_to_array_dummy_rejected()
        ! A plain scalar variable passed where the callee declares an
        ! explicit-shape array dummy is a rank mismatch (F2018 15.5.2.4).
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  call s(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    integer :: x(3)'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_scalar_var_to_array_dummy_rejected = expect_error_contains( &
            source, 'Rank mismatch in argument', '/tmp/ffc_session_rank_scalar')
    end function test_scalar_var_to_array_dummy_rejected

    logical function test_literal_to_array_dummy_rejected()
        ! A scalar literal actual to an array dummy is likewise a rank mismatch.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call s(5)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    integer :: x(3)'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_literal_to_array_dummy_rejected = expect_error_contains( &
            source, 'Rank mismatch in argument', '/tmp/ffc_session_rank_literal')
    end function test_literal_to_array_dummy_rejected

    logical function test_array_to_array_dummy_accepted()
        ! The nearest valid form: a whole array actual of matching rank to an
        ! array dummy compiles and runs.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  a = 7'//new_line('a')// &
            '  call s(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    integer :: x(3)'//new_line('a')// &
            '    print *, x(1)'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_array_to_array_dummy_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_rank_array_ok')
    end function test_array_to_array_dummy_accepted

    logical function test_scalar_to_scalar_dummy_accepted()
        ! A scalar actual to a scalar dummy is valid and must still run.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a'//new_line('a')// &
            '  a = 9'//new_line('a')// &
            '  call s(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_scalar_to_scalar_dummy_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_rank_scalar_ok')
    end function test_scalar_to_scalar_dummy_accepted

    logical function test_scalar_to_assumed_rank_accepted()
        ! An assumed-rank dummy arr(..) binds to an actual of any rank, scalar
        ! included, so a scalar actual must not be flagged as a rank mismatch.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: x = 1'//new_line('a')// &
            '  call sub(x)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(inp)'//new_line('a')// &
            '    integer, dimension(..) :: inp'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program p'

        test_scalar_to_assumed_rank_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_rank_assumed')
    end function test_scalar_to_assumed_rank_accepted

end program test_session_rank_mismatch_arg_compiler
