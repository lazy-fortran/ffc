program test_session_ambiguous_interface_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== ambiguous interface compiler test ==='

    all_passed = .true.
    if (.not. test_ambiguous_rejected()) all_passed = .false.
    if (.not. test_distinguishable_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: ambiguous generic interfaces rejected, distinct ones accepted'

contains

    logical function test_ambiguous_rejected()
        ! Two specifics with identical scalar dummy signatures (both a single
        ! default integer) are not distinguishable by type/kind/rank and must be
        ! rejected (F2018 C1514).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface iface'//new_line('a')// &
            '    module procedure sub_a'//new_line('a')// &
            '    module procedure sub_b'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub_a(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            '  subroutine sub_b(y)'//new_line('a')// &
            '    integer, intent(in) :: y'//new_line('a')// &
            '    print *, y'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call iface(3)'//new_line('a')// &
            'end program main'

        test_ambiguous_rejected = expect_error_contains( &
            source, 'ambiguous interfaces', '/tmp/ffc_session_ambiguous_iface')
    end function test_ambiguous_rejected

    logical function test_distinguishable_accepted()
        ! The nearest valid form: the two specifics differ in dummy type
        ! (integer vs real), so the generic is distinguishable and must compile
        ! and run.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface iface'//new_line('a')// &
            '    module procedure sub_int'//new_line('a')// &
            '    module procedure sub_real'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub_int(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            '  subroutine sub_real(y)'//new_line('a')// &
            '    real, intent(in) :: y'//new_line('a')// &
            '    print *, y'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call iface(7)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_distinguishable_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_distinct_iface')
    end function test_distinguishable_accepted

end program test_session_ambiguous_interface_compiler
