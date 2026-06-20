program test_session_operator_overload_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== operator overload compiler test ==='

    all_passed = .true.
    if (.not. test_named_integer_operator()) all_passed = .false.
    if (.not. test_named_real_operator()) all_passed = .false.
    if (.not. test_overloaded_assignment()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: user-defined operators and assignment dispatch to specifics'

contains

    logical function test_named_integer_operator()
        ! interface operator(.myadd.) routes `a .myadd. b` to myadd(a, b).
        character(len=*), parameter :: source = &
            'module opm'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface operator(.myadd.)'//new_line('a')// &
            '    module procedure myadd'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function myadd(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    myadd = a + b + 100'//new_line('a')// &
            '  end function myadd'//new_line('a')// &
            'end module opm'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use opm'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = 3 .myadd. 4'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        ! 3 .myadd. 4 -> 3 + 4 + 100 = 107
        test_named_integer_operator = expect_exit_status( &
            source, 107, '/tmp/ffc_session_op_named_int')
    end function test_named_integer_operator

    logical function test_named_real_operator()
        ! A real(dp) named operator must produce the same value as a plain call.
        character(len=*), parameter :: source = &
            'module dm'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
            new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface operator(.scaledsum.)'//new_line('a')// &
            '    module procedure scaled_sum'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  real(dp) function scaled_sum(a, b)'//new_line('a')// &
            '    real(dp), intent(in) :: a, b'//new_line('a')// &
            '    scaled_sum = (a + b) * 2.0_dp'//new_line('a')// &
            '  end function scaled_sum'//new_line('a')// &
            'end module dm'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use dm'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
            new_line('a')// &
            '  real(dp) :: r'//new_line('a')// &
            '  r = 1.5_dp .scaledsum. 2.5_dp'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'

        ! (1.5 + 2.5) * 2 = 8.0; stdout matches gfortran's list-directed form.
        test_named_real_operator = expect_output( &
            source, '   8.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_session_op_named_real')
    end function test_named_real_operator

    logical function test_overloaded_assignment()
        ! interface assignment(=) routes `x = rhs` to assign_doubled(x, rhs).
        character(len=*), parameter :: source = &
            'module am'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface assignment(=)'//new_line('a')// &
            '    module procedure assign_doubled'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine assign_doubled(lhs, rhs)'//new_line('a')// &
            '    integer, intent(out) :: lhs'//new_line('a')// &
            '    integer, intent(in) :: rhs'//new_line('a')// &
            '    lhs = rhs * 2'//new_line('a')// &
            '  end subroutine assign_doubled'//new_line('a')// &
            'end module am'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use am'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 21'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        ! x = 21 dispatches to assign_doubled -> 21 * 2 = 42
        test_overloaded_assignment = expect_exit_status( &
            source, 42, '/tmp/ffc_session_op_assignment')
    end function test_overloaded_assignment

end program test_session_operator_overload_compiler
