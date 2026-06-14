program test_session_generic_interface_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== generic interface compiler test ==='

    all_passed = .true.
    if (.not. test_generic_integer_specific()) all_passed = .false.
    if (.not. test_generic_real_specific()) all_passed = .false.
    if (.not. test_generic_subroutine()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: generic interfaces resolve to correct specific procedures'

contains

    logical function test_generic_integer_specific()
        ! B7c: generic foo dispatches to foo_int when passed an integer.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface foo'//new_line('a')// &
            '    module procedure foo_int, foo_real'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function foo_int(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    foo_int = x + 10'//new_line('a')// &
            '  end function foo_int'//new_line('a')// &
            '  integer function foo_real(x)'//new_line('a')// &
            '    real, intent(in) :: x'//new_line('a')// &
            '    foo_real = int(x) + 20'//new_line('a')// &
            '  end function foo_real'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 5'//new_line('a')// &
            '  stop foo(n)'//new_line('a')// &
            'end program main'

        ! foo(5) -> foo_int(5) -> 5 + 10 = 15
        test_generic_integer_specific = expect_exit_status( &
            source, 15, '/tmp/ffc_session_generic_int')
    end function test_generic_integer_specific

    logical function test_generic_real_specific()
        ! B7c: generic foo dispatches to foo_real when passed a real.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface foo'//new_line('a')// &
            '    module procedure foo_int, foo_real'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function foo_int(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    foo_int = x + 10'//new_line('a')// &
            '  end function foo_int'//new_line('a')// &
            '  integer function foo_real(x)'//new_line('a')// &
            '    real, intent(in) :: x'//new_line('a')// &
            '    foo_real = int(x) + 20'//new_line('a')// &
            '  end function foo_real'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  r = 3.0'//new_line('a')// &
            '  stop foo(r)'//new_line('a')// &
            'end program main'

        ! foo(3.0) -> foo_real(3.0) -> int(3.0) + 20 = 23
        test_generic_real_specific = expect_exit_status( &
            source, 23, '/tmp/ffc_session_generic_real')
    end function test_generic_real_specific

    logical function test_generic_subroutine()
        ! B7c: generic subroutine interface dispatches by first arg type.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface print_val'//new_line('a')// &
            '    module procedure print_int, print_real'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine print_int(x, out)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    integer, intent(out) :: out'//new_line('a')// &
            '    out = x + 1'//new_line('a')// &
            '  end subroutine print_int'//new_line('a')// &
            '  subroutine print_real(x, out)'//new_line('a')// &
            '    real, intent(in) :: x'//new_line('a')// &
            '    integer, intent(out) :: out'//new_line('a')// &
            '    out = int(x) + 2'//new_line('a')// &
            '  end subroutine print_real'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer :: n, result'//new_line('a')// &
            '  n = 7'//new_line('a')// &
            '  call print_val(n, result)'//new_line('a')// &
            '  stop result'//new_line('a')// &
            'end program main'

        ! print_val(7, result) -> print_int(7, result) -> result = 7 + 1 = 8
        test_generic_subroutine = expect_exit_status( &
            source, 8, '/tmp/ffc_session_generic_sub')
    end function test_generic_subroutine

end program test_session_generic_interface_compiler
