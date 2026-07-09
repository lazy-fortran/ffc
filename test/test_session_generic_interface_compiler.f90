program test_session_generic_interface_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== generic interface compiler test ==='

    all_passed = .true.
    if (.not. test_generic_integer_specific()) all_passed = .false.
    if (.not. test_generic_real_specific()) all_passed = .false.
    if (.not. test_generic_real_result()) all_passed = .false.
    if (.not. test_generic_subroutine()) all_passed = .false.
    if (.not. test_generic_full_signature_dispatch()) all_passed = .false.
    if (.not. test_generic_rank_dispatch()) all_passed = .false.

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

    logical function test_generic_real_result()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface add_values'//new_line('a')// &
            '    module procedure add_integers, add_reals'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function add_integers(a, b)'//new_line('a')// &
            '    integer, intent(in) :: a, b'//new_line('a')// &
            '    add_integers = a + b'//new_line('a')// &
            '  end function add_integers'//new_line('a')// &
            '  real function add_reals(a, b)'//new_line('a')// &
            '    real, intent(in) :: a, b'//new_line('a')// &
            '    add_reals = a + b'//new_line('a')// &
            '  end function add_reals'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop int(add_values(5.0, 3.0))'//new_line('a')// &
            'end program main'

        test_generic_real_result = expect_exit_status( &
            source, 8, '/tmp/ffc_session_generic_real_result')
    end function test_generic_real_result

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

        logical function test_generic_full_signature_dispatch()
            ! B7c: when two specifics share the first argument kind, dispatch uses
            ! the full signature so overloads like (integer, integer) and
            ! (integer, real) stay distinct.
            character(len=*), parameter :: source = &
                'module m'//new_line('a')// &
                '  implicit none'//new_line('a')// &
                '  interface mix'//new_line('a')// &
                '    module procedure mix_ii, mix_ir'//new_line('a')// &
                '  end interface'//new_line('a')// &
                'contains'//new_line('a')// &
                '  integer function mix_ii(a, b)'//new_line('a')// &
                '    integer, intent(in) :: a, b'//new_line('a')// &
                '    mix_ii = a + b + 10'//new_line('a')// &
                '  end function mix_ii'//new_line('a')// &
                '  integer function mix_ir(a, b)'//new_line('a')// &
                '    integer, intent(in) :: a'//new_line('a')// &
                '    real(8), intent(in) :: b'//new_line('a')// &
                '    mix_ir = a + 100 + int(b)'//new_line('a')// &
                '  end function mix_ir'//new_line('a')// &
                'end module m'//new_line('a')// &
                'program main'//new_line('a')// &
                '  use m'//new_line('a')// &
                '  integer :: n'//new_line('a')// &
                '  real(8) :: r'//new_line('a')// &
                '  n = mix(5, 6)'//new_line('a')// &
                '  if (n /= 21) error stop 1'//new_line('a')// &
                '  n = mix(5, 6.0d0)'//new_line('a')// &
                '  if (n /= 111) error stop 2'//new_line('a')// &
                '  stop n'//new_line('a')// &
                'end program main'

            test_generic_full_signature_dispatch = expect_exit_status( &
                source, 111, '/tmp/ffc_session_generic_sig')
        end function test_generic_full_signature_dispatch

        logical function test_generic_rank_dispatch()
            ! B7c: generic with two specifics sharing element kind but differing
            ! in rank (scalar vs array) resolves to the correct specific based on
            ! the actual argument rank, not just the kind vector.
            character(len=*), parameter :: source = &
                'module m'//new_line('a')// &
                '  implicit none'//new_line('a')// &
                '  interface proc'//new_line('a')// &
                '    module procedure proc_scalar, proc_array'//new_line('a')// &
                '  end interface'//new_line('a')// &
                'contains'//new_line('a')// &
                '  integer function proc_scalar(x)'//new_line('a')// &
                '    integer, intent(in) :: x'//new_line('a')// &
                '    proc_scalar = x * 10'//new_line('a')// &
                '  end function proc_scalar'//new_line('a')// &
                '  integer function proc_array(x)'//new_line('a')// &
                '    integer, intent(in) :: x(3)'//new_line('a')// &
                '    proc_array = x(1) + x(2) + x(3)'//new_line('a')// &
                '  end function proc_array'//new_line('a')// &
                'end module m'//new_line('a')// &
                'program main'//new_line('a')// &
                '  use m'//new_line('a')// &
                '  integer :: n, a(3)'//new_line('a')// &
                '  n = 5'//new_line('a')// &
                '  a = [1, 2, 3]'//new_line('a')// &
                '  stop proc(n) - proc(a)'//new_line('a')// &
                'end program main'

            ! proc(5) -> proc_scalar(5) -> 50
            ! proc([1,2,3]) -> proc_array([1,2,3]) -> 6
            ! exit code = 50 - 6 = 44
            test_generic_rank_dispatch = expect_exit_status( &
                source, 44, '/tmp/ffc_session_generic_rank')
        end function test_generic_rank_dispatch

    end program test_session_generic_interface_compiler
