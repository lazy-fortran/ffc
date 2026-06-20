program test_session_module_procedure_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== module procedure compiler test ==='

    all_passed = .true.
    if (.not. test_module_subroutine_call()) all_passed = .false.
    if (.not. test_module_integer_function_call()) all_passed = .false.
    if (.not. test_two_modules_same_procedure_name()) all_passed = .false.
    if (.not. test_module_subroutine_with_real_arg()) all_passed = .false.
    if (.not. test_module_subroutine_with_logical_arg()) all_passed = .false.
    if (.not. test_bare_module_compiles_and_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module procedures lower under mangled symbols'

contains

    logical function test_module_subroutine_call()
        ! #119: a module-defined subroutine is callable from the program.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine emit5()'//new_line('a')// &
            '    stop 5'//new_line('a')// &
            '  end subroutine emit5'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call emit5()'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_module_subroutine_call = expect_exit_status( &
            source, 5, '/tmp/ffc_session_mod_sub_test')
    end function test_module_subroutine_call

    logical function test_module_integer_function_call()
        ! #119/#165: a module integer function returns its value.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function twice(n)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    twice = n + n'//new_line('a')// &
            '  end function twice'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop twice(6)'//new_line('a')// &
            'end program main'

        test_module_integer_function_call = expect_exit_status( &
            source, 12, '/tmp/ffc_session_mod_fn_test')
    end function test_module_integer_function_call

    logical function test_two_modules_same_procedure_name()
        ! #119: two modules define init; both compile and link (distinct
        ! mangled symbols).
        character(len=*), parameter :: source = &
            'module a'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine init()'//new_line('a')// &
            '  end subroutine init'//new_line('a')// &
            'end module a'//new_line('a')// &
            'module b'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine init()'//new_line('a')// &
            '  end subroutine init'//new_line('a')// &
            'end module b'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use a'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_two_modules_same_procedure_name = expect_exit_status( &
            source, 0, '/tmp/ffc_session_two_mod_test')
    end function test_two_modules_same_procedure_name

    logical function test_module_subroutine_with_real_arg()
        ! #165: a module subroutine mutates a real intent(inout) argument.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump(x)'//new_line('a')// &
            '    real, intent(inout) :: x'//new_line('a')// &
            '    x = x + 1.0'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  real :: y'//new_line('a')// &
            '  y = 4.0'//new_line('a')// &
            '  call bump(y)'//new_line('a')// &
            '  stop int(y)'//new_line('a')// &
            'end program main'

        test_module_subroutine_with_real_arg = expect_exit_status( &
            source, 5, '/tmp/ffc_session_mod_real_test')
    end function test_module_subroutine_with_real_arg

    logical function test_module_subroutine_with_logical_arg()
        ! #165: a module subroutine reads a logical intent(in) argument.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine pick(flag, out)'//new_line('a')// &
            '    logical, intent(in) :: flag'//new_line('a')// &
            '    integer, intent(out) :: out'//new_line('a')// &
            '    if (flag) then'//new_line('a')// &
            '      out = 7'//new_line('a')// &
            '    else'//new_line('a')// &
            '      out = 3'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine pick'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = 0'//new_line('a')// &
            '  call pick(.true., r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_module_subroutine_with_logical_arg = expect_exit_status( &
            source, 7, '/tmp/ffc_session_mod_logical_test')
    end function test_module_subroutine_with_logical_arg

    logical function test_bare_module_compiles_and_runs()
        ! #263: a module-only source (no program) lowers its procedures under
        ! mangled names and emits a runnable no-main executable that produces
        ! no output and exits 0. This is the dominant conformance-corpus shape.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function square(x) result(r)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    r = x * x'//new_line('a')// &
            '  end function square'//new_line('a')// &
            '  subroutine noop()'//new_line('a')// &
            '  end subroutine noop'//new_line('a')// &
            'end module m'

        test_bare_module_compiles_and_runs = expect_output( &
            source, '', '/tmp/ffc_session_bare_module_test')
    end function test_bare_module_compiles_and_runs

end program test_session_module_procedure_compiler
