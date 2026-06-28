program test_session_recursive_function_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== recursive + scalar pure function compiler test ==='

    all_passed = .true.
    if (.not. test_recursive_and_pure_output()) all_passed = .false.
    if (.not. test_recursive_module_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: recursive and scalar pure functions lower through direct LIRIC'

contains

    logical function test_recursive_and_pure_output()
        ! A recursive scalar factorial and a scalar pure function in one program.
        ! Output is compared against the gfortran reference (120\n21\n).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print '(I0)', fact(5)"//new_line('a')// &
            "  print '(I0)', triple(7)"//new_line('a')// &
            'contains'//new_line('a')// &
            '  recursive function fact(n) result(r)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    integer :: r'//new_line('a')// &
            '    if (n <= 1) then'//new_line('a')// &
            '      r = 1'//new_line('a')// &
            '    else'//new_line('a')// &
            '      r = n * fact(n - 1)'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end function fact'//new_line('a')// &
            '  pure function triple(n) result(r)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    integer :: r'//new_line('a')// &
            '    r = n * 3'//new_line('a')// &
            '  end function triple'//new_line('a')// &
            'end program main'

        test_recursive_and_pure_output = expect_output( &
            source, '120'//new_line('a')//'21'//new_line('a'), &
            '/tmp/ffc_session_recursive_function_test')
    end function test_recursive_and_pure_output

    logical function test_recursive_module_function()
        ! A recursive function defined in a module (no main program) must register
        ! itself so the self-call resolves; the unit lowers to a no-main object.
        character(len=*), parameter :: source = &
            'module recursion_mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  recursive function factorial(n) result(f)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    integer :: f'//new_line('a')// &
            '    if (n <= 1) then'//new_line('a')// &
            '      f = 1'//new_line('a')// &
            '    else'//new_line('a')// &
            '      f = n * factorial(n - 1)'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end function factorial'//new_line('a')// &
            'end module recursion_mod'

        ! A bare module lowers but has no entry point; a successful compile is the
        ! pass condition, so reuse expect_exit_status with the no-main exit code 0.
        test_recursive_module_function = expect_exit_status( &
                source, 0, '/tmp/ffc_session_recursive_module_test')
        end function test_recursive_module_function

    end program test_session_recursive_function_compiler
