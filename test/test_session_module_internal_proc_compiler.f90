program test_session_module_internal_proc
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== module procedure with internal procedure compiler test ==='

    all_passed = .true.
    if (.not. test_module_sub_with_internal_sub()) all_passed = .false.
    if (.not. test_module_sub_with_internal_function()) all_passed = .false.
    if (.not. test_module_function_with_internal_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module procedures may contain internal procedures'

contains

    logical function test_module_sub_with_internal_sub()
        ! #249 B7f: a module subroutine that contains an internal subroutine.
        ! The inner subroutine is called from the outer one and terminates the
        ! program with a known status.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer()'//new_line('a')// &
            '    call inner()'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    subroutine inner()'//new_line('a')// &
            '      stop 7'//new_line('a')// &
            '    end subroutine inner'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call outer()'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_module_sub_with_internal_sub = expect_exit_status( &
            source, 7, '/tmp/ffc_session_mod_internal_sub_test')
    end function test_module_sub_with_internal_sub

    logical function test_module_sub_with_internal_function()
        ! A module subroutine that contains an internal integer function and
        ! uses its result to set the process exit status.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer()'//new_line('a')// &
            '    integer :: r'//new_line('a')// &
            '    r = triple(4)'//new_line('a')// &
            '    stop r'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    integer function triple(n)'//new_line('a')// &
            '      integer, intent(in) :: n'//new_line('a')// &
            '      triple = n + n + n'//new_line('a')// &
            '    end function triple'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call outer()'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_module_sub_with_internal_function = expect_exit_status( &
                source, 12, '/tmp/ffc_session_mod_internal_fn_test')
        end function test_module_sub_with_internal_function

        logical function test_module_function_with_internal_function()
            ! A module integer function that contains an internal integer function;
            ! the program reports the outer function's result.
            character(len=*), parameter :: source = &
                'module m'//new_line('a')// &
                '  implicit none'//new_line('a')// &
                'contains'//new_line('a')// &
                '  integer function compute(n)'//new_line('a')// &
                '    integer, intent(in) :: n'//new_line('a')// &
                '    compute = bump(n) + 1'//new_line('a')// &
                '  contains'//new_line('a')// &
                '    integer function bump(x)'//new_line('a')// &
                '      integer, intent(in) :: x'//new_line('a')// &
                '      bump = x * 2'//new_line('a')// &
                '    end function bump'//new_line('a')// &
                '  end function compute'//new_line('a')// &
                'end module m'//new_line('a')// &
                'program main'//new_line('a')// &
                '  use m'//new_line('a')// &
                '  stop compute(5)'//new_line('a')// &
                'end program main'

            test_module_function_with_internal_function = expect_exit_status( &
                    source, 11, '/tmp/ffc_session_mod_fn_internal_fn_test')
            end function test_module_function_with_internal_function

        end program test_session_module_internal_proc
