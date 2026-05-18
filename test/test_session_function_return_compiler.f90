program test_session_function_return_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: y'//new_line('a')// &
        '  y = bounded(0)'//new_line('a')// &
        '  stop y'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function bounded(x)'//new_line('a')// &
        '    integer, intent(in) :: x'//new_line('a')// &
        '    if (x == 0) then'//new_line('a')// &
        '      bounded = 17'//new_line('a')// &
        '      return'//new_line('a')// &
        '    else'//new_line('a')// &
        '      bounded = x'//new_line('a')// &
        '    end if'//new_line('a')// &
        '  end function bounded'//new_line('a')// &
        'end program main'

    print *, '=== function early-return compiler test ==='

    if (.not. expect_exit_status(source, 17, &
        '/tmp/ffc_session_func_return_test')) stop 1

    print *, 'PASS: function early return lowers through direct LIRIC session'
end program test_session_function_return_compiler
