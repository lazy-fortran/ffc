program test_session_subroutine_return_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: x'//new_line('a')// &
        '  x = 0'//new_line('a')// &
        '  call maybe_set(1, x)'//new_line('a')// &
        '  call maybe_set(0, x)'//new_line('a')// &
        '  stop x'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine maybe_set(skip, target)'//new_line('a')// &
        '    integer, intent(in) :: skip'//new_line('a')// &
        '    integer, intent(inout) :: target'//new_line('a')// &
        '    if (skip == 1) then'//new_line('a')// &
        '      return'//new_line('a')// &
        '    else'//new_line('a')// &
        '      target = target + 7'//new_line('a')// &
        '    end if'//new_line('a')// &
        '  end subroutine maybe_set'//new_line('a')// &
        'end program main'

    print *, '=== subroutine early-return compiler test ==='

    if (.not. expect_exit_status(source, 7, &
        '/tmp/ffc_session_subr_return_test')) stop 1

    print *, 'PASS: subroutine early return lowers through direct LIRIC session'
end program test_session_subroutine_return_compiler
