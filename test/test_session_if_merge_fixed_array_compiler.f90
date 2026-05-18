program test_session_if_merge_fixed_array_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: a(3)'//new_line('a')// &
        '  integer :: flag'//new_line('a')// &
        '  a(1) = 0'//new_line('a')// &
        '  a(2) = 0'//new_line('a')// &
        '  a(3) = 0'//new_line('a')// &
        '  flag = 1'//new_line('a')// &
        '  if (flag == 1) then'//new_line('a')// &
        '    a(1) = 11'//new_line('a')// &
        '  else'//new_line('a')// &
        '    a(2) = 22'//new_line('a')// &
        '  end if'//new_line('a')// &
        '  stop a(1) + a(3)'//new_line('a')// &
        'end program main'

    print *, '=== if-merge fixed-size array compiler test ==='

    if (.not. expect_exit_status(source, 11, &
        '/tmp/ffc_session_if_merge_array_test')) stop 1

    print *, 'PASS: fixed-size array survives IF merge through direct LIRIC session'
end program test_session_if_merge_fixed_array_compiler
