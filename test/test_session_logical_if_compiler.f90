program test_session_logical_if_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    print *, '=== direct session single-line logical IF compiler test ==='

    if (.not. expect_output( &
        'program main'//new_line('a')// &
        'integer :: i'//new_line('a')// &
        'i = 0'//new_line('a')// &
        'if (2 < 3) i = i + 5'//new_line('a')// &
        'if (3 < 2) i = i + 100'//new_line('a')// &
        'print *, i'//new_line('a')// &
        'end program main', &
        '           5'//new_line('a'), &
        '/tmp/ffc_session_logical_if_assign')) stop 1

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer :: i'//new_line('a')// &
        'i = 7'//new_line('a')// &
        'if (i /= 7) error stop'//new_line('a')// &
        'stop 3'//new_line('a')// &
        'end program main', 3, &
        '/tmp/ffc_session_logical_if_guard_false')) stop 1

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer :: i'//new_line('a')// &
        'i = 4'//new_line('a')// &
        'if (i /= 7) error stop'//new_line('a')// &
        'stop 3'//new_line('a')// &
        'end program main', 1, &
        '/tmp/ffc_session_logical_if_guard_true')) stop 1

    print *, 'PASS: single-line logical IF lowers through direct LIRIC session'
end program test_session_logical_if_compiler
