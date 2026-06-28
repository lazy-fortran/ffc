program test_session_stop_code_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session stop code compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'stop 2 + 3 * 4'//new_line('a')// &
        'end program main', 14, &
        '/tmp/ffc_session_stop_code_test')) stop 1

    print *, 'PASS: integer stop expression lowers through direct LIRIC session'
end program test_session_stop_code_compiler
