program test_session_block_if_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session block if compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'if (2 < 3) then'//new_line('a')// &
        '    stop 7'//new_line('a')// &
        'else'//new_line('a')// &
        '    stop 1'//new_line('a')// &
        'end if'//new_line('a')// &
        'end program main', 7, &
        '/tmp/ffc_session_block_if_test')) stop 1

    print *, 'PASS: block IF lowers through direct LIRIC session'
end program test_session_block_if_compiler
