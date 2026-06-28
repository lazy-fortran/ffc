program test_session_logical_literal_print_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session logical literal print compiler test ==='

    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  print *, .true.'//new_line('a')// &
        'end program main', ' T'//new_line('a'), &
        '/tmp/ffc_session_logical_literal_print_test')) stop 1

    print *, 'PASS: logical literal print lowers through direct LIRIC session'
end program test_session_logical_literal_print_compiler
