program test_session_real_literal_print_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session real literal print compiler test ==='

    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  print *, 2.5'//new_line('a')// &
         'end program main', '2.500000'//new_line('a'), &
         '/tmp/ffc_session_real_literal_print_test')) stop 1

    print *, 'PASS: real literal print lowers through direct LIRIC session'
end program test_session_real_literal_print_compiler
