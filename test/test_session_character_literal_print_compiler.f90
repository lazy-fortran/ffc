program test_session_character_literal_print_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session character literal print compiler test ==='

    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  print *, "hello"'//new_line('a')// &
         'end program main', ' hello'//new_line('a'), &
         '/tmp/ffc_session_character_literal_print_test')) stop 1

    print *, 'PASS: character literal print lowers through direct LIRIC session'
end program test_session_character_literal_print_compiler
