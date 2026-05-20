program test_session_empty_program_object_compiler
    use ffc_test_support, only: expect_object_exists
    implicit none

    print *, '=== direct session empty program object compiler test ==='

    if (.not. expect_object_exists( &
         'program main'//new_line('a')// &
         'end program main', &
         '/tmp/ffc_session_empty_program_test.o')) stop 1

    print *, 'PASS: empty program emits object through direct LIRIC session'
end program test_session_empty_program_object_compiler
