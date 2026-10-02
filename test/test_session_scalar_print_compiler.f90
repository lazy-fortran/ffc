! fo: dispatcher
module ffc_case_test_session_scalar_print_compiler
    implicit none
    private
    public :: case_test_session_scalar_print_compiler
    interface
        subroutine case_test_session_scalar_print_compiler()
        end subroutine case_test_session_scalar_print_compiler
    end interface
end module ffc_case_test_session_scalar_print_compiler

subroutine case_test_session_scalar_print_compiler()
    use ffc_test_support, only: expect_output
    implicit none
    save

    print *, '=== direct session scalar print compiler test ==='

    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer :: x'//new_line('a')// &
        '  x = 2 + 3'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'end program main', '           5'//new_line('a'), &
        '/tmp/ffc_session_scalar_print_test')) stop 1

    print *, 'PASS: integer print lowers through direct LIRIC session'
end subroutine case_test_session_scalar_print_compiler
