! fo: dispatcher
module ffc_case_test_session_empty_program_compiler
    implicit none
    private
    public :: case_test_session_empty_program_compiler
    interface
        subroutine case_test_session_empty_program_compiler()
        end subroutine case_test_session_empty_program_compiler
    end interface
end module ffc_case_test_session_empty_program_compiler

subroutine case_test_session_empty_program_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    print *, '=== direct session empty program compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'end program main', 0, &
        '/tmp/ffc_session_empty_program_test')) stop 1

    print *, 'PASS: empty program compiles and runs through direct LIRIC session'
end subroutine case_test_session_empty_program_compiler
