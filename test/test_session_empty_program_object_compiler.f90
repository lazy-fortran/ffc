! fo: dispatcher
module ffc_case_test_session_empty_program_object_compiler
    implicit none
    private
    public :: case_test_session_empty_program_object_compiler
    interface
        subroutine case_test_session_empty_program_object_compiler()
        end subroutine case_test_session_empty_program_object_compiler
    end interface
end module ffc_case_test_session_empty_program_object_compiler

subroutine case_test_session_empty_program_object_compiler()
    use ffc_test_support, only: expect_object_exists
    implicit none
    save

    print *, '=== direct session empty program object compiler test ==='

    if (.not. expect_object_exists( &
        'program main'//new_line('a')// &
        'end program main', &
        '/tmp/ffc_session_empty_program_test.o')) stop 1

    print *, 'PASS: empty program emits object through direct LIRIC session'
end subroutine case_test_session_empty_program_object_compiler
