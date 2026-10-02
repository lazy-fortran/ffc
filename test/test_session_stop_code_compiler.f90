! fo: dispatcher
module ffc_case_test_session_stop_code_compiler
    implicit none
    private
    public :: case_test_session_stop_code_compiler
    interface
        subroutine case_test_session_stop_code_compiler()
        end subroutine case_test_session_stop_code_compiler
    end interface
end module ffc_case_test_session_stop_code_compiler

subroutine case_test_session_stop_code_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    print *, '=== direct session stop code compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'stop 2 + 3 * 4'//new_line('a')// &
        'end program main', 14, &
        '/tmp/ffc_session_stop_code_expr_test')) stop 1

    print *, 'PASS: integer stop expression lowers through direct LIRIC session'
end subroutine case_test_session_stop_code_compiler
