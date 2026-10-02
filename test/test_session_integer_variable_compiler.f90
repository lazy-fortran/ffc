! fo: dispatcher
module ffc_case_test_session_integer_variable_compiler
    implicit none
    private
    public :: case_test_session_integer_variable_compiler
    interface
        subroutine case_test_session_integer_variable_compiler()
        end subroutine case_test_session_integer_variable_compiler
    end interface
end module ffc_case_test_session_integer_variable_compiler

subroutine case_test_session_integer_variable_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    print *, '=== direct session integer variable compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer :: x'//new_line('a')// &
        'x = 40 + 2'//new_line('a')// &
        'stop x'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_session_integer_var_test')) stop 1

    print *, 'PASS: integer variable lowers through direct LIRIC session'
end subroutine case_test_session_integer_variable_compiler
