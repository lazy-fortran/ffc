! fo: dispatcher
module ffc_case_test_session_logical_variable_compiler
    implicit none
    private
    public :: case_test_session_logical_variable_compiler
    interface
        subroutine case_test_session_logical_variable_compiler()
        end subroutine case_test_session_logical_variable_compiler
    end interface
end module ffc_case_test_session_logical_variable_compiler

subroutine case_test_session_logical_variable_compiler()
    use ffc_test_support, only: expect_output
    implicit none
    save

    print *, '=== direct session logical variable compiler test ==='

    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  logical :: flag'//new_line('a')// &
        '  flag = .true.'//new_line('a')// &
        '  if (flag) then'//new_line('a')// &
        '    print *, flag'//new_line('a')// &
        '  else'//new_line('a')// &
        '    print *, .false.'//new_line('a')// &
        '  end if'//new_line('a')// &
        'end program main', ' T'//new_line('a'), &
        '/tmp/ffc_session_logical_var_test')) stop 1

    print *, 'PASS: logical variables lower through direct LIRIC session'
end subroutine case_test_session_logical_variable_compiler
