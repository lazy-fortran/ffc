! fo: dispatcher
module ffc_case_test_session_integer_function_compiler
    implicit none
    private
    public :: case_test_session_integer_function_compiler
    interface
        subroutine case_test_session_integer_function_compiler()
        end subroutine case_test_session_integer_function_compiler
    end interface
end module ffc_case_test_session_integer_function_compiler

subroutine case_test_session_integer_function_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    print *, '=== direct session integer function compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        '  integer :: x'//new_line('a')// &
        '  x = add(2, 3)'//new_line('a')// &
        '  stop x'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function add(a, b)'//new_line('a')// &
        '    add = a + b'//new_line('a')// &
        '  end function add'//new_line('a')// &
        'end program main', 5, &
        '/tmp/ffc_session_integer_fn_test')) stop 1

    print *, 'PASS: integer function calls lower through direct LIRIC session'
end subroutine case_test_session_integer_function_compiler
