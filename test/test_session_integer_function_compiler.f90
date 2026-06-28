program test_session_integer_function_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

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
end program test_session_integer_function_compiler
