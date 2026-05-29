program test_session_recursive_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session recursive procedure compiler test ==='

    all_passed = .true.
    if (.not. test_recursive_integer_factorial()) all_passed = .false.
    if (.not. test_named_result_variable()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: recursive procedures lower through direct LIRIC'

contains

    logical function test_recursive_integer_factorial()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop fact(5)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  recursive function fact(n) result(r)'//new_line('a')// &
            '    integer :: r'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    if (n <= 1) then'//new_line('a')// &
            '      r = 1'//new_line('a')// &
            '    else'//new_line('a')// &
            '      r = n * fact(n - 1)'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end function fact'//new_line('a')// &
            'end program main'

        test_recursive_integer_factorial = expect_exit_status( &
            source, 120, '/tmp/ffc_session_recursive_fact_test')
    end function test_recursive_integer_factorial

    logical function test_named_result_variable()
        ! A function whose result variable is named (result(r)) returns r, not
        ! the function-name symbol.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop add4(3)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function add4(n) result(r)'//new_line('a')// &
            '    integer :: r'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    r = n + 4'//new_line('a')// &
            '  end function add4'//new_line('a')// &
            'end program main'

        test_named_result_variable = expect_exit_status( &
            source, 7, '/tmp/ffc_session_named_result_test')
    end function test_named_result_variable

end program test_session_recursive_compiler
