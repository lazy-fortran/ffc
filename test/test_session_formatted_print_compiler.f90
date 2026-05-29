program test_session_formatted_print_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session formatted print compiler test ==='

    all_passed = .true.
    if (.not. test_integer_i0()) all_passed = .false.
    if (.not. test_integer_i5()) all_passed = .false.
    if (.not. test_string_a_literal()) all_passed = .false.
    if (.not. test_string_a_variable()) all_passed = .false.
    if (.not. test_compound_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: formatted print lowers through direct LIRIC session'

contains

    logical function test_integer_i0()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            "  print '(I0)', i"//new_line('a')// &
            'end program main'

        test_integer_i0 = expect_output(source, '42'//new_line('a'), &
                                        '/tmp/ffc_fmt_i0_test')
    end function test_integer_i0

    logical function test_integer_i5()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            "  print '(I5)', i"//new_line('a')// &
            'end program main'

        test_integer_i5 = expect_output(source, '   42'//new_line('a'), &
                                        '/tmp/ffc_fmt_i5_test')
    end function test_integer_i5

    logical function test_string_a_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print '(A)', 'hello'"//new_line('a')// &
            'end program main'

        test_string_a_literal = expect_output(source, 'hello'//new_line('a'), &
                                              '/tmp/ffc_fmt_a_lit_test')
    end function test_string_a_literal

    logical function test_string_a_variable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            "  s = 'hello'"//new_line('a')// &
            "  print '(A)', s"//new_line('a')// &
            'end program main'

        test_string_a_variable = expect_output(source, 'hello'//new_line('a'), &
                                               '/tmp/ffc_fmt_a_var_test')
    end function test_string_a_variable

    logical function test_compound_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print '(I0, A)', 1, 'x'"//new_line('a')// &
            'end program main'

        test_compound_rejected = expect_error_contains( &
            source, 'compound format strings are not supported', &
            '/tmp/ffc_fmt_compound_test')
    end function test_compound_rejected

end program test_session_formatted_print_compiler
