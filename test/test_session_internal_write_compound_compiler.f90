program test_session_internal_write_compound_compiler
    ! Compound literal format string in an internal write: write (buf, '(A,I0)')
    ! s, n formats one edit descriptor per value in order.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session compound internal write compiler test ==='

    all_passed = .true.
    if (.not. test_compound_string_then_integer()) all_passed = .false.
    if (.not. test_compound_integer_then_string()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: compound internal write lowers through direct LIRIC session'

contains

    logical function test_compound_string_then_integer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=24) :: res'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 42'//new_line('a')// &
            "  write(res, '(A,I0)') 'via entry: ', n"//new_line('a')// &
            "  print '(A)', trim(res)"//new_line('a')// &
            'end program main'

        test_compound_string_then_integer = expect_output( &
            source, 'via entry: 42'//new_line('a'), &
            '/tmp/ffc_iwrite_compound_ai_test')
    end function test_compound_string_then_integer

    logical function test_compound_integer_then_string()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=24) :: res'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 7'//new_line('a')// &
            "  write(res, '(I0,A)') n, ' items'"//new_line('a')// &
            "  print '(A)', trim(res)"//new_line('a')// &
            'end program main'

        test_compound_integer_then_string = expect_output( &
            source, '7 items'//new_line('a'), &
            '/tmp/ffc_iwrite_compound_ia_test')
    end function test_compound_integer_then_string

end program test_session_internal_write_compound_compiler
