program test_session_internal_write_compiler
    ! Internal write: write (buf, fmt) value into a fixed-length character
    ! variable. Output is checked by printing the buffer with '(A)' and
    ! comparing the exact bytes.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session internal write compiler test ==='

    all_passed = .true.
    if (.not. test_internal_write_integer_with_i0()) all_passed = .false.
    if (.not. test_internal_write_integer_with_i5()) all_passed = .false.
    if (.not. test_internal_write_string_with_a()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: internal write lowers through direct LIRIC session'

contains

    logical function test_internal_write_integer_with_i0()
        ! buf(len=20): "42" then 18 blanks. print '(A)', buf adds one leading
        ! list-directed... no: '(A)' is formatted, no leading blank.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            "  write (buf, '(I0)') 42"//new_line('a')// &
            "  print '(A)', buf"//new_line('a')// &
            'end program main'

        test_internal_write_integer_with_i0 = expect_output(source, &
            '42                  '//new_line('a'), '/tmp/ffc_iwrite_i0_test')
    end function test_internal_write_integer_with_i0

    logical function test_internal_write_integer_with_i5()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            "  write (buf, '(I5)') 42"//new_line('a')// &
            "  print '(A)', buf"//new_line('a')// &
            'end program main'

        test_internal_write_integer_with_i5 = expect_output(source, &
            '   42               '//new_line('a'), '/tmp/ffc_iwrite_i5_test')
    end function test_internal_write_integer_with_i5

    logical function test_internal_write_string_with_a()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            "  write (buf, '(A)') 'hello'"//new_line('a')// &
            "  print '(A)', buf"//new_line('a')// &
            'end program main'

        test_internal_write_string_with_a = expect_output(source, &
            'hello               '//new_line('a'), '/tmp/ffc_iwrite_a_test')
    end function test_internal_write_string_with_a

end program test_session_internal_write_compiler
