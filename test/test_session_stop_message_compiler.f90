program test_session_stop_message_compiler
    ! gfortran writes a STOP banner to stderr: "STOP <message>" for a character
    ! message (exit 0) and "STOP <n>" for an integer stop code (exit n).
    use ffc_test_support, only: expect_stderr_and_exit
    implicit none

    logical :: all_passed

    all_passed = .true.

    if (.not. test_stop_message()) all_passed = .false.
    if (.not. test_stop_nonzero_code()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: stop banners match gfortran'
    else
        print *, 'FAIL: stop banner mismatch'
        stop 1
    end if

contains

    logical function test_stop_message()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  if (x > 3) then'//new_line('a')// &
            "    stop 'Error: x is too large'"//new_line('a')// &
            '  end if'//new_line('a')// &
            "  print *, 'unreachable'"//new_line('a')// &
            'end program main'

        test_stop_message = expect_stderr_and_exit( &
            source, 'STOP Error: x is too large'//new_line('a'), 0, &
            '/tmp/ffc_session_stop_message_test')
    end function test_stop_message

    logical function test_stop_nonzero_code()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop 99'//new_line('a')// &
            'end program main'

        test_stop_nonzero_code = expect_stderr_and_exit( &
            source, 'STOP 99'//new_line('a'), 99, &
            '/tmp/ffc_session_stop_code_test')
    end function test_stop_nonzero_code

end program test_session_stop_message_compiler
