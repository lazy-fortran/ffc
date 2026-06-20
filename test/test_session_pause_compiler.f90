program test_session_pause_compiler
    ! #280: PAUSE writes a banner to stderr and waits on stdin. With no input
    ! (end-of-file) the job terminates with exit 0, suppressing later output,
    ! matching gfortran. Run each case with stdin from /dev/null.
    use ffc_test_support, only: expect_eof_stderr_and_exit
    implicit none
    logical :: all_passed
    character(len=*), parameter :: resume_line = &
        'To resume execution, type go.  Other input will terminate the job.'

    all_passed = .true.
    print *, '=== PAUSE banner compiler test ==='

    if (.not. test_pause_with_message()) all_passed = .false.
    if (.not. test_pause_without_message()) all_passed = .false.
    if (.not. test_pause_terminates_before_later_output()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: PAUSE banners lower through direct LIRIC session'
    else
        print *, 'FAIL: PAUSE banner test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_pause_with_message()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  pause 'Debug pause'"//new_line('a')// &
            '  print *, "after"'//new_line('a')// &
            'end program main'
        ! EOF on stdin ends the job after the banner; "after" is never printed.
        test_pause_with_message = expect_eof_stderr_and_exit( &
            source, 'PAUSE Debug pause'//new_line('a')// &
            resume_line//new_line('a'), 0, '/tmp/ffc_pause_msg_test')
    end function test_pause_with_message

    logical function test_pause_without_message()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  pause'//new_line('a')// &
            '  print *, "after"'//new_line('a')// &
            'end program main'
        ! gfortran prints "PAUSE " with a trailing space when there is no message.
        test_pause_without_message = expect_eof_stderr_and_exit( &
            source, 'PAUSE '//new_line('a')// &
            resume_line//new_line('a'), 0, '/tmp/ffc_pause_nomsg_test')
    end function test_pause_without_message

    logical function test_pause_terminates_before_later_output()
        ! Output printed before the PAUSE is flushed at exit; output after is
        ! suppressed because EOF terminates the job at the PAUSE.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, "before"'//new_line('a')// &
            "  pause 'stop here'"//new_line('a')// &
            '  print *, "never"'//new_line('a')// &
            'end program main'
        test_pause_terminates_before_later_output = expect_eof_stderr_and_exit( &
            source, 'PAUSE stop here'//new_line('a')// &
            resume_line//new_line('a')// &
            ' before'//new_line('a'), 0, '/tmp/ffc_pause_order_test')
    end function test_pause_terminates_before_later_output

end program test_session_pause_compiler
