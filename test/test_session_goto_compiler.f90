program test_session_goto_compiler
    use ffc_test_support, only: expect_output, expect_exit_status, &
                                expect_eof_stderr_and_exit
    implicit none

    print *, '=== direct session goto/pause compiler test ==='

    ! Forward goto skips the dead assignment, lands on the labelled CONTINUE.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    implicit none'//new_line('a')// &
         '    integer :: i'//new_line('a')// &
         '    i = 0'//new_line('a')// &
         '    goto 100'//new_line('a')// &
         '    i = 999'//new_line('a')// &
         '100 continue'//new_line('a')// &
         '    print *, i'//new_line('a')// &
         'end program main', &
         '           0'//new_line('a'), &
         '/tmp/ffc_session_goto_forward_test')) stop 1
    print *, 'PASS: forward goto skips statements'

    ! Backward goto drives a counting loop to completion.
    if (.not. expect_exit_status( &
         'program main'//new_line('a')// &
         '    implicit none'//new_line('a')// &
         '    integer :: i'//new_line('a')// &
         '    i = 0'//new_line('a')// &
         '10  i = i + 1'//new_line('a')// &
         '    if (i < 3) goto 10'//new_line('a')// &
         '    stop i'//new_line('a')// &
         'end program main', 3, &
         '/tmp/ffc_session_goto_backward_test')) stop 1
    print *, 'PASS: backward goto loops until the condition fails'

    ! Computed goto branches to the selector-th label (1-based).
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    implicit none'//new_line('a')// &
         '    integer :: choice'//new_line('a')// &
         '    choice = 2'//new_line('a')// &
         '    goto (100, 200, 300), choice'//new_line('a')// &
         '100 print *, "one"'//new_line('a')// &
         '    goto 999'//new_line('a')// &
         '200 print *, "two"'//new_line('a')// &
         '    goto 999'//new_line('a')// &
         '300 print *, "three"'//new_line('a')// &
         '999 continue'//new_line('a')// &
         'end program main', &
         ' two'//new_line('a'), &
         '/tmp/ffc_session_computed_goto_test')) stop 1
    print *, 'PASS: computed goto selects the second label'

    ! Out-of-range selector falls through to the next statement.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    implicit none'//new_line('a')// &
         '    integer :: choice'//new_line('a')// &
         '    choice = 5'//new_line('a')// &
         '    goto (100, 200), choice'//new_line('a')// &
         '    print *, "fell through"'//new_line('a')// &
         '    goto 999'//new_line('a')// &
         '100 print *, "one"'//new_line('a')// &
         '    goto 999'//new_line('a')// &
         '200 print *, "two"'//new_line('a')// &
         '999 continue'//new_line('a')// &
         'end program main', &
         ' fell through'//new_line('a'), &
         '/tmp/ffc_session_computed_goto_oob_test')) stop 1
    print *, 'PASS: out-of-range computed goto falls through'

    ! PAUSE writes its banner to stderr and waits on stdin; end-of-input ends
    ! the job after flushing earlier output, so "after" is never printed (#280).
    if (.not. expect_eof_stderr_and_exit( &
         'program main'//new_line('a')// &
         '    implicit none'//new_line('a')// &
         '    print *, "before"'//new_line('a')// &
         '    pause "halt"'//new_line('a')// &
         '    print *, "after"'//new_line('a')// &
         'end program main', &
         'PAUSE halt'//new_line('a')// &
         'To resume execution, type go.  Other input will terminate the job.'// &
         new_line('a')//' before'//new_line('a'), 0, &
         '/tmp/ffc_session_pause_test')) stop 1
    print *, 'PASS: pause halts on end-of-input'

    print *, 'PASS: goto and pause lower through direct LIRIC session'
end program test_session_goto_compiler
