program test_session_goto_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
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

    ! PAUSE continues execution under batch conformance.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '    implicit none'//new_line('a')// &
         '    print *, "before"'//new_line('a')// &
         '    pause "halt"'//new_line('a')// &
         '    print *, "after"'//new_line('a')// &
         'end program main', &
         ' before'//new_line('a')//' after'//new_line('a'), &
         '/tmp/ffc_session_pause_test')) stop 1
    print *, 'PASS: pause continues execution'

    print *, 'PASS: goto and pause lower through direct LIRIC session'
end program test_session_goto_compiler
