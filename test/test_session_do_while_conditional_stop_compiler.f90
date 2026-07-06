program test_session_do_while_conditional_stop_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== do while conditional early-exit body test ==='

    all_passed = .true.
    if (.not. test_do_while_conditional_error_stop()) all_passed = .false.
    if (.not. test_do_while_conditional_stop_carries_counter()) &
        all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: do while body with a conditional early exit carries state'

contains

    ! A conditional early exit in the body ends the body in a merge block, not
    ! the loop body block. The header phi backedge must come from a dedicated
    ! latch so the carried updates (go = .false.) reach the header. Without the
    ! latch, go stays .true. and the loop never terminates.
    logical function test_do_while_conditional_error_stop()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'logical :: go'//new_line('a')// &
            'integer :: c'//new_line('a')// &
            'go = .true.'//new_line('a')// &
            'c = 0'//new_line('a')// &
            'do while (go)'//new_line('a')// &
            '    go = .false.'//new_line('a')// &
            '    c = c + 1'//new_line('a')// &
            '    if (c > 1) error stop'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop c'//new_line('a')// &
            'end program main'

        test_do_while_conditional_error_stop = expect_exit_status( &
            source, 1, &
            '/tmp/ffc_session_do_while_cond_stop_test')
    end function test_do_while_conditional_error_stop

    logical function test_do_while_conditional_stop_carries_counter()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: c'//new_line('a')// &
            'c = 0'//new_line('a')// &
            'do while (c < 4)'//new_line('a')// &
            '    c = c + 1'//new_line('a')// &
            '    if (c > 100) error stop'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop c'//new_line('a')// &
            'end program main'

        test_do_while_conditional_stop_carries_counter = expect_exit_status( &
            source, 4, &
            '/tmp/ffc_session_do_while_cond_carry_test')
    end function test_do_while_conditional_stop_carries_counter

end program test_session_do_while_conditional_stop_compiler
