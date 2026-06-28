program test_session_loop_cycle_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session loop cycle compiler test ==='

    all_passed = .true.
    if (.not. test_cycle_skips_remainder_of_body()) all_passed = .false.
    if (.not. test_cycle_does_not_skip_latch()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: bare cycle in counted DO lowers through direct LIRIC'

contains

    logical function test_cycle_skips_remainder_of_body()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer :: total'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do i = 1, 5'//new_line('a')// &
            '    if (i == 3) then'//new_line('a')// &
            '        cycle'//new_line('a')// &
            '    else'//new_line('a')// &
            '    end if'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop total'//new_line('a')// &
            'end program main'

        test_cycle_skips_remainder_of_body = expect_exit_status( &
            source, 12, &
            '/tmp/ffc_session_cycle_skip_test')
    end function test_cycle_skips_remainder_of_body

    logical function test_cycle_does_not_skip_latch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer :: visits'//new_line('a')// &
            'visits = 0'//new_line('a')// &
            'do i = 1, 5'//new_line('a')// &
            '    visits = visits + 1'//new_line('a')// &
            '    if (i == 3) then'//new_line('a')// &
            '        cycle'//new_line('a')// &
            '    else'//new_line('a')// &
            '    end if'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop visits'//new_line('a')// &
            'end program main'

        test_cycle_does_not_skip_latch = expect_exit_status( &
            source, 5, &
            '/tmp/ffc_session_cycle_latch_test')
    end function test_cycle_does_not_skip_latch

end program test_session_loop_cycle_compiler
