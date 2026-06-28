program test_session_loop_exit_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session loop exit compiler test ==='

    all_passed = .true.
    if (.not. test_exit_terminates_simple_loop()) all_passed = .false.
    if (.not. test_exit_carries_accumulator()) all_passed = .false.
    if (.not. test_exit_inside_nested_loops_targets_inner()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: bare exit in counted DO lowers through direct LIRIC'

contains

    logical function test_exit_terminates_simple_loop()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'do i = 1, 10'//new_line('a')// &
            '    if (i == 5) then'//new_line('a')// &
            '        exit'//new_line('a')// &
            '    else'//new_line('a')// &
            '    end if'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop i'//new_line('a')// &
            'end program main'

        test_exit_terminates_simple_loop = expect_exit_status( &
            source, 5, &
            '/tmp/ffc_session_exit_simple_test')
    end function test_exit_terminates_simple_loop

    logical function test_exit_carries_accumulator()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer :: total'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do i = 1, 10'//new_line('a')// &
            '    if (i == 5) then'//new_line('a')// &
            '        exit'//new_line('a')// &
            '    else'//new_line('a')// &
            '        total = total + i'//new_line('a')// &
            '    end if'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop total'//new_line('a')// &
            'end program main'

        test_exit_carries_accumulator = expect_exit_status( &
            source, 10, &
            '/tmp/ffc_session_exit_accum_test')
    end function test_exit_carries_accumulator

    logical function test_exit_inside_nested_loops_targets_inner()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'integer :: j'//new_line('a')// &
            'integer :: total'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do i = 1, 3'//new_line('a')// &
            '    do j = 1, 10'//new_line('a')// &
            '        if (j == 3) then'//new_line('a')// &
            '            exit'//new_line('a')// &
            '        else'//new_line('a')// &
            '        total = total + 1'//new_line('a')// &
            '        end if'//new_line('a')// &
            '    end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop total'//new_line('a')// &
            'end program main'

        test_exit_inside_nested_loops_targets_inner = expect_exit_status( &
            source, 6, &
            '/tmp/ffc_session_exit_nested_test')
    end function test_exit_inside_nested_loops_targets_inner

end program test_session_loop_exit_compiler
