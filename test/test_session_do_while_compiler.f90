program test_session_do_while_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session do while compiler test ==='

    all_passed = .true.
    if (.not. test_do_while_counts_up_to_threshold()) all_passed = .false.
    if (.not. test_do_while_zero_iterations()) all_passed = .false.
    if (.not. test_do_while_logical_accumulator()) all_passed = .false.
    if (.not. test_do_while_exit_preserves_body_values()) all_passed = .false.
    if (.not. test_do_while_cycle_reaches_latch()) all_passed = .false.
    if (.not. test_do_while_xfail_fixture()) all_passed = .false.
    if (.not. test_do_while_nested_return_fixture()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: do while loops lower through direct LIRIC'

contains

    logical function test_do_while_counts_up_to_threshold()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: counter'//new_line('a')// &
            'counter = 0'//new_line('a')// &
            'do while (counter < 5)'//new_line('a')// &
            '    counter = counter + 1'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop counter'//new_line('a')// &
            'end program main'

        test_do_while_counts_up_to_threshold = expect_exit_status( &
            source, 5, &
            '/tmp/ffc_session_do_while_count_test')
    end function test_do_while_counts_up_to_threshold

    logical function test_do_while_zero_iterations()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: counter'//new_line('a')// &
            'counter = 9'//new_line('a')// &
            'do while (counter < 5)'//new_line('a')// &
            '    counter = counter + 1'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop counter'//new_line('a')// &
            'end program main'

        test_do_while_zero_iterations = expect_exit_status( &
            source, 9, &
            '/tmp/ffc_session_do_while_zero_test')
    end function test_do_while_zero_iterations

    logical function test_do_while_logical_accumulator()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: counter'//new_line('a')// &
            'logical :: seen'//new_line('a')// &
            'counter = 0'//new_line('a')// &
            'seen = .false.'//new_line('a')// &
            'do while (counter < 3)'//new_line('a')// &
            '    counter = counter + 1'//new_line('a')// &
            '    seen = .true.'//new_line('a')// &
            'end do'//new_line('a')// &
            'if (seen) then'//new_line('a')// &
            '    stop counter'//new_line('a')// &
            'else'//new_line('a')// &
            '    stop 0'//new_line('a')// &
            'end if'//new_line('a')// &
            'end program main'

        test_do_while_logical_accumulator = expect_exit_status( &
            source, 3, &
            '/tmp/ffc_session_do_while_logical_test')
    end function test_do_while_logical_accumulator

    logical function test_do_while_exit_preserves_body_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i, total'//new_line('a')// &
            'i = 0'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do while (i < 10)'//new_line('a')// &
            '    i = i + 1'//new_line('a')// &
            '    if (i == 2) exit'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop 10 * i + total'//new_line('a')// &
            'end program main'

        ! The EXIT path observes i=2 and total=1, not the loop-header values.
        test_do_while_exit_preserves_body_values = expect_exit_status( &
            source, 21, '/tmp/ffc_session_do_while_exit_test')
    end function test_do_while_exit_preserves_body_values

    logical function test_do_while_cycle_reaches_latch()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i, total'//new_line('a')// &
            'i = 0'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do while (i < 10)'//new_line('a')// &
            '    i = i + 1'//new_line('a')// &
            '    if (i == 2) cycle'//new_line('a')// &
            '    total = total + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'stop total'//new_line('a')// &
            'end program main'

        test_do_while_cycle_reaches_latch = expect_exit_status( &
            source, 53, '/tmp/ffc_session_do_while_cycle_test')
    end function test_do_while_cycle_reaches_latch

    logical function test_do_while_xfail_fixture()
        character(len=*), parameter :: source = &
            'program while_02'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'i = 0'//new_line('a')// &
            'j = 0'//new_line('a')// &
            'do while (i < 10)'//new_line('a')// &
            '    i = i + 1'//new_line('a')// &
            '    j = j + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'if (j /= 55) error stop'//new_line('a')// &
            'if (i /= 10) error stop'//new_line('a')// &
            'i = 0'//new_line('a')// &
            'j = 0'//new_line('a')// &
            'do while (i < 10)'//new_line('a')// &
            '    i = i + 1'//new_line('a')// &
            '    if (i == 2) exit'//new_line('a')// &
            '    j = j + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'if (j /= 1) error stop'//new_line('a')// &
            'if (i /= 2) error stop'//new_line('a')// &
            'i = 0'//new_line('a')// &
            'j = 0'//new_line('a')// &
            'do while (i < 10)'//new_line('a')// &
            '    i = i + 1'//new_line('a')// &
            '    if (i == 2) cycle'//new_line('a')// &
            '    j = j + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'if (j /= 53) error stop'//new_line('a')// &
            'if (i /= 10) error stop'//new_line('a')// &
            'end'

        test_do_while_xfail_fixture = expect_exit_status( &
            source, 0, '/tmp/ffc_session_do_while_xfail_fixture_test')
    end function test_do_while_xfail_fixture

    logical function test_do_while_nested_return_fixture()
        character(len=*), parameter :: source = &
            'program while_03'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'integer :: n, i, j'//new_line('a')// &
            'i = 0'//new_line('a')// &
            'n = 2'//new_line('a')// &
            'do while (n >= 2)'//new_line('a')// &
            '    if (n >= i) then'//new_line('a')// &
            '        if (n >= i) then'//new_line('a')// &
            '            do j= i+1,n'//new_line('a')// &
            '                i = j + i'//new_line('a')// &
            '            end do'//new_line('a')// &
            '        endif'//new_line('a')// &
            '        n = n - 1'//new_line('a')// &
            '    else'//new_line('a')// &
            '        i = j + i'//new_line('a')// &
            '    endif'//new_line('a')// &
            'end do'//new_line('a')// &
            'return'//new_line('a')// &
            'end'

        test_do_while_nested_return_fixture = expect_exit_status( &
            source, 0, '/tmp/ffc_session_do_while_nested_return_test')
    end function test_do_while_nested_return_fixture

end program test_session_do_while_compiler
