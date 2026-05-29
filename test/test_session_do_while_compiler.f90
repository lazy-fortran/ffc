program test_session_do_while_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session do while compiler test ==='

    all_passed = .true.
    if (.not. test_do_while_counts_up_to_threshold()) all_passed = .false.
    if (.not. test_do_while_zero_iterations()) all_passed = .false.
    if (.not. test_do_while_logical_accumulator()) all_passed = .false.

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

end program test_session_do_while_compiler
