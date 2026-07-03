program test_session_char_array_compare_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session character array comparison test ==='

    all_passed = .true.
    if (.not. test_element_equal_passes()) all_passed = .false.
    if (.not. test_element_notequal_stops()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character array element comparison lowers correctly'

contains

    logical function test_element_equal_passes()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: arr(3)'//new_line('a')// &
            '  arr(1) = "A"'//new_line('a')// &
            '  arr(2) = "B"'//new_line('a')// &
            '  if (arr(1) /= "A") error stop'//new_line('a')// &
            '  if (arr(2) == "A") error stop'//new_line('a')// &
            'end program main'

        test_element_equal_passes = expect_exit_status( &
            source, 0, '/tmp/ffc_session_char_arr_cmp_ok')
    end function test_element_equal_passes

    logical function test_element_notequal_stops()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: arr(2)'//new_line('a')// &
            '  arr(1) = "xy"'//new_line('a')// &
            '  if (arr(1) /= "xy") error stop 7'//new_line('a')// &
            '  if (arr(1) == "xy") error stop 7'//new_line('a')// &
            'end program main'

        test_element_notequal_stops = expect_exit_status( &
            source, 7, '/tmp/ffc_session_char_arr_cmp_stop')
    end function test_element_notequal_stops

end program test_session_char_array_compare_compiler
