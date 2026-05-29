program test_session_select_case_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session select case compiler test ==='

    all_passed = .true.
    if (.not. test_select_case_one_arm_matches()) all_passed = .false.
    if (.not. test_select_case_one_arm_default()) all_passed = .false.
    if (.not. test_select_case_three_arms_matches_first()) all_passed = .false.
    if (.not. test_select_case_three_arms_matches_middle()) all_passed = .false.
    if (.not. test_select_case_three_arms_matches_default()) all_passed = .false.
    if (.not. test_select_case_multi_label_first_matches()) all_passed = .false.
    if (.not. test_select_case_multi_label_second_matches()) all_passed = .false.
    if (.not. test_select_case_multi_label_no_match()) all_passed = .false.
    if (.not. test_select_case_closed_range()) all_passed = .false.
    if (.not. test_select_case_unbounded_low()) all_passed = .false.
    if (.not. test_select_case_unbounded_high()) all_passed = .false.
    if (.not. test_select_case_mixed_labels_and_ranges()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: SELECT CASE lowers through direct LIRIC'

contains

    logical function test_select_case_one_arm_matches()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 3'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_one_arm_matches = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_match_test')
    end function test_select_case_one_arm_matches

    logical function test_select_case_one_arm_default()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_one_arm_default = expect_exit_status( &
            source, 99, '/tmp/ffc_session_select_default_test')
    end function test_select_case_one_arm_default

    logical function test_select_case_three_arms_matches_first()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_three_arms_matches_first = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_first_test')
    end function test_select_case_three_arms_matches_first

    logical function test_select_case_three_arms_matches_middle()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_three_arms_matches_middle = expect_exit_status( &
            source, 22, '/tmp/ffc_session_select_middle_test')
    end function test_select_case_three_arms_matches_middle

    logical function test_select_case_three_arms_matches_default()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    stop 33'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_three_arms_matches_default = expect_exit_status( &
            source, 99, '/tmp/ffc_session_select_three_default_test')
    end function test_select_case_three_arms_matches_default

    logical function test_select_case_multi_label_first_matches()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (2, 5)'//new_line('a')// &
            '    stop 25'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_multi_label_first_matches = expect_exit_status( &
            source, 25, '/tmp/ffc_session_select_multi_first_test')
    end function test_select_case_multi_label_first_matches

    logical function test_select_case_multi_label_second_matches()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (2, 5)'//new_line('a')// &
            '    stop 25'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_multi_label_second_matches = expect_exit_status( &
            source, 25, '/tmp/ffc_session_select_multi_second_test')
    end function test_select_case_multi_label_second_matches

    logical function test_select_case_multi_label_no_match()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (2, 5)'//new_line('a')// &
            '    stop 25'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_multi_label_no_match = expect_exit_status( &
            source, 99, '/tmp/ffc_session_select_multi_none_test')
    end function test_select_case_multi_label_no_match

    logical function test_select_case_closed_range()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 4'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1:5)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_closed_range = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_closed_range_test')
    end function test_select_case_closed_range

    logical function test_select_case_unbounded_low()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (:5)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_unbounded_low = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_unbounded_low_test')
    end function test_select_case_unbounded_low

    logical function test_select_case_unbounded_high()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (3:)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_unbounded_high = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_unbounded_high_test')
    end function test_select_case_unbounded_high

    logical function test_select_case_mixed_labels_and_ranges()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 9'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (1, 3:5, 9)'//new_line('a')// &
            '    stop 11'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_mixed_labels_and_ranges = expect_exit_status( &
            source, 11, '/tmp/ffc_session_select_mixed_range_test')
    end function test_select_case_mixed_labels_and_ranges

end program test_session_select_case_compiler
