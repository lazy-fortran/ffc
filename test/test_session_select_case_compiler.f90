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
    if (.not. test_select_case_character_match()) all_passed = .false.
    if (.not. test_select_case_character_default()) all_passed = .false.
    if (.not. test_select_case_character_blank_padded()) all_passed = .false.
    if (.not. test_select_case_closed_range()) all_passed = .false.
    if (.not. test_select_case_unbounded_low()) all_passed = .false.
    if (.not. test_select_case_unbounded_high()) all_passed = .false.
    if (.not. test_select_case_mixed_labels_and_ranges()) all_passed = .false.
    if (.not. test_select_case_non_terminating_arms()) all_passed = .false.
    if (.not. test_select_case_partially_terminating()) all_passed = .false.
    if (.not. test_select_case_no_default_match()) all_passed = .false.
    if (.not. test_select_case_no_default_fallthrough()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: SELECT CASE lowers through direct LIRIC'

contains

    logical function test_select_case_non_terminating_arms()
        ! Three non-terminating arms each assign a carried integer; after the
        ! construct stop x reflects the chosen arm (here case (2)).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, s'//new_line('a')// &
            '  s = command_argument_count() + 2'//new_line('a')// &
            '  select case (s)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    x = 10'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    x = 20'//new_line('a')// &
            '  case (3)'//new_line('a')// &
            '    x = 30'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    x = 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_select_case_non_terminating_arms = expect_exit_status( &
            source, 20, '/tmp/ffc_select_nonterm_test')
    end function test_select_case_non_terminating_arms

    logical function test_select_case_partially_terminating()
        ! First arm stops, the other arm and default assign; the merge has two
        ! predecessors. Here case (2) is chosen via a runtime selector.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, s'//new_line('a')// &
            '  s = command_argument_count() + 2'//new_line('a')// &
            '  x = 0'//new_line('a')// &
            '  select case (s)'//new_line('a')// &
            '  case (1)'//new_line('a')// &
            '    stop 7'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    x = 20'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    x = 99'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_select_case_partially_terminating = expect_exit_status( &
            source, 20, '/tmp/ffc_select_partterm_test')
    end function test_select_case_partially_terminating

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

    logical function test_select_case_character_match()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: s'//new_line('a')// &
            '  s = "bar"'//new_line('a')// &
            '  select case (s)'//new_line('a')// &
            '  case ("foo")'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  case ("bar")'//new_line('a')// &
            '    stop 2'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 9'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_character_match = expect_exit_status( &
            source, 2, '/tmp/ffc_session_select_char_match_test')
    end function test_select_case_character_match

    logical function test_select_case_character_default()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: s'//new_line('a')// &
            '  s = "zzz"'//new_line('a')// &
            '  select case (s)'//new_line('a')// &
            '  case ("foo")'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 9'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_character_default = expect_exit_status( &
            source, 9, '/tmp/ffc_session_select_char_default_test')
    end function test_select_case_character_default

    logical function test_select_case_character_blank_padded()
        ! A len=5 selector "bar  " matches the 3-char label "bar" (blank pad).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  s = "bar"'//new_line('a')// &
            '  select case (s)'//new_line('a')// &
            '  case ("bar")'//new_line('a')// &
            '    stop 2'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    stop 9'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_character_blank_padded = expect_exit_status( &
            source, 2, '/tmp/ffc_session_select_char_padded_test')
    end function test_select_case_character_blank_padded

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

    logical function test_select_case_no_default_match()
        ! No case default arm; the matching case still runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  stop 7'//new_line('a')// &
            'end program main'

        test_select_case_no_default_match = expect_exit_status( &
            source, 22, '/tmp/ffc_session_select_no_default_match_test')
    end function test_select_case_no_default_match

    logical function test_select_case_no_default_fallthrough()
        ! No case default arm and no matching case; control falls through to the
        ! statement after the construct.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 5'//new_line('a')// &
            '  select case (x)'//new_line('a')// &
            '  case (2)'//new_line('a')// &
            '    stop 22'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  stop 7'//new_line('a')// &
            'end program main'

        test_select_case_no_default_fallthrough = expect_exit_status( &
            source, 7, '/tmp/ffc_session_select_no_default_fall_test')
    end function test_select_case_no_default_fallthrough

end program test_session_select_case_compiler
