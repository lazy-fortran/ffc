program test_session_character_intrinsics_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character intrinsics compiler test ==='

    all_passed = .true.
    if (.not. test_len_of_deferred_after_assignment()) all_passed = .false.
    if (.not. test_len_trim_of_padded_fixed_length()) all_passed = .false.
    if (.not. test_len_of_unallocated_is_zero()) all_passed = .false.
    if (.not. test_len_of_fixed_length()) all_passed = .false.
    if (.not. test_trim_fixed_length_padded()) all_passed = .false.
    if (.not. test_trim_print()) all_passed = .false.
    if (.not. test_trim_assigned_to_deferred()) all_passed = .false.
    if (.not. test_iachar_of_literal()) all_passed = .false.
    if (.not. test_achar_iachar_roundtrip()) all_passed = .false.
    if (.not. test_achar_print()) all_passed = .false.
    if (.not. test_index_found_position()) all_passed = .false.
    if (.not. test_index_not_found_returns_zero()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: len/len_trim lower through direct LIRIC session'

contains

    logical function test_len_of_deferred_after_assignment()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_len_of_deferred_after_assignment = expect_exit_status( &
            source, 5, '/tmp/ffc_session_len_deferred_test')
    end function test_len_of_deferred_after_assignment

    logical function test_len_trim_of_padded_fixed_length()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  stop len_trim(s)'//new_line('a')// &
            'end program main'

        test_len_trim_of_padded_fixed_length = expect_exit_status( &
            source, 2, '/tmp/ffc_session_len_trim_padded_test')
    end function test_len_trim_of_padded_fixed_length

    logical function test_len_of_unallocated_is_zero()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_len_of_unallocated_is_zero = expect_exit_status( &
            source, 0, '/tmp/ffc_session_len_unalloc_test')
    end function test_len_of_unallocated_is_zero

    logical function test_len_of_fixed_length()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=7) :: s'//new_line('a')// &
            '  s = "ab"'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_len_of_fixed_length = expect_exit_status( &
            source, 7, '/tmp/ffc_session_len_fixed_test')
    end function test_len_of_fixed_length

    logical function test_trim_fixed_length_padded()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  stop len(trim(s))'//new_line('a')// &
            'end program main'

        ! trim("hi" padded to 10) has length 2
        test_trim_fixed_length_padded = expect_exit_status( &
            source, 2, '/tmp/ffc_session_trim_len_test')
    end function test_trim_fixed_length_padded

    logical function test_trim_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  print *, trim(s)'//new_line('a')// &
            'end program main'

        test_trim_print = expect_output( &
            source, ' hi'//new_line('a'), '/tmp/ffc_session_trim_print_test')
    end function test_trim_print

    logical function test_trim_assigned_to_deferred()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=8) :: s'//new_line('a')// &
            '  character(len=:), allocatable :: t'//new_line('a')// &
            '  s = "hey"'//new_line('a')// &
            '  t = trim(s)'//new_line('a')// &
            '  stop len(t)'//new_line('a')// &
            'end program main'

        ! trim("hey" padded to 8) assigned to t; len(t) == 3
        test_trim_assigned_to_deferred = expect_exit_status( &
            source, 3, '/tmp/ffc_session_trim_assign_test')
    end function test_trim_assigned_to_deferred

    logical function test_iachar_of_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop iachar("A")'//new_line('a')// &
            'end program main'

        test_iachar_of_literal = expect_exit_status( &
            source, 65, '/tmp/ffc_session_iachar_test')
    end function test_iachar_of_literal

    logical function test_achar_iachar_roundtrip()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop iachar(achar(67))'//new_line('a')// &
            'end program main'

        test_achar_iachar_roundtrip = expect_exit_status( &
            source, 67, '/tmp/ffc_session_achar_roundtrip_test')
    end function test_achar_iachar_roundtrip

    logical function test_achar_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, achar(66)'//new_line('a')// &
            'end program main'

        test_achar_print = expect_output( &
            source, ' B'//new_line('a'), '/tmp/ffc_session_achar_print_test')
    end function test_achar_print

    logical function test_index_found_position()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop index("hello", "ll")'//new_line('a')// &
            'end program main'

        test_index_found_position = expect_exit_status( &
            source, 3, '/tmp/ffc_session_index_found_test')
    end function test_index_found_position

    logical function test_index_not_found_returns_zero()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  stop index("hello", "x")'//new_line('a')// &
            'end program main'

        test_index_not_found_returns_zero = expect_exit_status( &
            source, 0, '/tmp/ffc_session_index_notfound_test')
    end function test_index_not_found_returns_zero

end program test_session_character_intrinsics_compiler
