program test_session_character_variable_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character variable compiler test ==='

    all_passed = .true.
    if (.not. test_exact_length_character_print()) all_passed = .false.
    if (.not. test_short_character_assignment_pads()) all_passed = .false.
    if (.not. test_long_character_assignment_truncates()) all_passed = .false.
    if (.not. test_character_concat_two_literals()) all_passed = .false.
    if (.not. test_character_concat_three_literals()) all_passed = .false.
    if (.not. test_character_concat_pads_short_result()) all_passed = .false.
    if (.not. test_character_concat_truncates_long_result()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character variables lower through direct LIRIC session'

contains

    logical function test_exact_length_character_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_exact_length_character_print = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_char_exact_test')
    end function test_exact_length_character_print

    logical function test_short_character_assignment_pads()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_short_character_assignment_pads = expect_output( &
            source, ' hi   '//new_line('a'), &
            '/tmp/ffc_session_char_pad_test')
    end function test_short_character_assignment_pads

    logical function test_long_character_assignment_truncates()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: s'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_long_character_assignment_truncates = expect_output( &
            source, ' hel'//new_line('a'), &
            '/tmp/ffc_session_char_trunc_test')
    end function test_long_character_assignment_truncates

    logical function test_character_concat_two_literals()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  s = "he" // "llo"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_character_concat_two_literals = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_char_concat_two_test')
    end function test_character_concat_two_literals

    logical function test_character_concat_three_literals()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: s'//new_line('a')// &
            '  s = "a" // "b" // "c"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_character_concat_three_literals = expect_output( &
            source, ' abc'//new_line('a'), &
            '/tmp/ffc_session_char_concat_three_test')
    end function test_character_concat_three_literals

    logical function test_character_concat_pads_short_result()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  s = "hi" // "!"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_character_concat_pads_short_result = expect_output( &
            source, ' hi!  '//new_line('a'), &
            '/tmp/ffc_session_char_concat_pad_test')
    end function test_character_concat_pads_short_result

    logical function test_character_concat_truncates_long_result()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  s = "hello" // "world"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_character_concat_truncates_long_result = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_char_concat_trunc_test')
    end function test_character_concat_truncates_long_result

end program test_session_character_variable_compiler
