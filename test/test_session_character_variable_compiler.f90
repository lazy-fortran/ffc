! fo: dispatcher
module ffc_case_test_session_character_variable_compiler
    implicit none
    private
    public :: case_test_session_character_variable_compiler
    interface
        subroutine case_test_session_character_variable_compiler()
        end subroutine case_test_session_character_variable_compiler
    end interface
end module ffc_case_test_session_character_variable_compiler

subroutine case_test_session_character_variable_compiler()
    use ffc_test_support, only: expect_output
    implicit none
    save

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

    if (.not. test_borrowed_character_dummy_assignment()) all_passed = .false.

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
            '/tmp/ffc_session_char_variable_pad_test')
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
            '/tmp/ffc_session_char_variable_trunc_test')
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

    logical function test_borrowed_character_dummy_assignment()
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  character(len=6) :: s'//new_line('a')// &
            '  s = "abcdef"'//new_line('a')// &
            '  call short(s)'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            '  call whole(s)'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine short(x)'//new_line('a')// &
            '    character(len=3), intent(inout) :: x'//new_line('a')// &
            '    x = "XY"'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            '  subroutine whole(x)'//new_line('a')// &
            '    character(len=*), intent(inout) :: x'//new_line('a')// &
            '    x = x(2:)'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '    x = "Z"'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end program'

        test_borrowed_character_dummy_assignment = expect_output( &
            source, ' XY '//new_line('a')//' XY def'//new_line('a')// &
            ' Y def '//new_line('a')//' Z     '//new_line('a'), &
            '/var/tmp/ffc_character_dummy_assignment_test')
    end function test_borrowed_character_dummy_assignment

end subroutine case_test_session_character_variable_compiler
