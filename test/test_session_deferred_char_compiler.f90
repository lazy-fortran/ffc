program test_session_deferred_char_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== deferred-char function result compiler tests ==='

    all_passed = .true.
    if (.not. test_declare_deferred_character_compiles()) all_passed = .false.
    if (.not. test_two_deferred_characters_get_independent_descriptors()) &
        all_passed = .false.
    if (.not. test_deferred_literal_assignment_sets_length()) all_passed = .false.
    if (.not. test_deferred_assignment_after_assignment_replaces()) &
        all_passed = .false.
    if (.not. test_deferred_freed_on_normal_exit()) all_passed = .false.
    if (.not. test_unallocated_descriptor_does_not_free()) all_passed = .false.
    if (.not. test_deferred_literal_concat_assign()) all_passed = .false.
    if (.not. test_deferred_three_literal_concat()) all_passed = .false.
    if (.not. test_function_returns_concatenated_deferred_character()) &
        all_passed = .false.
    if (.not. test_function_returns_input_with_suffix()) all_passed = .false.
    if (.not. test_function_result_prints_directly()) all_passed = .false.
    if (.not. test_pass_deferred_to_assumed_length_dummy_uses_len()) &
        all_passed = .false.
    if (.not. test_pass_literal_to_assumed_length_dummy()) all_passed = .false.
    if (.not. test_assumed_length_len_inside_callee()) all_passed = .false.
    if (.not. test_assumed_length_len_trim()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-char function result ABI'

contains

    logical function test_assumed_length_len_inside_callee()
        ! len(s) inside a callee returns the actual length of a literal actual.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call show("hello")'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(s)'//new_line('a')// &
            '    character(len=*), intent(in) :: s'//new_line('a')// &
            '    stop len(s)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_assumed_length_len_inside_callee = expect_exit_status( &
            source, 5, '/tmp/ffc_session_assumed_len_call_test')
    end function test_assumed_length_len_inside_callee

    logical function test_assumed_length_len_trim()
        ! len_trim inside a callee ignores the trailing blanks of a
        ! fixed-length actual.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: fixed'//new_line('a')// &
            '  fixed = "hi"'//new_line('a')// &
            '  call show(fixed)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(s)'//new_line('a')// &
            '    character(len=*), intent(in) :: s'//new_line('a')// &
            '    stop len_trim(s)'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_assumed_length_len_trim = expect_exit_status( &
            source, 2, '/tmp/ffc_session_assumed_len_trim_test')
    end function test_assumed_length_len_trim

    logical function test_pass_deferred_to_assumed_length_dummy_uses_len()
        ! A deferred-length character actual passed to a character(len=*)
        ! intent(in) dummy carries its length; len_trim sees it.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hello world"'//new_line('a')// &
            '  call print_it(s)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine print_it(arg)'//new_line('a')// &
            '    character(len=*), intent(in) :: arg'//new_line('a')// &
            '    stop len_trim(arg)'//new_line('a')// &
            '  end subroutine print_it'//new_line('a')// &
            'end program main'

        test_pass_deferred_to_assumed_length_dummy_uses_len = expect_exit_status( &
            source, 11, '/tmp/ffc_session_assumed_len_deferred_test')
    end function test_pass_deferred_to_assumed_length_dummy_uses_len

    logical function test_pass_literal_to_assumed_length_dummy()
        ! A character literal actual passed to the same dummy carries its
        ! length too.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call print_it("hello")'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine print_it(arg)'//new_line('a')// &
            '    character(len=*), intent(in) :: arg'//new_line('a')// &
            '    stop len_trim(arg)'//new_line('a')// &
            '  end subroutine print_it'//new_line('a')// &
            'end program main'

        test_pass_literal_to_assumed_length_dummy = expect_exit_status( &
            source, 5, '/tmp/ffc_session_assumed_len_literal_test')
    end function test_pass_literal_to_assumed_length_dummy

    logical function test_declare_deferred_character_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_declare_deferred_character_compiles = expect_exit_status( &
            source, 0, '/tmp/ffc_session_deferred_declare_test')
    end function test_declare_deferred_character_compiles

    logical function test_two_deferred_characters_get_independent_descriptors()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(:), allocatable :: a'//new_line('a')// &
            '  character(len=:), allocatable :: b'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_two_deferred_characters_get_independent_descriptors = &
            expect_exit_status( &
            source, 0, '/tmp/ffc_session_deferred_two_decl_test')
    end function test_two_deferred_characters_get_independent_descriptors

    logical function test_deferred_literal_assignment_sets_length()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_deferred_literal_assignment_sets_length = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_deferred_literal_assign_test')
    end function test_deferred_literal_assignment_sets_length

    logical function test_deferred_assignment_after_assignment_replaces()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  s = "world"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_deferred_assignment_after_assignment_replaces = expect_output( &
            source, ' world'//new_line('a'), &
            '/tmp/ffc_session_deferred_reassign_test')
    end function test_deferred_assignment_after_assignment_replaces

    logical function test_deferred_freed_on_normal_exit()
        ! Local deferred-char data is static (literal) or stack (concat), so
        ! normal scope exit reclaims it without an explicit free; just verify
        ! the program runs to completion without crashing.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_deferred_freed_on_normal_exit = expect_exit_status( &
            source, 0, '/tmp/ffc_session_deferred_free_exit_test')
    end function test_deferred_freed_on_normal_exit

    logical function test_unallocated_descriptor_does_not_free()
        ! An unallocated descriptor (data == 0) must not be freed at exit.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_unallocated_descriptor_does_not_free = expect_exit_status( &
            source, 0, '/tmp/ffc_session_deferred_unalloc_free_test')
    end function test_unallocated_descriptor_does_not_free

    logical function test_deferred_literal_concat_assign()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "he" // "llo"'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_deferred_literal_concat_assign = expect_exit_status( &
            source, 5, '/tmp/ffc_session_deferred_concat2_test')
    end function test_deferred_literal_concat_assign

    logical function test_deferred_three_literal_concat()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "a" // "b" // "c"'//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_deferred_three_literal_concat = expect_exit_status( &
            source, 3, '/tmp/ffc_session_deferred_concat3_test')
    end function test_deferred_three_literal_concat

    logical function test_function_returns_concatenated_deferred_character()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  character(len=:), allocatable :: r, s'//new_line('a')// &
            '  r = helper()'//new_line('a')// &
            '  s = r // "cd"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function helper() result(res)'//new_line('a')// &
            '    character(len=:), allocatable :: res'//new_line('a')// &
            '    res = "a" // "b"'//new_line('a')// &
            '  end function helper'//new_line('a')// &
            'end program main'

        test_function_returns_concatenated_deferred_character = expect_output( &
            source, ' abcd'//new_line('a'), &
            '/tmp/ffc_deferred_char_func_concat')
    end function test_function_returns_concatenated_deferred_character

    logical function test_function_returns_input_with_suffix()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  character(len=:), allocatable :: r, s'//new_line('a')// &
            '  r = append_bang("hi")'//new_line('a')// &
            '  s = r // "x"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function append_bang(arg) result(res)'//new_line('a')// &
            '    character(len=*), intent(in) :: arg'//new_line('a')// &
            '    character(len=:), allocatable :: res'//new_line('a')// &
            '    res = arg // "!"'//new_line('a')// &
            '  end function append_bang'//new_line('a')// &
            'end program main'

        test_function_returns_input_with_suffix = expect_output( &
            source, ' hi!x'//new_line('a'), &
            '/tmp/ffc_deferred_char_func_suffix')
    end function test_function_returns_input_with_suffix

    logical function test_function_result_prints_directly()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  print *, greet("world")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function greet(name) result(res)'//new_line('a')// &
            '    character(len=*), intent(in) :: name'//new_line('a')// &
            '    character(len=:), allocatable :: res'//new_line('a')// &
            '    res = "hello, " // name'//new_line('a')// &
            '  end function greet'//new_line('a')// &
            'end program main'

        test_function_result_prints_directly = expect_output( &
            source, ' hello, world'//new_line('a'), &
            '/tmp/ffc_deferred_char_func_print')
    end function test_function_result_prints_directly

end program test_session_deferred_char_compiler
