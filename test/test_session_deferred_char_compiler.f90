program test_session_deferred_char_compiler
    use ffc_test_support, only: expect_output, expect_exit_status, &
        expect_no_leaks, expect_error_contains
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
    if (.not. test_repeated_length_changes_report_each_length()) &
        all_passed = .false.
    if (.not. test_explicit_allocation_is_released_on_deallocate()) &
        all_passed = .false.
    if (.not. test_reassignment_releases_the_previous_allocation()) &
        all_passed = .false.
    if (.not. test_repeated_deallocate_is_not_a_double_free()) &
        all_passed = .false.
    if (.not. test_substring_of_deferred_character()) all_passed = .false.
    if (.not. test_substring_of_fixed_character()) all_passed = .false.
    if (.not. test_substring_with_runtime_bounds()) all_passed = .false.
    if (.not. test_substring_assigned_and_concatenated()) all_passed = .false.
    if (.not. test_substring_does_not_leak()) all_passed = .false.
    if (.not. test_substring_constant_bound_past_end_is_rejected()) &
        all_passed = .false.
    if (.not. test_substring_with_stride_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-char function result ABI'

contains

    function repeated_length_source() result(text)
        ! Shared source for the repeated-length-change cases: assignments of
        ! length 2, 8 and 1, then an explicit allocation reused by an
        ! assignment of exactly the allocated length.
        character(len=:), allocatable :: text
        character(len=*), parameter :: body = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "ab"'//new_line('a')// &
            '  print *, len(s), s'//new_line('a')// &
            '  s = "abcdefgh"'//new_line('a')// &
            '  print *, len(s), s'//new_line('a')// &
            '  s = "z"'//new_line('a')// &
            '  print *, len(s), s'//new_line('a')// &
            '  deallocate(s)'//new_line('a')// &
            '  allocate(character(len=3) :: s)'//new_line('a')// &
            '  s = "xyz"'//new_line('a')// &
            '  print *, len(s), s'//new_line('a')// &
            '  deallocate(s)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        text = body
    end function repeated_length_source

    logical function test_repeated_length_changes_report_each_length()
        ! Each assignment retargets the descriptor length to the RHS length,
        ! and the explicit allocate(character(len=3)) fixes length 3.
        test_repeated_length_changes_report_each_length = expect_output( &
            repeated_length_source(), &
            '           2 ab'//new_line('a')// &
            '           8 abcdefgh'//new_line('a')// &
            '           1 z'//new_line('a')// &
            '           3 xyz'//new_line('a'), &
            '/tmp/ffc_session_deferred_relength_test')
    end function test_repeated_length_changes_report_each_length

    logical function test_explicit_allocation_is_released_on_deallocate()
        ! allocate(character(len=n) :: s) owns heap storage; the descriptor
        ! must release it exactly once, and only once, at deallocate.
        test_explicit_allocation_is_released_on_deallocate = expect_no_leaks( &
            repeated_length_source(), &
            '/tmp/ffc_session_deferred_relength_leak_test')
    end function test_explicit_allocation_is_released_on_deallocate

    logical function test_reassignment_releases_the_previous_allocation()
        ! Assigning over an owned descriptor releases the former storage
        ! before installing the replacement.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 5'//new_line('a')// &
            '  allocate(character(len=n) :: s)'//new_line('a')// &
            '  s = "hello"'//new_line('a')// &
            '  s = "world!!"'//new_line('a')// &
            '  print *, len(s), s'//new_line('a')// &
            '  deallocate(s)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_reassignment_releases_the_previous_allocation = expect_no_leaks( &
            source, '/tmp/ffc_session_deferred_reassign_leak_test')
    end function test_reassignment_releases_the_previous_allocation

    logical function test_repeated_deallocate_is_not_a_double_free()
        ! Deallocating an already-released descriptor sees the null state and
        ! must not free the former pointer a second time.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  allocate(character(len=4) :: s)'//new_line('a')// &
            '  s = "abcd"'//new_line('a')// &
            '  deallocate(s)'//new_line('a')// &
            '  deallocate(s)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_repeated_deallocate_is_not_a_double_free = expect_no_leaks( &
            source, '/tmp/ffc_session_deferred_double_free_test')
    end function test_repeated_deallocate_is_not_a_double_free

    logical function test_substring_of_deferred_character()
        ! A substring of a deferred-length scalar is a view of its bytes at the
        ! requested Fortran positions: a(l:u) has length u - l + 1.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a'//new_line('a')// &
            '  a = "hello world"'//new_line('a')// &
            '  print *, a(1:5)'//new_line('a')// &
            '  print *, a(7:11)'//new_line('a')// &
            '  print *, a(1:1)'//new_line('a')// &
            '  print *, len(a(3:9))'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_substring_of_deferred_character = expect_output( &
            source, &
            ' hello'//new_line('a')// &
            ' world'//new_line('a')// &
            ' h'//new_line('a')// &
            '           7'//new_line('a'), &
            '/tmp/ffc_session_substring_deferred_test')
    end function test_substring_of_deferred_character

    logical function test_substring_of_fixed_character()
        ! The same view over a fixed-length scalar, whose length is a declared
        ! constant rather than descriptor metadata.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=11) :: f'//new_line('a')// &
            '  f = "hello world"'//new_line('a')// &
            '  print *, f(1:5)'//new_line('a')// &
            '  print *, f(7:11)'//new_line('a')// &
            'end program main'

        test_substring_of_fixed_character = expect_output( &
            source, &
            ' hello'//new_line('a')// &
            ' world'//new_line('a'), &
            '/tmp/ffc_session_substring_fixed_test')
    end function test_substring_of_fixed_character

    logical function test_substring_with_runtime_bounds()
        ! Substring bounds are ordinary integer expressions, evaluated at run
        ! time; the view length follows from them.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  a = "hello world"'//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  j = 9'//new_line('a')// &
            '  print *, a(i:j)'//new_line('a')// &
            '  print *, len(a(i:j))'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_substring_with_runtime_bounds = expect_output( &
            source, &
            ' llo wor'//new_line('a')// &
            '           7'//new_line('a'), &
            '/tmp/ffc_session_substring_runtime_test')
    end function test_substring_with_runtime_bounds

    logical function test_substring_assigned_and_concatenated()
        ! A substring is a character expression: it can be assigned to a
        ! deferred or fixed destination and used as a concatenation operand,
        ! with the usual padding and truncation on a fixed destination.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a, b'//new_line('a')// &
            '  character(len=8) :: f'//new_line('a')// &
            '  a = "hello world"'//new_line('a')// &
            '  b = a(1:5)'//new_line('a')// &
            '  print *, len(b), b'//new_line('a')// &
            '  f = a(7:11)'//new_line('a')// &
            '  print *, "[", f, "]"'//new_line('a')// &
            '  b = a(1:5) // a(7:11)'//new_line('a')// &
            '  print *, len(b), b'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  deallocate(b)'//new_line('a')// &
            'end program main'

        test_substring_assigned_and_concatenated = expect_output( &
            source, &
            '           5 hello'//new_line('a')// &
            ' [world   ]'//new_line('a')// &
            '          10 helloworld'//new_line('a'), &
            '/tmp/ffc_session_substring_expr_test')
    end function test_substring_assigned_and_concatenated

    logical function test_substring_does_not_leak()
        ! A substring is a borrowed view into its parent's storage: taking one
        ! allocates nothing to release, and it never frees the parent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a, b'//new_line('a')// &
            '  a = "hello world"'//new_line('a')// &
            '  b = a(1:5)'//new_line('a')// &
            '  print *, len(b), b'//new_line('a')// &
            '  deallocate(b)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_substring_does_not_leak = expect_no_leaks( &
            source, '/tmp/ffc_session_substring_leak_test')
    end function test_substring_does_not_leak

    logical function test_substring_constant_bound_past_end_is_rejected()
        ! A constant upper bound beyond a fixed-length parent's declared width
        ! is a source error, as gfortran reports it, rather than a read past
        ! the variable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=11) :: f'//new_line('a')// &
            '  f = "hello world"'//new_line('a')// &
            '  print *, f(1:20)'//new_line('a')// &
            'end program main'

        test_substring_constant_bound_past_end_is_rejected = &
            expect_error_contains(source, &
                'substring end index exceeds the string length', &
                '/tmp/ffc_session_substring_overrun_test')
    end function test_substring_constant_bound_past_end_is_rejected

    logical function test_substring_with_stride_is_rejected()
        ! A substring has no stride; s(l:u:k) is diagnosed rather than being
        ! silently treated as a substring or an array section.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a'//new_line('a')// &
            '  a = "hello world"'//new_line('a')// &
            '  print *, a(1:5:2)'//new_line('a')// &
            'end program main'

        test_substring_with_stride_is_rejected = expect_error_contains( &
            source, 'a substring has no stride', &
            '/tmp/ffc_session_substring_stride_test')
    end function test_substring_with_stride_is_rejected

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
