program test_session_deferred_char_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== deferred-char function result compiler tests ==='

    all_passed = .true.
    if (.not. test_declare_deferred_character_compiles()) all_passed = .false.
    if (.not. test_two_deferred_characters_get_independent_descriptors()) &
        all_passed = .false.
    if (.not. test_function_returns_concatenated_deferred_character()) &
        all_passed = .false.
    if (.not. test_function_returns_input_with_suffix()) all_passed = .false.
    if (.not. test_function_result_prints_directly()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: deferred-char function result ABI'

contains

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
