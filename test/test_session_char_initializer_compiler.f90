program test_session_char_initializer_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character initializer compiler test ==='

    all_passed = .true.
    if (.not. test_len_trim_of_initialized()) all_passed = .false.
    if (.not. test_trim_of_initialized_prints()) all_passed = .false.
    if (.not. test_concat_literal_then_trim()) all_passed = .false.
    if (.not. test_concat_trim_then_literal()) all_passed = .false.
    if (.not. test_concat_two_trims()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character initializers and concatenation lower correctly'

contains

    logical function test_len_trim_of_initialized()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: s = "hello  "'//new_line('a')// &
            '  stop len_trim(s)'//new_line('a')// &
            'end program main'

        test_len_trim_of_initialized = expect_exit_status( &
            source, 5, '/tmp/ffc_char_init_len_trim')
    end function test_len_trim_of_initialized

    logical function test_trim_of_initialized_prints()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: s = "hello  "'//new_line('a')// &
            '  print *, trim(s)'//new_line('a')// &
            'end program main'

        test_trim_of_initialized_prints = expect_output( &
            source, ' hello'//new_line('a'), '/tmp/ffc_char_init_trim')
    end function test_trim_of_initialized_prints

    logical function test_concat_literal_then_trim()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: s = "hi  "'//new_line('a')// &
            '  print *, "["//trim(s)'//new_line('a')// &
            'end program main'

        test_concat_literal_then_trim = expect_output( &
            source, ' [hi'//new_line('a'), '/tmp/ffc_char_concat_lit_trim')
    end function test_concat_literal_then_trim

    logical function test_concat_trim_then_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: s = "hi  "'//new_line('a')// &
            '  print *, trim(s)//"]"'//new_line('a')// &
            'end program main'

        test_concat_trim_then_literal = expect_output( &
            source, ' hi]'//new_line('a'), '/tmp/ffc_char_concat_trim_lit')
    end function test_concat_trim_then_literal

    logical function test_concat_two_trims()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=10) :: a = "ab  ", b = "cd  "'//new_line('a')// &
            '  print *, trim(a)//trim(b)'//new_line('a')// &
            'end program main'

        test_concat_two_trims = expect_output( &
            source, ' abcd'//new_line('a'), '/tmp/ffc_char_concat_two_trims')
    end function test_concat_two_trims

end program test_session_char_initializer_compiler
