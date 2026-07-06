program test_session_character_fixed_ops_compiler
    ! Fixed-length character operations: var-to-var assignment with blank
    ! padding/truncation, fixed-length dummy arguments, lexical comparisons,
    ! SELECT CASE character ranges, and named constants with a declared
    ! length or a concatenation of earlier named constants.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== character fixed-length ops compiler test ==='

    all_passed = .true.
    if (.not. test_assignment_pads_shorter_source()) all_passed = .false.
    if (.not. test_assignment_truncates_longer_source()) all_passed = .false.
    if (.not. test_fixed_dummy_keeps_declared_length()) all_passed = .false.
    if (.not. test_equality_comparison()) all_passed = .false.
    if (.not. test_lexical_ordering_with_blank_padding()) all_passed = .false.
    if (.not. test_select_case_character_range()) all_passed = .false.
    if (.not. test_named_constant_declared_length()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character fixed-length ops lower through direct LIRIC'

contains

    logical function test_assignment_pads_shorter_source()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=2) :: short'//new_line('a')// &
            '  character(len=6) :: dest'//new_line('a')// &
            '  short = "hi"'//new_line('a')// &
            '  dest = short'//new_line('a')// &
            '  print *, "[", dest, "]"'//new_line('a')// &
            'end program main'

        test_assignment_pads_shorter_source = expect_output( &
            source, ' [hi    ]'//new_line('a'), &
            '/tmp/ffc_session_char_fixed_ops_pad_test')
    end function test_assignment_pads_shorter_source

    logical function test_assignment_truncates_longer_source()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=8) :: long'//new_line('a')// &
            '  character(len=3) :: dest'//new_line('a')// &
            '  long = "helloworld"'//new_line('a')// &
            '  dest = long'//new_line('a')// &
            '  print *, "[", dest, "]"'//new_line('a')// &
            'end program main'

        test_assignment_truncates_longer_source = expect_output( &
            source, ' [hel]'//new_line('a'), &
            '/tmp/ffc_session_char_fixed_ops_trunc_test')
    end function test_assignment_truncates_longer_source

    logical function test_fixed_dummy_keeps_declared_length()
        ! A character(len=1) dummy sees only the first character of a
        ! longer actual, not the actual's full runtime length.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: test_char'//new_line('a')// &
            '  logical :: r'//new_line('a')// &
            '  test_char = "2021"'//new_line('a')// &
            '  r = isdigit(test_char)'//new_line('a')// &
            '  if (.not. r) error stop 1'//new_line('a')// &
            'contains'//new_line('a')// &
            '  logical function isdigit(c)'//new_line('a')// &
            '    character(len=1), intent(in) :: c'//new_line('a')// &
            '    isdigit = index("12", c) > 0'//new_line('a')// &
            '  end function isdigit'//new_line('a')// &
            'end program main'

        test_fixed_dummy_keeps_declared_length = expect_exit_status( &
            source, 0, '/tmp/ffc_session_char_fixed_dummy_test')
    end function test_fixed_dummy_keeps_declared_length

    logical function test_equality_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: a, b'//new_line('a')// &
            '  a = "abc"'//new_line('a')// &
            '  b = "abd"'//new_line('a')// &
            '  if (a == "abc") print *, "eq ok"'//new_line('a')// &
            '  if (a /= b) print *, "ne ok"'//new_line('a')// &
            'end program main'

        test_equality_comparison = expect_output( &
            source, ' eq ok'//new_line('a')//' ne ok'//new_line('a'), &
            '/tmp/ffc_session_char_eq_test')
    end function test_equality_comparison

    logical function test_lexical_ordering_with_blank_padding()
        ! Fortran pads the shorter operand with blanks before comparing, so
        ! "ab" == "ab  " and "ab" < "abc " but "ab" > "ab" // a control byte
        ! (a byte below blank in the tail sorts lower than the padded blank).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=2) :: short'//new_line('a')// &
            '  character(len=4) :: long'//new_line('a')// &
            '  short = "ab"'//new_line('a')// &
            '  long = "ab  "'//new_line('a')// &
            '  if (short == long) print *, "padded eq"'//new_line('a')// &
            '  long = "abc "'//new_line('a')// &
            '  if (short < long) print *, "lt ok"'//new_line('a')// &
            '  long = "ab"//achar(9)//" "'//new_line('a')// &
            '  if (short > long) print *, "gt ok"'//new_line('a')// &
            'end program main'

        test_lexical_ordering_with_blank_padding = expect_output( &
            source, ' padded eq'//new_line('a')//' lt ok'//new_line('a')// &
            ' gt ok'//new_line('a'), &
            '/tmp/ffc_session_char_lexical_test')
    end function test_lexical_ordering_with_blank_padding

    logical function test_select_case_character_range()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=1) :: c'//new_line('a')// &
            '  c = "e"'//new_line('a')// &
            '  select case (c)'//new_line('a')// &
            '  case ("a":"j")'//new_line('a')// &
            '    print *, "a-j"'//new_line('a')// &
            '  case ("l":"p")'//new_line('a')// &
            '    print *, "l-p"'//new_line('a')// &
            '  case default'//new_line('a')// &
            '    print *, "other"'//new_line('a')// &
            '  end select'//new_line('a')// &
            'end program main'

        test_select_case_character_range = expect_output( &
            source, ' a-j'//new_line('a'), &
            '/tmp/ffc_session_char_case_range_test')
    end function test_select_case_character_range

    logical function test_named_constant_declared_length()
        ! character(len=N), parameter pads/truncates to N; a later constant's
        ! initializer may concatenate earlier named constants, not just
        ! literals.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=8), parameter :: x_pad = "apple"'//new_line('a')// &
            '  character(len=10), parameter :: y_pad = "Ball"'//new_line('a')// &
            '  character(len=30), parameter :: z_pad = '// &
            'x_pad // y_pad // x_pad'//new_line('a')// &
            '  if (len(x_pad) /= 8) error stop 1'//new_line('a')// &
            '  if (len(z_pad) /= 30) error stop 2'//new_line('a')// &
            '  print *, trim(x_pad), "|", trim(z_pad)'//new_line('a')// &
            'end program main'

        test_named_constant_declared_length = expect_output( &
            source, ' apple|apple   Ball      apple'//new_line('a'), &
            '/tmp/ffc_session_char_const_len_test')
    end function test_named_constant_declared_length

end program test_session_character_fixed_ops_compiler
