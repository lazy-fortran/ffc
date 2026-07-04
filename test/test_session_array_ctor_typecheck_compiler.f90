program test_session_array_ctor_typecheck_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session typed array-constructor type-mismatch test ==='

    all_passed = .true.
    if (.not. test_char_in_integer_ctor_rejected()) all_passed = .false.
    if (.not. test_char_in_real_ctor_rejected()) all_passed = .false.
    if (.not. test_integer_in_char_ctor_rejected()) all_passed = .false.
    if (.not. test_logical_in_integer_ctor_rejected()) all_passed = .false.
    if (.not. test_char_in_nested_ctor_rejected()) all_passed = .false.
    if (.not. test_integer_ctor_still_runs()) all_passed = .false.
    if (.not. test_real_ctor_numeric_mix_still_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: mismatched typed array-constructor elements are rejected'

contains

    logical function test_char_in_integer_ctor_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, [integer :: 1, "x"]'//new_line('a')// &
            'end program main'

        test_char_in_integer_ctor_rejected = expect_error_contains( &
            source, 'array constructor', '/tmp/ffc_ctor_char_int')
    end function test_char_in_integer_ctor_rejected

    logical function test_char_in_real_ctor_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, [real :: 1.0, "x"]'//new_line('a')// &
            'end program main'

        test_char_in_real_ctor_rejected = expect_error_contains( &
            source, 'array constructor', '/tmp/ffc_ctor_char_real')
    end function test_char_in_real_ctor_rejected

    logical function test_integer_in_char_ctor_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, [character(len=1) :: "a", 2]'//new_line('a')// &
            'end program main'

        test_integer_in_char_ctor_rejected = expect_error_contains( &
            source, 'array constructor', '/tmp/ffc_ctor_int_char')
    end function test_integer_in_char_ctor_rejected

    logical function test_logical_in_integer_ctor_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, [integer :: 1, .true.]'//new_line('a')// &
            'end program main'

        test_logical_in_integer_ctor_rejected = expect_error_contains( &
            source, 'array constructor', '/tmp/ffc_ctor_logical_int')
    end function test_logical_in_integer_ctor_rejected

    logical function test_char_in_nested_ctor_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, [integer :: 1, [integer :: 2, "x"]]'//new_line('a')// &
            'end program main'

        test_char_in_nested_ctor_rejected = expect_error_contains( &
            source, 'array constructor', '/tmp/ffc_ctor_nested_char')
    end function test_char_in_nested_ctor_rejected

    logical function test_integer_ctor_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  a = [integer :: 1, 2, 3]'//new_line('a')// &
            '  if (a(2) == 2) stop 5'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_integer_ctor_still_runs = expect_exit_status( &
            source, 5, '/tmp/ffc_ctor_int_valid')
    end function test_integer_ctor_still_runs

    logical function test_real_ctor_numeric_mix_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2)'//new_line('a')// &
            '  a = [integer :: 1, 2.0]'//new_line('a')// &
            '  if (a(2) == 2) stop 6'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_real_ctor_numeric_mix_still_runs = expect_exit_status( &
            source, 6, '/tmp/ffc_ctor_numeric_mix')
    end function test_real_ctor_numeric_mix_still_runs

end program test_session_array_ctor_typecheck_compiler
