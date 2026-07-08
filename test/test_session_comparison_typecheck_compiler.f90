program test_session_comparison_typecheck_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session comparison operand type-mismatch test ==='

    all_passed = .true.
    if (.not. test_logical_vs_integer_rejected()) all_passed = .false.
    if (.not. test_character_vs_integer_rejected()) all_passed = .false.
    if (.not. test_integer_vs_logical_rejected()) all_passed = .false.
    if (.not. test_logical_vs_character_rejected()) all_passed = .false.
    if (.not. test_hollerith_vs_integer_rejected()) all_passed = .false.
    if (.not. test_integer_vs_hollerith_rejected()) all_passed = .false.
    if (.not. test_integer_comparison_still_runs()) all_passed = .false.
    if (.not. test_real_comparison_still_runs()) all_passed = .false.
    if (.not. test_character_comparison_still_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: mismatched comparison operands are rejected'

contains

    logical function test_logical_vs_integer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: b'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  b = .true.'//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  if (b == i) stop 1'//new_line('a')// &
            'end program main'

        test_logical_vs_integer_rejected = expect_error_contains( &
            source, 'mismatched types', '/tmp/ffc_cmp_logical_int')
    end function test_logical_vs_integer_rejected

    logical function test_character_vs_integer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: c'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            "  c = 'abc'"//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  if (c == i) stop 1'//new_line('a')// &
            'end program main'

        test_character_vs_integer_rejected = expect_error_contains( &
            source, 'mismatched types', '/tmp/ffc_cmp_char_int')
    end function test_character_vs_integer_rejected

    logical function test_integer_vs_logical_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: b'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  b = .false.'//new_line('a')// &
            '  i = 7'//new_line('a')// &
            '  if (i > b) stop 1'//new_line('a')// &
            'end program main'

        test_integer_vs_logical_rejected = expect_error_contains( &
            source, 'mismatched types', '/tmp/ffc_cmp_int_logical')
    end function test_integer_vs_logical_rejected

    logical function test_logical_vs_character_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: b'//new_line('a')// &
            '  character(len=2) :: c'//new_line('a')// &
            '  b = .true.'//new_line('a')// &
            "  c = 'ab'"//new_line('a')// &
            '  if (b == c) stop 1'//new_line('a')// &
            'end program main'

        test_logical_vs_character_rejected = expect_error_contains( &
            source, 'mismatched types', '/tmp/ffc_cmp_logical_char')
    end function test_logical_vs_character_rejected

    logical function test_hollerith_vs_integer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 1'//new_line('a')// &
            '  if (4HABCD == i) stop 1'//new_line('a')// &
            'end program main'

        test_hollerith_vs_integer_rejected = expect_error_contains( &
            source, 'mismatched types', '/tmp/ffc_cmp_hollerith_int')
    end function test_hollerith_vs_integer_rejected

    logical function test_integer_vs_hollerith_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 1'//new_line('a')// &
            '  if (i /= 4HABCD) stop 1'//new_line('a')// &
            'end program main'

        test_integer_vs_hollerith_rejected = expect_error_contains( &
            source, 'mismatched types', '/tmp/ffc_cmp_int_hollerith')
    end function test_integer_vs_hollerith_rejected

    logical function test_integer_comparison_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 3'//new_line('a')// &
            '  if (i == 3) stop 7'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_integer_comparison_still_runs = expect_exit_status( &
            source, 7, '/tmp/ffc_cmp_int_valid')
    end function test_integer_comparison_still_runs

    logical function test_real_comparison_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  r = 2.5'//new_line('a')// &
            '  if (r > 1.0) stop 8'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_real_comparison_still_runs = expect_exit_status( &
            source, 8, '/tmp/ffc_cmp_real_valid')
    end function test_real_comparison_still_runs

    logical function test_character_comparison_still_runs()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=2) :: c'//new_line('a')// &
            "  c = 'ab'"//new_line('a')// &
            "  if (c == 'ab') stop 9"//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_character_comparison_still_runs = expect_exit_status( &
            source, 9, '/tmp/ffc_cmp_char_valid')
    end function test_character_comparison_still_runs

end program test_session_comparison_typecheck_compiler
