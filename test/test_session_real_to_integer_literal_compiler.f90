program test_session_real_to_integer_literal_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    print *, '=== direct session real-to-integer literal compiler test ==='

    ! Test real literals in integer contexts
    if (.not. test_simple_real_to_int()) error stop 'FAIL: simple_real_to_int'
    if (.not. test_real_dot_no_digit_after()) error stop 'FAIL: real_dot'
    if (.not. test_real_d_exponent()) error stop 'FAIL: real_d_exp'
    if (.not. test_real_with_kind_suffix()) error stop 'FAIL: real_kind'
    if (.not. test_boz_integer()) error stop 'FAIL: boz_int'
    if (.not. test_boz_real()) error stop 'FAIL: boz_real'
    if (.not. test_exponentiation_real()) error stop 'FAIL: exponent'

    print *, 'PASS: real-to-integer literal forms lowers through direct LIRIC'

contains

    logical function test_simple_real_to_int()
        ! When assigning a simple real literal to an integer, truncate decimal.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: y(3)'//new_line('a')// &
            '  real :: x(3)'//new_line('a')// &
            '  x = 1'//new_line('a')// &
            '  y = 2.0'//new_line('a')// &
            '  print *, y'//new_line('a')// &
            'end program main'

        test_simple_real_to_int = expect_output(source, &
            '           2           2           2'//new_line('a'), &
            '/tmp/ffc_real_to_int_simple_test')
    end function test_simple_real_to_int

    logical function test_real_dot_no_digit_after()
        ! Real literal with decimal point but no digit after (e.g., 2. means 2.0).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2.'//new_line('a')// &
            '  if (x /= 2) error stop'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_real_dot_no_digit_after = expect_exit_status(source, 0, &
            '/tmp/ffc_real_dot_test')
    end function test_real_dot_no_digit_after

    logical function test_real_d_exponent()
        ! Double precision real literal (d or D exponent).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = int(1.0D0)'//new_line('a')// &
            '  if (x /= 1) error stop'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_real_d_exponent = expect_exit_status(source, 0, &
            '/tmp/ffc_real_d_exp_test')
    end function test_real_d_exponent

    logical function test_real_with_kind_suffix()
        ! Real literal with kind suffix (_8, _dp, etc.).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = int(5.6_8)'//new_line('a')// &
            '  if (x /= 5) error stop'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_real_with_kind_suffix = expect_exit_status(source, 0, &
            '/tmp/ffc_real_kind_test')
    end function test_real_with_kind_suffix

    logical function test_boz_integer()
        ! Binary, Octal, or Zonal (hex) literals in integer context.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: b, o, h'//new_line('a')// &
            '  b = int(b''01011101'')'//new_line('a')// &
            '  o = int(o''2347'')'//new_line('a')// &
            '  h = int(z''ABC'')'//new_line('a')// &
            '  if (b /= 93) error stop'//new_line('a')// &
            '  if (o /= 1255) error stop'//new_line('a')// &
            '  if (h /= 2748) error stop'//new_line('a')// &
            '  print *, b, o, h'//new_line('a')// &
            'end program main'

        test_boz_integer = expect_exit_status(source, 0, &
            '/tmp/ffc_boz_integer_test')
    end function test_boz_integer

    logical function test_boz_real()
        ! REAL() applied to a BOZ literal reinterprets its bit pattern as the
        ! target kind's representation (F2008 13.7.128) rather than converting
        ! the magnitude numerically: real(b'01101') is the f32 bit pattern of
        ! integer 13, a tiny denormal (gfortran-verified: 1.82168800E-44), not
        ! the numeric conversion 13.0.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  r = real(b''01101'')'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'

        test_boz_real = expect_output(source, '   1.82168800E-44'//new_line('a'), &
            '/tmp/ffc_boz_real_test')
    end function test_boz_real

    logical function test_exponentiation_real()
        ! Exponentiation with real operands (both base and exponent real).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 2**4'//new_line('a')// &
            '  if (x /= 16) error stop'//new_line('a')// &
            '  x = 2.**4.'//new_line('a')// &
            '  if (x /= 16) error stop'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_exponentiation_real = expect_exit_status(source, 0, &
            '/tmp/ffc_exponent_real_test')
    end function test_exponentiation_real

end program test_session_real_to_integer_literal_compiler
