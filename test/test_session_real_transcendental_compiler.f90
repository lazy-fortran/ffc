program test_session_real_transcendental_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session real transcendental compiler test ==='

    all_passed = .true.
    if (.not. test_sqrt_four()) all_passed = .false.
    if (.not. test_sin_zero()) all_passed = .false.
    if (.not. test_cos_zero()) all_passed = .false.
    if (.not. test_exp_zero()) all_passed = .false.
    if (.not. test_log_one()) all_passed = .false.
    if (.not. test_atan2_one_one()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real transcendental intrinsics lower through libm'

contains

    logical function check(expr, expected, stem)
        character(len=*), intent(in) :: expr
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem

        check = expect_output( &
            'program main'//new_line('a')// &
            'real :: x'//new_line('a')// &
            'x = '//expr//new_line('a')// &
            'print *, x'//new_line('a')// &
            'end program main', expected//new_line('a'), stem)
    end function check

    logical function test_sqrt_four()
        test_sqrt_four = check('sqrt(4.0)', '   2.00000000    ', &
            '/tmp/ffc_session_sqrt_test')
    end function test_sqrt_four

    logical function test_sin_zero()
        test_sin_zero = check('sin(0.0)', '   0.00000000    ', &
            '/tmp/ffc_session_sin_test')
    end function test_sin_zero

    logical function test_cos_zero()
        test_cos_zero = check('cos(0.0)', '   1.00000000    ', &
            '/tmp/ffc_session_cos_test')
    end function test_cos_zero

    logical function test_exp_zero()
        test_exp_zero = check('exp(0.0)', '   1.00000000    ', &
            '/tmp/ffc_session_exp_test')
    end function test_exp_zero

    logical function test_log_one()
        test_log_one = check('log(1.0)', '   0.00000000    ', &
            '/tmp/ffc_session_log_test')
    end function test_log_one

    logical function test_atan2_one_one()
        ! atan2(1.0, 1.0) == pi/4, printed in gfortran real(4) list-directed form
        test_atan2_one_one = check('atan2(1.0, 1.0)', &
            '  0.785398185    ', &
            '/tmp/ffc_session_atan2_test')
    end function test_atan2_one_one

end program test_session_real_transcendental_compiler
