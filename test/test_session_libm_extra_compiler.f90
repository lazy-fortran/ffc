program test_session_libm_extra
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session extended libm intrinsic compiler test ==='

    all_passed = .true.
    if (.not. test_sinh_zero()) all_passed = .false.
    if (.not. test_cosh_zero()) all_passed = .false.
    if (.not. test_tanh_zero()) all_passed = .false.
    if (.not. test_asin_one()) all_passed = .false.
    if (.not. test_acos_one()) all_passed = .false.
    if (.not. test_asinh_zero()) all_passed = .false.
    if (.not. test_acosh_one()) all_passed = .false.
    if (.not. test_atanh_zero()) all_passed = .false.
    if (.not. test_log10_thousand()) all_passed = .false.
    if (.not. test_erf_zero()) all_passed = .false.
    if (.not. test_erfc_zero()) all_passed = .false.
    if (.not. test_gamma_five()) all_passed = .false.
    if (.not. test_log_gamma_one()) all_passed = .false.
    if (.not. test_hypot_three_four()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: extended libm intrinsics lower through libm'

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

    logical function test_sinh_zero()
        test_sinh_zero = check('sinh(0.0)', '   0.00000000    ', &
                               '/tmp/ffc_session_sinh_test')
    end function test_sinh_zero

    logical function test_cosh_zero()
        test_cosh_zero = check('cosh(0.0)', '   1.00000000    ', &
                               '/tmp/ffc_session_cosh_test')
    end function test_cosh_zero

    logical function test_tanh_zero()
        test_tanh_zero = check('tanh(0.0)', '   0.00000000    ', &
                               '/tmp/ffc_session_tanh_test')
    end function test_tanh_zero

    logical function test_asin_one()
        ! asin of a default real lowers to asinf and matches gfortran at runtime
        test_asin_one = check('asin(1.0)', '   1.57079637    ', &
                              '/tmp/ffc_session_asin_test')
    end function test_asin_one

    logical function test_acos_one()
        test_acos_one = check('acos(1.0)', '   0.00000000    ', &
                              '/tmp/ffc_session_acos_test')
    end function test_acos_one

    logical function test_asinh_zero()
        test_asinh_zero = check('asinh(0.0)', '   0.00000000    ', &
                                '/tmp/ffc_session_asinh_test')
    end function test_asinh_zero

    logical function test_acosh_one()
        test_acosh_one = check('acosh(1.0)', '   0.00000000    ', &
                               '/tmp/ffc_session_acosh_test')
    end function test_acosh_one

    logical function test_atanh_zero()
        test_atanh_zero = check('atanh(0.0)', '   0.00000000    ', &
                                '/tmp/ffc_session_atanh_test')
    end function test_atanh_zero

    logical function test_log10_thousand()
        test_log10_thousand = check('log10(1000.0)', &
                                    '   3.00000000    ', &
                                    '/tmp/ffc_session_log10_test')
    end function test_log10_thousand

    logical function test_erf_zero()
        test_erf_zero = check('erf(0.0)', '   0.00000000    ', &
                              '/tmp/ffc_session_erf_test')
    end function test_erf_zero

    logical function test_erfc_zero()
        test_erfc_zero = check('erfc(0.0)', '   1.00000000    ', &
                               '/tmp/ffc_session_erfc_test')
    end function test_erfc_zero

    logical function test_gamma_five()
        ! gamma(5.0) == 4! == 24, printed in gfortran real(4) list-directed form
        test_gamma_five = check('gamma(5.0)', '   24.0000000    ', &
                                '/tmp/ffc_session_gamma_test')
    end function test_gamma_five

    logical function test_log_gamma_one()
        test_log_gamma_one = check('log_gamma(1.0)', &
                                   '   0.00000000    ', &
                                   '/tmp/ffc_session_log_gamma_test')
    end function test_log_gamma_one

    logical function test_hypot_three_four()
        test_hypot_three_four = check('hypot(3.0, 4.0)', &
                                      '   5.00000000    ', &
                                      '/tmp/ffc_session_hypot_test')
    end function test_hypot_three_four

end program test_session_libm_extra
