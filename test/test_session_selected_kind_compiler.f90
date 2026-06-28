program test_session_selected_kind
    use ffc_test_support, only: expect_exit_status, expect_output, &
        expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session selected_*_kind constant-folding test ==='

    all_passed = .true.
    if (.not. test_selected_int_kind_print()) all_passed = .false.
    if (.not. test_selected_int_kind_unavailable()) all_passed = .false.
    if (.not. test_selected_real_kind_one_arg()) all_passed = .false.
    if (.not. test_selected_real_kind_two_args()) all_passed = .false.
    if (.not. test_selected_real_kind_error_codes()) all_passed = .false.
    if (.not. test_selected_int_kind_in_parameter()) all_passed = .false.
    if (.not. test_selected_int_kind_constant_expr()) all_passed = .false.
    if (.not. test_selected_int_kind_nonconstant_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: selected_*_kind fold to gfortran kind numbers'

contains

    ! selected_int_kind folds to the gfortran kind selecting the requested
    ! decimal range: r<=2 -> 1, r<=4 -> 2, r<=9 -> 4, r<=18 -> 8, r<=38 -> 16.
    logical function test_selected_int_kind_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, selected_int_kind(2)'// &
            new_line('a')// &
            '  print *, selected_int_kind(9)'// &
            new_line('a')// &
            '  print *, selected_int_kind(18)'// &
            new_line('a')// &
            'end program main'

        test_selected_int_kind_print = expect_output( &
            source, &
            '           1'//new_line('a')// &
            '           4'//new_line('a')// &
            '           8'//new_line('a'), &
            '/tmp/ffc_session_selected_int_kind_print')
    end function test_selected_int_kind_print

    ! No available integer kind reaches 39 decimal digits: result is -1.
    logical function test_selected_int_kind_unavailable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, selected_int_kind(39)'// &
            new_line('a')// &
            'end program main'

        test_selected_int_kind_unavailable = expect_output( &
            source, '          -1'//new_line('a'), &
            '/tmp/ffc_session_selected_int_kind_unavail')
    end function test_selected_int_kind_unavailable

    ! selected_real_kind(p): smallest real kind with >= p decimal digits.
    logical function test_selected_real_kind_one_arg()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, selected_real_kind(6)'// &
            new_line('a')// &
            '  print *, selected_real_kind(15)'// &
            new_line('a')// &
            'end program main'

        test_selected_real_kind_one_arg = expect_output( &
            source, &
            '           4'//new_line('a')// &
            '           8'//new_line('a'), &
            '/tmp/ffc_session_selected_real_kind_one')
    end function test_selected_real_kind_one_arg

    ! selected_real_kind(p, r) requires both precision and range to be met.
    logical function test_selected_real_kind_two_args()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, selected_real_kind(6, 30)'// &
            new_line('a')// &
            '  print *, selected_real_kind(15, 307)'// &
            new_line('a')// &
            'end program main'

        test_selected_real_kind_two_args = expect_output( &
            source, &
            '           4'//new_line('a')// &
            '           8'//new_line('a'), &
            '/tmp/ffc_session_selected_real_kind_two')
    end function test_selected_real_kind_two_args

    ! Standard error codes: -1 precision unmet, -2 range unmet, -3 both unmet.
    logical function test_selected_real_kind_error_codes()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, selected_real_kind(40)'// &
            new_line('a')// &
            '  print *, selected_real_kind(0, 5000)'// &
            new_line('a')// &
            '  print *, selected_real_kind(40, 5000)'// &
            new_line('a')// &
            'end program main'

        test_selected_real_kind_error_codes = expect_output( &
            source, &
            '          -1'//new_line('a')// &
            '          -2'//new_line('a')// &
            '          -3'//new_line('a'), &
            '/tmp/ffc_session_selected_real_kind_err')
    end function test_selected_real_kind_error_codes

    ! The fold result is usable as a compile-time integer parameter value.
    logical function test_selected_int_kind_in_parameter()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: ik = '// &
            'selected_int_kind(9)'//new_line('a')// &
            '  stop ik'//new_line('a')// &
            'end program main'

        test_selected_int_kind_in_parameter = expect_exit_status( &
            source, 4, '/tmp/ffc_session_selected_int_kind_param')
    end function test_selected_int_kind_in_parameter

    ! The argument may itself be a compile-time integer expression.
    logical function test_selected_int_kind_constant_expr()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, selected_int_kind(4 + 5)'// &
            new_line('a')// &
            'end program main'

        test_selected_int_kind_constant_expr = expect_output( &
            source, '           4'//new_line('a'), &
            '/tmp/ffc_session_selected_int_kind_expr')
    end function test_selected_int_kind_constant_expr

    ! A runtime variable argument is not a constant expression and is rejected.
    logical function test_selected_int_kind_nonconstant_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 9'//new_line('a')// &
            '  print *, selected_int_kind(n)'// &
            new_line('a')// &
            'end program main'

        test_selected_int_kind_nonconstant_diagnostic = expect_error_contains( &
            source, &
            'selected_int_kind requires a compile-time integer argument', &
            '/tmp/ffc_session_selected_int_kind_nonconst')
    end function test_selected_int_kind_nonconstant_diagnostic

end program test_session_selected_kind
