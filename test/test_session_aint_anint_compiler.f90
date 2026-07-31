program test_session_aint_anint_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session aint/anint compiler test ==='

    all_passed = .true.
    if (.not. test_aint_truncates_positive()) all_passed = .false.
    if (.not. test_aint_truncates_negative()) all_passed = .false.
    if (.not. test_anint_rounds_half_up()) all_passed = .false.
    if (.not. test_anint_rounds_half_down()) all_passed = .false.
    if (.not. test_aint_double_precision()) all_passed = .false.
    if (.not. test_aint_kind_widens_result()) all_passed = .false.
    if (.not. test_anint_kind_widens_result()) all_passed = .false.
    if (.not. test_aint_kind_narrows_result()) all_passed = .false.
    if (.not. test_aint_anint_arrays()) all_passed = .false.
    if (.not. test_unsupported_kind_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: aint/anint lower through direct LIRIC session'

contains

    logical function test_aint_truncates_positive()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = 3.7'//new_line('a')// &
            '  print *, aint(x)'//new_line('a')// &
            'end program main'

        test_aint_truncates_positive = expect_output( &
            source, '   3.00000000    '//new_line('a'), '/tmp/ffc_aint_pos')
    end function test_aint_truncates_positive

    logical function test_aint_truncates_negative()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = -3.7'//new_line('a')// &
            '  print *, aint(x)'//new_line('a')// &
            'end program main'

        test_aint_truncates_negative = expect_output( &
            source, '  -3.00000000    '//new_line('a'), '/tmp/ffc_aint_neg')
    end function test_aint_truncates_negative

    logical function test_anint_rounds_half_up()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = 3.5'//new_line('a')// &
            '  print *, anint(x)'//new_line('a')// &
            'end program main'

        test_anint_rounds_half_up = expect_output( &
            source, '   4.00000000    '//new_line('a'), '/tmp/ffc_anint_up')
    end function test_anint_rounds_half_up

    logical function test_anint_rounds_half_down()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = -3.5'//new_line('a')// &
            '  print *, anint(x)'//new_line('a')// &
            'end program main'

        test_anint_rounds_half_down = expect_output( &
            source, '  -4.00000000    '//new_line('a'), '/tmp/ffc_anint_down')
    end function test_anint_rounds_half_down

    logical function test_aint_double_precision()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  double precision :: x = 2.9d0'//new_line('a')// &
            '  print *, aint(x), anint(x)'//new_line('a')// &
            'end program main'

        test_aint_double_precision = expect_output( &
            source, &
            '   2.0000000000000000        3.0000000000000000     '// &
            new_line('a'), '/tmp/ffc_aint_dp')
    end function test_aint_double_precision

    logical function test_aint_kind_widens_result()
        ! aint(x, 8) on a real(4) argument yields a real(8) result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = 3.7'//new_line('a')// &
            '  print *, aint(x, 8)'//new_line('a')// &
            'end program main'

        test_aint_kind_widens_result = expect_output( &
            source, '   3.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_aint_kind8')
    end function test_aint_kind_widens_result

    logical function test_anint_kind_widens_result()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = 3.7'//new_line('a')// &
            '  print *, anint(x, kind=8)'//new_line('a')// &
            'end program main'

        test_anint_kind_widens_result = expect_output( &
            source, '   4.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_anint_kind8')
    end function test_anint_kind_widens_result

    logical function test_aint_kind_narrows_result()
        ! aint(d, 4) on a real(8) argument yields a real(4) result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  double precision :: d = -2.75d0'//new_line('a')// &
            '  print *, aint(d, 4), anint(d, 4)'//new_line('a')// &
            'end program main'

        test_aint_kind_narrows_result = expect_output( &
            source, '  -2.00000000      -3.00000000    '//new_line('a'), &
            '/tmp/ffc_aint_kind4')
    end function test_aint_kind_narrows_result

    logical function test_aint_anint_arrays()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3) = [1.7, -1.7, 2.5]'//new_line('a')// &
            '  print *, aint(a)'//new_line('a')// &
            '  print *, anint(a)'//new_line('a')// &
            'end program main'

        test_aint_anint_arrays = expect_output( &
            source, &
            '   1.00000000      -1.00000000       2.00000000    '// &
            new_line('a')// &
            '   2.00000000      -2.00000000       3.00000000    '// &
            new_line('a'), '/tmp/ffc_aint_array')
    end function test_aint_anint_arrays

    logical function test_unsupported_kind_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = 1.5'//new_line('a')// &
            '  print *, aint(x, 3)'//new_line('a')// &
            'end program main'

        test_unsupported_kind_rejected = expect_error_contains( &
            source, 'aint result kind', '/tmp/ffc_aint_badkind')
    end function test_unsupported_kind_rejected

end program test_session_aint_anint_compiler
