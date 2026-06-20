program test_session_aint_anint_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session aint/anint compiler test ==='

    all_passed = .true.
    if (.not. test_aint_truncates_positive()) all_passed = .false.
    if (.not. test_aint_truncates_negative()) all_passed = .false.
    if (.not. test_anint_rounds_half_up()) all_passed = .false.
    if (.not. test_anint_rounds_half_down()) all_passed = .false.
    if (.not. test_aint_double_precision()) all_passed = .false.

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

end program test_session_aint_anint_compiler
