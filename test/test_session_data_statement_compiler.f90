program test_session_data_statement
    ! DATA statement lowering through direct LIRIC (#2349, #2251). DATA gives
    ! variables their initial value before execution, independent of textual
    ! position. Covers scalar and array initialisation, an executable
    ! assignment overriding DATA regardless of source order, a real array
    ! implied-do, and a hexadecimal BOZ constant.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session DATA statement compiler test ==='

    all_passed = .true.
    if (.not. test_scalar_init()) all_passed = .false.
    if (.not. test_array_init()) all_passed = .false.
    if (.not. test_assignment_overrides_data()) all_passed = .false.
    if (.not. test_real_array_implied_do()) all_passed = .false.
    if (.not. test_boz_constant()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: DATA statements lower through direct LIRIC session'

contains

    logical function test_scalar_init()
        ! Multiple scalar objects take values in list order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a, b'//new_line('a')// &
            '  data a, b /10, 20/'//new_line('a')// &
            '  print *, a, b'//new_line('a')// &
            'end program main'

        test_scalar_init = expect_output( &
            source, '          10          20'//new_line('a'), &
            '/tmp/ffc_data_scalar')
    end function test_scalar_init

    logical function test_array_init()
        ! An array object consumes one value per element in storage order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: arr(3)'//new_line('a')// &
            '  data arr /1, 2, 3/'//new_line('a')// &
            '  print *, arr(1), arr(2), arr(3)'//new_line('a')// &
            'end program main'

        test_array_init = expect_output( &
            source, '           1           2           3'//new_line('a'), &
            '/tmp/ffc_data_array')
    end function test_array_init

    logical function test_assignment_overrides_data()
        ! DATA initialises before execution; an executable assignment that
        ! precedes the DATA statement textually still wins at run time.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x(2)'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  data x /4, 5/'//new_line('a')// &
            '  print *, x(1), x(2)'//new_line('a')// &
            'end program main'

        test_assignment_overrides_data = expect_output( &
            source, '           7           7'//new_line('a'), &
            '/tmp/ffc_data_override')
    end function test_assignment_overrides_data

    logical function test_real_array_implied_do()
        ! Implied-do object: coeff(i)/coeff(i+2) resolve against the unrolled
        ! control value, matching gfortran storage order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: coeff(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  data (coeff(i), coeff(i+2), i=1,2) /1.0, 2.0, 3.0, 4.0/'// &
            new_line('a')// &
            '  print *, coeff(1), coeff(2), coeff(3), coeff(4)'//new_line('a')// &
            'end program main'

        test_real_array_implied_do = expect_output( &
            source, &
            '   1.00000000       3.00000000       2.00000000       4.00000000    '// &
            new_line('a'), &
            '/tmp/ffc_data_implied_do')
    end function test_real_array_implied_do

    logical function test_boz_constant()
        ! A hexadecimal BOZ initialiser decodes by radix (z'10' = 16).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: k'//new_line('a')// &
            "  data k /z'10'/"//new_line('a')// &
            '  print *, k'//new_line('a')// &
            'end program main'

        test_boz_constant = expect_output( &
            source, '          16'//new_line('a'), &
            '/tmp/ffc_data_boz')
    end function test_boz_constant

end program test_session_data_statement
