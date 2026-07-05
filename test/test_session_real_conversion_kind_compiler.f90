program test_session_real_conversion_kind_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session real() conversion kind compiler test ==='

    all_passed = .true.
    if (.not. test_real_of_integer_is_single()) all_passed = .false.
    if (.not. test_real_of_double_is_single()) all_passed = .false.
    if (.not. test_real_with_kind_eight_is_double()) all_passed = .false.
    if (.not. test_real_of_complex_keeps_kind()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real() conversion result kind follows F2018 16.9.160'

contains

    logical function test_real_of_integer_is_single()
        !! real(i) with no KIND is default real (single); a prior lowering
        !! printed it as double.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i = 3'//new_line('a')// &
            '  print *, real(i)'//new_line('a')// &
            'end program main'

        test_real_of_integer_is_single = expect_output( &
            source, '   3.00000000    '//new_line('a'), &
            '/tmp/ffc_session_real_int_single_test')
    end function test_real_of_integer_is_single

    logical function test_real_of_double_is_single()
        !! real(d) on a real(8) with no KIND narrows to default real (single).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: d = 2.5d0'//new_line('a')// &
            '  print *, real(d)'//new_line('a')// &
            'end program main'

        test_real_of_double_is_single = expect_output( &
            source, '   2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_real_double_single_test')
    end function test_real_of_double_is_single

    logical function test_real_with_kind_eight_is_double()
        !! An explicit KIND=8 selector yields real(8).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i = 3'//new_line('a')// &
            '  print *, real(i, 8)'//new_line('a')// &
            'end program main'

        test_real_with_kind_eight_is_double = expect_output( &
            source, '   3.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_session_real_kind8_double_test')
    end function test_real_with_kind_eight_is_double

    logical function test_real_of_complex_keeps_kind()
        !! real(z) on a complex(8) with no KIND keeps the argument kind (real(8)).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: z = (1.0d0, 2.0d0)'//new_line('a')// &
            '  print *, real(z)'//new_line('a')// &
            'end program main'

        test_real_of_complex_keeps_kind = expect_output( &
            source, '   1.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_session_real_complex_double_test')
    end function test_real_of_complex_keeps_kind

end program test_session_real_conversion_kind_compiler
