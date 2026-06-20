program test_session_norm2_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session norm2 compiler test ==='

    all_passed = .true.
    if (.not. test_norm2_real_345()) all_passed = .false.
    if (.not. test_norm2_double_precision()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: norm2 lowers through direct LIRIC session'

contains

    logical function test_norm2_real_345()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3) = [3.0, 4.0, 0.0]'//new_line('a')// &
            '  print *, norm2(a)'//new_line('a')// &
            'end program main'

        test_norm2_real_345 = expect_output( &
            source, '   5.00000000    '//new_line('a'), '/tmp/ffc_norm2_f32')
    end function test_norm2_real_345

    logical function test_norm2_double_precision()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  double precision :: a(4) = [1.0d0, 2.0d0, 2.0d0, 4.0d0]'// &
            new_line('a')// &
            '  print *, norm2(a)'//new_line('a')// &
            'end program main'

        test_norm2_double_precision = expect_output( &
            source, '   5.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_norm2_f64')
    end function test_norm2_double_precision

end program test_session_norm2_compiler
