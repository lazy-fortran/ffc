program test_session_real_pow_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session real power compiler test ==='

    all_passed = .true.
    if (.not. test_real_pow_two_three()) all_passed = .false.
    if (.not. test_real_pow_half()) all_passed = .false.
    if (.not. test_real_pow_variable()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real ** real lowers through direct LIRIC via libm pow'

contains

    logical function test_real_pow_two_three()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'real :: x'//new_line('a')// &
                                       'x = 2.0 ** 3.0'//new_line('a')// &
                                       'print *, x'//new_line('a')// &
                                       'end program main'

        test_real_pow_two_three = expect_output( &
                                  source, '   8.00000000    '//new_line('a'), &
                                  '/tmp/ffc_session_real_pow_two_three_test')
    end function test_real_pow_two_three

    logical function test_real_pow_half()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'real :: x'//new_line('a')// &
                                       'x = 4.0 ** 0.5'//new_line('a')// &
                                       'print *, x'//new_line('a')// &
                                       'end program main'

        test_real_pow_half = expect_output( &
                             source, '   2.00000000    '//new_line('a'), &
                             '/tmp/ffc_session_real_pow_half_test')
    end function test_real_pow_half

    logical function test_real_pow_variable()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'real :: b'//new_line('a')// &
                                       'real :: e'//new_line('a')// &
                                       'real :: x'//new_line('a')// &
                                       'b = 3.0'//new_line('a')// &
                                       'e = 2.0'//new_line('a')// &
                                       'x = b ** e'//new_line('a')// &
                                       'print *, x'//new_line('a')// &
                                       'end program main'

        test_real_pow_variable = expect_output( &
                                 source, '   9.00000000    '//new_line('a'), &
                                 '/tmp/ffc_session_real_pow_variable_test')
    end function test_real_pow_variable

end program test_session_real_pow_compiler
