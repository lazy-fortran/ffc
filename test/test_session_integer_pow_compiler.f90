program test_session_integer_pow_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session integer power compiler test ==='

    all_passed = .true.
    if (.not. test_integer_pow_zero_is_one()) all_passed = .false.
    if (.not. test_integer_pow_one_is_identity()) all_passed = .false.
    if (.not. test_integer_pow_small()) all_passed = .false.
    if (.not. test_integer_pow_variable_base()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer ** literal exponent lowers through direct LIRIC'

contains

    logical function test_integer_pow_zero_is_one()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: x'//new_line('a')// &
                                       'x = 5 ** 0'//new_line('a')// &
                                       'stop x'//new_line('a')// &
                                       'end program main'

        test_integer_pow_zero_is_one = expect_exit_status( &
                                       source, 1, &
                                       '/tmp/ffc_session_pow_zero_test')
    end function test_integer_pow_zero_is_one

    logical function test_integer_pow_one_is_identity()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: x'//new_line('a')// &
                                       'x = 7 ** 1'//new_line('a')// &
                                       'stop x'//new_line('a')// &
                                       'end program main'

        test_integer_pow_one_is_identity = expect_exit_status( &
                                       source, 7, &
                                       '/tmp/ffc_session_pow_one_test')
    end function test_integer_pow_one_is_identity

    logical function test_integer_pow_small()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: x'//new_line('a')// &
                                       'x = 2 ** 5'//new_line('a')// &
                                       'stop x'//new_line('a')// &
                                       'end program main'

        test_integer_pow_small = expect_exit_status( &
                                 source, 32, &
                                 '/tmp/ffc_session_pow_small_test')
    end function test_integer_pow_small

    logical function test_integer_pow_variable_base()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       'integer :: b'//new_line('a')// &
                                       'integer :: x'//new_line('a')// &
                                       'b = 3'//new_line('a')// &
                                       'x = b ** 3'//new_line('a')// &
                                       'stop x'//new_line('a')// &
                                       'end program main'

        test_integer_pow_variable_base = expect_exit_status( &
                                         source, 27, &
                                         '/tmp/ffc_session_pow_varbase_test')
    end function test_integer_pow_variable_base

end program test_session_integer_pow_compiler
