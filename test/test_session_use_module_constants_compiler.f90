program test_session_use_module_constants_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session use-module-constants compiler test ==='

    all_passed = .true.
    if (.not. test_module_parameter_used_in_stop()) all_passed = .false.
    if (.not. test_module_parameter_in_arithmetic()) all_passed = .false.
    if (.not. test_module_parameter_real_still_diagnosed()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer module parameter constants import through USE'

contains

    logical function test_module_parameter_used_in_stop()
        character(len=*), parameter :: source = &
            'module consts'//new_line('a')// &
            '  integer, parameter :: WIDTH = 80'//new_line('a')// &
            'end module consts'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use consts'//new_line('a')// &
            '  stop WIDTH'//new_line('a')// &
            'end program main'

        test_module_parameter_used_in_stop = expect_exit_status( &
            source, 80, '/tmp/ffc_session_module_const_stop_test')
    end function test_module_parameter_used_in_stop

    logical function test_module_parameter_in_arithmetic()
        character(len=*), parameter :: source = &
            'module consts'//new_line('a')// &
            '  integer, parameter :: A = 5'//new_line('a')// &
            'end module consts'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use consts'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = A * 2'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_module_parameter_in_arithmetic = expect_exit_status( &
            source, 10, '/tmp/ffc_session_module_const_arith_test')
    end function test_module_parameter_in_arithmetic

    logical function test_module_parameter_real_still_diagnosed()
        character(len=*), parameter :: source = &
            'module consts'//new_line('a')// &
            '  real, parameter :: PI = 3.14'//new_line('a')// &
            'end module consts'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use consts'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_module_parameter_real_still_diagnosed = expect_error_contains( &
            source, 'only supports integer scalar parameters', &
            '/tmp/ffc_session_module_const_real_test')
    end function test_module_parameter_real_still_diagnosed

end program test_session_use_module_constants_compiler
