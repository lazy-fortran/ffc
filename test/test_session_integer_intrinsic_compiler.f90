program test_session_integer_intrinsic_compiler
    use ffc_test_support, only: expect_exit_status, expect_output, &
                                expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session scalar intrinsic compiler test ==='

    all_passed = .true.
    if (.not. test_integer_intrinsic_values()) all_passed = .false.
    if (.not. test_integer_mod_intrinsic()) all_passed = .false.
    if (.not. test_integer_iand_intrinsic()) all_passed = .false.
    if (.not. test_integer_ior_intrinsic()) all_passed = .false.
    if (.not. test_integer_ieor_intrinsic()) all_passed = .false.
    if (.not. test_integer_not_intrinsic()) all_passed = .false.
    if (.not. test_real_intrinsic_values()) all_passed = .false.
    if (.not. test_real_conversion_intrinsic()) all_passed = .false.
    if (.not. test_unsupported_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_unsupported_real_intrinsic_diagnostic()) all_passed = .false.
    if (.not. test_user_function_shadowing()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar intrinsics lower through direct LIRIC session'

contains

    logical function test_integer_intrinsic_values()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  integer :: y'//new_line('a')// &
                                       '  x = abs(0 - 7)'//new_line('a')// &
                                  '  y = min(x, 9, 4) + max(1, 2, 3)'//new_line('a')// &
                                       '  stop y'//new_line('a')// &
                                       'end program main'

        test_integer_intrinsic_values = expect_exit_status( &
                                   source, 7, &
                                   '/tmp/ffc_session_integer_intrinsic_test')
    end function test_integer_intrinsic_values

    logical function test_integer_mod_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a'//new_line('a')// &
                                       '  integer :: b'//new_line('a')// &
                                       '  a = 17'//new_line('a')// &
                                       '  b = 5'//new_line('a')// &
                                       '  stop mod(a, b) + mod(-7, 3)'// &
                                       new_line('a')// &
                                       'end program main'

        test_integer_mod_intrinsic = expect_exit_status( &
                                     source, 1, &
                                     '/tmp/ffc_session_integer_mod_test')
    end function test_integer_mod_intrinsic

    logical function test_integer_iand_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = iand(12, 10)'//new_line('a')// &
                                       '  stop x'//new_line('a')// &
                                       'end program main'

        test_integer_iand_intrinsic = expect_exit_status( &
                                      source, 8, &
                                      '/tmp/ffc_session_integer_iand_test')
    end function test_integer_iand_intrinsic

    logical function test_integer_ior_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = ior(12, 10)'//new_line('a')// &
                                       '  stop x'//new_line('a')// &
                                       'end program main'

        test_integer_ior_intrinsic = expect_exit_status( &
                                     source, 14, &
                                     '/tmp/ffc_session_integer_ior_test')
    end function test_integer_ior_intrinsic

    logical function test_integer_ieor_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = ieor(12, 10)'//new_line('a')// &
                                       '  stop x'//new_line('a')// &
                                       'end program main'

        test_integer_ieor_intrinsic = expect_exit_status( &
                                      source, 6, &
                                      '/tmp/ffc_session_integer_ieor_test')
    end function test_integer_ieor_intrinsic

    logical function test_integer_not_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = abs(not(0))'//new_line('a')// &
                                       '  stop x'//new_line('a')// &
                                       'end program main'

        ! not(0) == -1, abs(-1) == 1
        test_integer_not_intrinsic = expect_exit_status( &
                                     source, 1, &
                                     '/tmp/ffc_session_integer_not_test')
    end function test_integer_not_intrinsic

    logical function test_real_intrinsic_values()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = abs(0.0 - 2.5) '// &
                                       '+ min(3.5, 1.25, 2.0) '// &
                                       '+ max(0.5, 4.25)'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'end program main'

        test_real_intrinsic_values = expect_output( &
                                     source, '8.000000'//new_line('a'), &
                                     '/tmp/ffc_session_real_intrinsic_test')
    end function test_real_intrinsic_values

    logical function test_real_conversion_intrinsic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: i'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  i = 4'//new_line('a')// &
                                       '  x = real(i) + 1.5'//new_line('a')// &
                                       '  print *, x'//new_line('a')// &
                                       'end program main'

        test_real_conversion_intrinsic = expect_output( &
                                        source, '5.500000'//new_line('a'), &
                                        '/tmp/ffc_session_real_conversion_test')
    end function test_real_conversion_intrinsic

    logical function test_unsupported_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  stop int(5.5)'//new_line('a')// &
                                       'end program main'

        test_unsupported_intrinsic_diagnostic = expect_error_contains( &
            source, 'unsupported scalar intrinsic: int', &
            '/tmp/ffc_session_unsupported_intrinsic_test')
    end function test_unsupported_intrinsic_diagnostic

    logical function test_unsupported_real_intrinsic_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  real :: x'//new_line('a')// &
                                       '  x = sqrt(4.0)'//new_line('a')// &
                                       'end program main'

        test_unsupported_real_intrinsic_diagnostic = expect_error_contains( &
            source, 'unsupported scalar intrinsic: sqrt', &
            '/tmp/ffc_session_unsupported_real_intrinsic_test')
    end function test_unsupported_real_intrinsic_diagnostic

    logical function test_user_function_shadowing()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: x'//new_line('a')// &
                                       '  x = min(2, 3)'//new_line('a')// &
                                       '  stop x'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  integer function min(a, b)'//new_line('a')// &
                                       '    min = a + b'//new_line('a')// &
                                       '  end function min'//new_line('a')// &
                                       'end program main'

        test_user_function_shadowing = expect_exit_status( &
                                      source, 5, &
                                      '/tmp/ffc_session_intrinsic_shadow_test')
    end function test_user_function_shadowing

end program test_session_integer_intrinsic_compiler
