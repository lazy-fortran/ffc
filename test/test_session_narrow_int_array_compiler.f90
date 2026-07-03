program test_session_narrow_int_array_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session narrow-integer array compiler test ==='

    all_passed = .true.
    if (.not. test_i64_array_element_rw()) all_passed = .false.
    if (.not. test_i64_array_rank2()) all_passed = .false.
    if (.not. test_i64_array_whole_print()) all_passed = .false.
    if (.not. test_i64_array_constructor()) all_passed = .false.
    if (.not. test_i64_array_scalar_broadcast()) all_passed = .false.
    if (.not. test_i8_array_element_rw()) all_passed = .false.
    if (.not. test_i16_array_element_rw()) all_passed = .false.
    if (.not. test_i64_array_comparison_stop_code()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer(1)/(2)/(8) fixed-size arrays lower through '// &
        'direct LIRIC session'

contains

    logical function test_i64_array_element_rw()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: a(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    a(i) = i * 100_8'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, a(1) + a(4)'//new_line('a')// &
            'end program main'

        test_i64_array_element_rw = expect_output( &
            source, '                  500'//new_line('a'), &
            '/tmp/ffc_session_i64_array_rw_test')
    end function test_i64_array_element_rw

    logical function test_i64_array_rank2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: ilist(1, 1)'//new_line('a')// &
            '  ilist(1, 1) = 1'//new_line('a')// &
            '  if (ilist(1, 1) /= 1) error stop'//new_line('a')// &
            '  print *, ilist(1, 1)'//new_line('a')// &
            'end program main'

        test_i64_array_rank2 = expect_output( &
            source, '                    1'//new_line('a'), &
            '/tmp/ffc_session_i64_array_rank2_test')
    end function test_i64_array_rank2

    logical function test_i64_array_whole_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: a(3)'//new_line('a')// &
            '  a(1) = 10'//new_line('a')// &
            '  a(2) = 20'//new_line('a')// &
            '  a(3) = 30'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_i64_array_whole_print = expect_output( &
            source, '                   10                   20'// &
            '                   30'//new_line('a'), &
            '/tmp/ffc_session_i64_array_whole_print_test')
    end function test_i64_array_whole_print

    logical function test_i64_array_constructor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: a(3) = [-14_8, 3_8, 20_8]'//new_line('a')// &
            '  print *, a(1) + a(2) + a(3)'//new_line('a')// &
            'end program main'

        test_i64_array_constructor = expect_output( &
            source, '                    9'//new_line('a'), &
            '/tmp/ffc_session_i64_array_ctor_test')
    end function test_i64_array_constructor

    logical function test_i64_array_scalar_broadcast()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: a(3)'//new_line('a')// &
            '  a = 7_8'//new_line('a')// &
            '  print *, a(1) + a(3)'//new_line('a')// &
            'end program main'

        test_i64_array_scalar_broadcast = expect_output( &
            source, '                   14'//new_line('a'), &
            '/tmp/ffc_session_i64_array_broadcast_test')
    end function test_i64_array_scalar_broadcast

    logical function test_i8_array_element_rw()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_fortran_env, only: int8'//new_line('a')// &
            '  integer(int8) :: a(3)'//new_line('a')// &
            '  a(1) = 1_int8'//new_line('a')// &
            '  a(2) = 2_int8'//new_line('a')// &
            '  a(3) = a(1) + a(2)'//new_line('a')// &
            '  print *, a(3)'//new_line('a')// &
            'end program main'

        test_i8_array_element_rw = expect_output( &
            source, '    3'//new_line('a'), &
            '/tmp/ffc_session_i8_array_rw_test')
    end function test_i8_array_element_rw

    logical function test_i16_array_element_rw()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_fortran_env, only: int16'//new_line('a')// &
            '  integer(int16) :: a(3)'//new_line('a')// &
            '  a(1) = 1000_int16'//new_line('a')// &
            '  a(2) = 2000_int16'//new_line('a')// &
            '  a(3) = a(1) + a(2)'//new_line('a')// &
            '  print *, a(3)'//new_line('a')// &
            'end program main'

        test_i16_array_element_rw = expect_output( &
            source, '   3000'//new_line('a'), &
            '/tmp/ffc_session_i16_array_rw_test')
    end function test_i16_array_element_rw

    logical function test_i64_array_comparison_stop_code()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8) :: a(2)'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  a(1) = 7_8'//new_line('a')// &
            '  a(2) = 9_8'//new_line('a')// &
            '  if (a(1) /= 7_8) error stop'//new_line('a')// &
            '  if (a(2) /= 9_8) error stop'//new_line('a')// &
            '  n = 9'//new_line('a')// &
            '  stop n'//new_line('a')// &
            'end program main'

        test_i64_array_comparison_stop_code = expect_exit_status( &
            source, 9, '/tmp/ffc_session_i64_array_stop_test')
    end function test_i64_array_comparison_stop_code

end program test_session_narrow_int_array_compiler
