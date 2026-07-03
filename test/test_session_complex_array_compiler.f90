program test_session_complex_array_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session complex array compiler test ==='

    all_passed = .true.
    if (.not. test_c4_array_element_add()) all_passed = .false.
    if (.not. test_c8_array_rank2()) all_passed = .false.
    if (.not. test_c4_array_element_mul()) all_passed = .false.
    if (.not. test_c4_array_element_print()) all_passed = .false.
    if (.not. test_c4_array_element_div()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fixed-size complex(4)/complex(8) arrays lower through '// &
        'direct LIRIC session'

contains

    logical function test_c4_array_element_add()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: a(3)'//new_line('a')// &
            '  a(1) = (1.0, 2.0)'//new_line('a')// &
            '  a(2) = (3.0, -1.0)'//new_line('a')// &
            '  a(3) = a(1) + a(2)'//new_line('a')// &
            '  print *, a(3)'//new_line('a')// &
            'end program main'

        test_c4_array_element_add = expect_output( &
            source, '             (4.00000000,1.00000000)'//new_line('a'), &
            '/tmp/ffc_session_c4_array_add_test')
    end function test_c4_array_element_add

    logical function test_c8_array_rank2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: m(2, 2)'//new_line('a')// &
            '  m(1, 1) = (1.0d0, 0.0d0)'//new_line('a')// &
            '  m(1, 2) = (2.0d0, 0.0d0)'//new_line('a')// &
            '  m(2, 1) = (0.0d0, 3.0d0)'//new_line('a')// &
            '  m(2, 2) = (0.0d0, 4.0d0)'//new_line('a')// &
            '  print *, m(2, 2)'//new_line('a')// &
            'end program main'

        test_c8_array_rank2 = expect_output( &
            source, &
            '               (0.0000000000000000,4.0000000000000000)'// &
            new_line('a'), '/tmp/ffc_session_c8_array_rank2_test')
    end function test_c8_array_rank2

    logical function test_c4_array_element_mul()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: a(2), b'//new_line('a')// &
            '  a(1) = (2.0, 3.0)'//new_line('a')// &
            '  a(2) = (1.0, 1.0)'//new_line('a')// &
            '  b = a(1) * a(2)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_c4_array_element_mul = expect_output( &
            source, '            (-1.00000000,5.00000000)'//new_line('a'), &
            '/tmp/ffc_session_c4_array_mul_test')
    end function test_c4_array_element_mul

    logical function test_c4_array_element_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: a(2)'//new_line('a')// &
            '  a(1) = (5.0, -2.0)'//new_line('a')// &
            '  a(2) = (1.0, 1.0)'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'

        test_c4_array_element_print = expect_output( &
            source, '            (5.00000000,-2.00000000)'//new_line('a'), &
            '/tmp/ffc_session_c4_array_print_test')
    end function test_c4_array_element_print

    logical function test_c4_array_element_div()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: a(2), b'//new_line('a')// &
            '  a(1) = (4.0, 2.0)'//new_line('a')// &
            '  a(2) = (2.0, 0.0)'//new_line('a')// &
            '  b = a(1) / a(2)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_c4_array_element_div = expect_output( &
            source, '             (2.00000000,1.00000000)'//new_line('a'), &
            '/tmp/ffc_session_c4_array_div_test')
    end function test_c4_array_element_div

end program test_session_complex_array_compiler
