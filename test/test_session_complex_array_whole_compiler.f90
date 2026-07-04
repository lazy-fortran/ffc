program test_session_complex_array_whole_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session complex whole-array compiler test ==='

    all_passed = .true.
    if (.not. test_c8_whole_add()) all_passed = .false.
    if (.not. test_c8_whole_sub()) all_passed = .false.
    if (.not. test_c8_whole_mul()) all_passed = .false.
    if (.not. test_c8_whole_copy()) all_passed = .false.
    if (.not. test_c8_whole_broadcast()) all_passed = .false.
    if (.not. test_c4_whole_add()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: elementwise whole-array complex(4)/complex(8) assignment '// &
        'lowers through direct LIRIC session'

contains

    logical function test_c8_whole_add()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: a(3), b(3), c(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    a(i) = (1.0d0, 2.0d0)'//new_line('a')// &
            '    b(i) = (3.0d0, 4.0d0)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  c = a + b'//new_line('a')// &
            '  print *, c(2)'//new_line('a')// &
            'end program main'

        test_c8_whole_add = expect_output( &
            source, &
            '               (4.0000000000000000,6.0000000000000000)'// &
            new_line('a'), '/tmp/ffc_session_c8_whole_add_test')
    end function test_c8_whole_add

    logical function test_c8_whole_sub()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: a(2), b(2), d(2)'//new_line('a')// &
            '  a(1) = (5.0d0, 1.0d0)'//new_line('a')// &
            '  a(2) = (2.0d0, 2.0d0)'//new_line('a')// &
            '  b(1) = (1.0d0, 1.0d0)'//new_line('a')// &
            '  b(2) = (1.0d0, 0.0d0)'//new_line('a')// &
            '  d = a - b'//new_line('a')// &
            '  print *, d(1)'//new_line('a')// &
            'end program main'

        test_c8_whole_sub = expect_output( &
            source, &
            '               (4.0000000000000000,0.0000000000000000)'// &
            new_line('a'), '/tmp/ffc_session_c8_whole_sub_test')
    end function test_c8_whole_sub

    logical function test_c8_whole_mul()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: a(2), b(2), e(2)'//new_line('a')// &
            '  a(1) = (5.0d0, 1.0d0)'//new_line('a')// &
            '  a(2) = (2.0d0, 2.0d0)'//new_line('a')// &
            '  b(1) = (1.0d0, 1.0d0)'//new_line('a')// &
            '  b(2) = (1.0d0, 0.0d0)'//new_line('a')// &
            '  e = a * b'//new_line('a')// &
            '  print *, e(2)'//new_line('a')// &
            'end program main'

        test_c8_whole_mul = expect_output( &
            source, &
            '               (2.0000000000000000,2.0000000000000000)'// &
            new_line('a'), '/tmp/ffc_session_c8_whole_mul_test')
    end function test_c8_whole_mul

    logical function test_c8_whole_copy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: a(3), c(3)'//new_line('a')// &
            '  a(1) = (7.0d0, -1.0d0)'//new_line('a')// &
            '  a(2) = (0.0d0, 0.0d0)'//new_line('a')// &
            '  a(3) = (0.0d0, 0.0d0)'//new_line('a')// &
            '  c = a'//new_line('a')// &
            '  print *, c(1)'//new_line('a')// &
            'end program main'

        test_c8_whole_copy = expect_output( &
            source, &
            '              (7.0000000000000000,-1.0000000000000000)'// &
            new_line('a'), '/tmp/ffc_session_c8_whole_copy_test')
    end function test_c8_whole_copy

    logical function test_c8_whole_broadcast()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(8) :: c(3)'//new_line('a')// &
            '  c = (2.0d0, 3.0d0)'//new_line('a')// &
            '  print *, c(3)'//new_line('a')// &
            'end program main'

        test_c8_whole_broadcast = expect_output( &
            source, &
            '               (2.0000000000000000,3.0000000000000000)'// &
            new_line('a'), '/tmp/ffc_session_c8_whole_broadcast_test')
    end function test_c8_whole_broadcast

    logical function test_c4_whole_add()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex(4) :: a(2), b(2), c(2)'//new_line('a')// &
            '  a(1) = (2.0, 3.0)'//new_line('a')// &
            '  a(2) = (1.0, 1.0)'//new_line('a')// &
            '  b(1) = (1.0, 1.0)'//new_line('a')// &
            '  b(2) = (4.0, 4.0)'//new_line('a')// &
            '  c = a + b'//new_line('a')// &
            '  print *, c(1)'//new_line('a')// &
            'end program main'

        test_c4_whole_add = expect_output( &
            source, '             (3.00000000,4.00000000)'//new_line('a'), &
            '/tmp/ffc_session_c4_whole_add_test')
    end function test_c4_whole_add

end program test_session_complex_array_whole_compiler
