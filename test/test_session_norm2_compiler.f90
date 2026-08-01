program test_session_norm2_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session norm2 compiler test ==='

    all_passed = .true.
    if (.not. test_norm2_real_345()) all_passed = .false.
    if (.not. test_norm2_double_precision()) all_passed = .false.
    if (.not. test_norm2_zeros_and_section()) all_passed = .false.
    if (.not. test_norm2_rank2_whole()) all_passed = .false.
    if (.not. test_norm2_dim()) all_passed = .false.
    if (.not. test_norm2_integer_rejected()) all_passed = .false.

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

    logical function test_norm2_zeros_and_section()
        ! Zeros stay exactly zero (no 0/0 from the scaling step), a section
        ! reduces over the section extent only, and mixed large magnitudes
        ! must not overflow the accumulator: a naive sum of squares of
        ! 3.0e30/4.0e30 in real(4) saturates to Infinity.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: z(3) = [0.0, 0.0, 0.0]'//new_line('a')// &
            '  real :: a(5) = [0.0, 3.0, 4.0, 0.0, 12.0]'//new_line('a')// &
            '  real :: h(2) = [3.0e30, 4.0e30]'//new_line('a')// &
            '  print *, norm2(z)'//new_line('a')// &
            '  print *, norm2(a(2:4))'//new_line('a')// &
            '  print *, norm2(h)'//new_line('a')// &
            'end program main'

        test_norm2_zeros_and_section = expect_output( &
            source, &
            '   0.00000000    '//new_line('a')// &
            '   5.00000000    '//new_line('a')// &
            '   4.99999992E+30'//new_line('a'), &
            '/tmp/ffc_norm2_scaled')
    end function test_norm2_zeros_and_section

    logical function test_norm2_rank2_whole()
        ! norm2 without dim reduces every element of a rank-2 array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: b(2, 2)'//new_line('a')// &
            '  b = reshape([3.0, 4.0, 0.0, 0.0], [2, 2])'//new_line('a')// &
            '  print *, norm2(b)'//new_line('a')// &
            'end program main'

        test_norm2_rank2_whole = expect_output( &
            source, '   5.00000000    '//new_line('a'), '/tmp/ffc_norm2_rank2')
    end function test_norm2_rank2_whole

    logical function test_norm2_dim()
        ! norm2(c, dim) removes one dimension of a rank-2 real array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: c(2, 3), r1(3), r2(2)'//new_line('a')// &
            '  c = reshape([3.0, 4.0, 0.0, 3.0, 4.0, 0.0], [2, 3])'// &
            new_line('a')// &
            '  r1 = norm2(c, dim=1)'//new_line('a')// &
            '  r2 = norm2(c, 2)'//new_line('a')// &
            '  print *, r1'//new_line('a')// &
            '  print *, r2'//new_line('a')// &
            'end program main'

        test_norm2_dim = expect_output( &
            source, &
            '   5.00000000       3.00000000       4.00000000    '// &
            new_line('a')// &
            '   5.00000000       5.00000000    '//new_line('a'), &
            '/tmp/ffc_norm2_dim')
    end function test_norm2_dim

    logical function test_norm2_integer_rejected()
        ! norm2 requires a real argument.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3) = [3, 4, 0]'//new_line('a')// &
            '  print *, norm2(a)'//new_line('a')// &
            'end program main'

        test_norm2_integer_rejected = expect_error_contains( &
            source, 'norm2', '/tmp/ffc_norm2_int')
    end function test_norm2_integer_rejected

end program test_session_norm2_compiler
