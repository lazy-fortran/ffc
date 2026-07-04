program test_session_array_intrinsics_cluster_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session array-intrinsics cluster test ==='

    all_passed = .true.
    if (.not. test_reshape_int_source_real_target()) all_passed = .false.
    if (.not. test_reshape_param_shape_and_source()) all_passed = .false.
    if (.not. test_mask_reductions_and_product()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array intrinsics cluster lowers through direct LIRIC session'

contains

    ! reshape in a declaration initializer converts an integer source literal to
    ! the real target kind and fills column-major.
    logical function test_reshape_int_source_real_target()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: b(3, 2) = reshape([-1, -3, 6, 7, 8, 9], [3, 2])'// &
            new_line('a')// &
            '  print *, b(1, 1)'//new_line('a')// &
            '  print *, b(3, 1)'//new_line('a')// &
            '  print *, b(3, 2)'//new_line('a')// &
            'end program main'
        test_reshape_int_source_real_target = expect_output( &
            source, '  -1.00000000    '//new_line('a')// &
            '   6.00000000    '//new_line('a')// &
            '   9.00000000    '//new_line('a'), &
            '/tmp/ffc_session_reshape_int_real_test')
    end function test_reshape_int_source_real_target

    ! reshape in a declaration initializer resolves a parameter shape and an
    ! integer parameter source array, converting to the real target kind.
    logical function test_reshape_param_shape_and_source()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: n = 2'//new_line('a')// &
            '  integer, parameter :: arr(4) = [1, 2, 3, 4]'//new_line('a')// &
            '  real :: b(2, 2) = reshape(arr, [n, n])'//new_line('a')// &
            '  print *, b(1, 1)'//new_line('a')// &
            '  print *, b(2, 1)'//new_line('a')// &
            '  print *, b(1, 2)'//new_line('a')// &
            '  print *, b(2, 2)'//new_line('a')// &
            'end program main'
        test_reshape_param_shape_and_source = expect_output( &
            source, '   1.00000000    '//new_line('a')// &
            '   2.00000000    '//new_line('a')// &
            '   3.00000000    '//new_line('a')// &
            '   4.00000000    '//new_line('a'), &
            '/tmp/ffc_session_reshape_param_test')
    end function test_reshape_param_shape_and_source

    ! count/any/all over a logical mask and product over an integer array all
    ! reduce to scalars.
    logical function test_mask_reductions_and_product()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  logical :: m(5)'//new_line('a')// &
            '  m = a > 2'//new_line('a')// &
            '  print *, count(m)'//new_line('a')// &
            '  print *, any(m)'//new_line('a')// &
            '  print *, all(m)'//new_line('a')// &
            '  print *, product(a)'//new_line('a')// &
            'end program main'
        test_mask_reductions_and_product = expect_output( &
            source, '           3'//new_line('a')// &
            ' T'//new_line('a')// &
            ' F'//new_line('a')// &
            '         120'//new_line('a'), &
            '/tmp/ffc_session_mask_reductions_test')
    end function test_mask_reductions_and_product

end program test_session_array_intrinsics_cluster_compiler
