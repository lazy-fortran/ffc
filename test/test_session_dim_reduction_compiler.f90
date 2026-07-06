program test_session_dim_reduction_compiler
    ! Whole-array reductions along one dimension of a rank-2 source into a
    ! rank-1 target: sum/product over a stored numeric array, and count/any/all
    ! over a stored logical mask or an elementwise relational comparison. Every
    ! extent is known at compile time, so each result element unrolls to a fixed
    ! accumulation. Also covers reshape with a shape(X) shape argument. Outputs
    ! match gfortran list-directed formatting.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session dim reduction compiler test ==='

    all_passed = .true.
    if (.not. test_sum_product_dim()) all_passed = .false.
    if (.not. test_mask_dim()) all_passed = .false.
    if (.not. test_reshape_shape_of()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: dim-wise reductions and shape(X) reshape lower correctly'

contains

    logical function test_sum_product_dim()
        ! sum along dim=1 (per column) and product along dim=2 (per row) of a
        ! reshape-initialized rank-2 integer array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 3)'//new_line('a')// &
            '  integer :: s1(3), p2(2)'//new_line('a')// &
            '  a = reshape([1, 2, 3, 4, 5, 6], [2, 3])'//new_line('a')// &
            '  s1 = sum(a, 1)'//new_line('a')// &
            '  p2 = product(a, dim=2)'//new_line('a')// &
            '  print *, s1'//new_line('a')// &
            '  print *, p2'//new_line('a')// &
            'end program main'

        test_sum_product_dim = expect_output( &
            source, &
            '           3           7          11'//new_line('a')// &
            '          15          48'//new_line('a'), &
            '/tmp/ffc_dim_reduce_sumprod')
    end function test_sum_product_dim

    logical function test_mask_dim()
        ! count/any/all along a dimension over both a relational comparison with
        ! a broadcast scalar and a comparison of two arrays.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 3), b(2, 3)'//new_line('a')// &
            '  integer :: c1(3)'//new_line('a')// &
            '  logical :: any1(3), all2(2)'//new_line('a')// &
            '  a = reshape([1, 2, 3, 4, 5, 6], [2, 3])'//new_line('a')// &
            '  b = reshape([1, 0, 3, 4, 0, 6], [2, 3])'//new_line('a')// &
            '  c1 = count(a > 3, 1)'//new_line('a')// &
            '  any1 = any(a == b, 1)'//new_line('a')// &
            '  all2 = all(a == b, 2)'//new_line('a')// &
            '  print *, c1'//new_line('a')// &
            '  print *, any1'//new_line('a')// &
            '  print *, all2'//new_line('a')// &
            'end program main'

        test_mask_dim = expect_output( &
            source, &
            '           0           1           2'//new_line('a')// &
            ' T T T'//new_line('a')// &
            ' F F'//new_line('a'), &
            '/tmp/ffc_dim_reduce_mask')
    end function test_mask_dim

    logical function test_reshape_shape_of()
        ! reshape whose shape argument is shape(X) folds to the target's static
        ! extents rather than requiring a literal shape.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: s(2, 3)'//new_line('a')// &
            '  s = reshape([1, 2, 3, 4, 5, 6], shape(s))'//new_line('a')// &
            '  print *, s(1, 1), s(2, 1), s(1, 3), s(2, 3)'//new_line('a')// &
            'end program main'

        test_reshape_shape_of = expect_output( &
            source, &
            '           1           2           5           6'//new_line('a'), &
            '/tmp/ffc_dim_reduce_reshape_shape')
    end function test_reshape_shape_of

end program test_session_dim_reduction_compiler
