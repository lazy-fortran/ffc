program test_session_allocatable_reduction_compiler
    ! Array reduction intrinsics (sum, product, maxval, minval) over a 1-D
    ! allocatable whose extent is known at compile time from the preceding
    ! allocate(a(N)). The reduction unrolls over that static extent, loading
    ! each element from the allocatable descriptor data pointer. Outputs match
    ! gfortran list-directed formatting.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable reduction compiler test ==='

    all_passed = .true.
    if (.not. test_integer_sum()) all_passed = .false.
    if (.not. test_real_sum_product()) all_passed = .false.
    if (.not. test_minval_maxval()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable reductions lower through direct LIRIC session'

contains

    logical function test_integer_sum()
        ! sum() over an integer allocatable broadcast-assigned to a constant.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(10))'//new_line('a')// &
            '  a = 42'//new_line('a')// &
            '  print *, sum(a)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_integer_sum = expect_output( &
            source, '         420'//new_line('a'), &
            '/tmp/ffc_alloc_reduce_isum')
    end function test_integer_sum

    logical function test_real_sum_product()
        ! sum and product over a real allocatable element-assigned by a loop.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  allocate(a(4))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    a(i) = real(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, sum(a)'//new_line('a')// &
            '  print *, product(a)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_real_sum_product = expect_output( &
            source, &
            '   10.0000000    '//new_line('a')// &
            '   24.0000000    '//new_line('a'), &
            '/tmp/ffc_alloc_reduce_rsp')
    end function test_real_sum_product

    logical function test_minval_maxval()
        ! minval/maxval over an integer allocatable constructor assignment.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [3, 1, 2]'//new_line('a')// &
            '  print *, minval(a), maxval(a)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_minval_maxval = expect_output( &
            source, '           1           3'//new_line('a'), &
            '/tmp/ffc_alloc_reduce_mm')
    end function test_minval_maxval

end program test_session_allocatable_reduction_compiler
