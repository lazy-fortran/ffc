program test_session_array_shift_intrinsics_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session cshift/eoshift dim/boundary compiler test ==='

    all_passed = .true.
    if (.not. test_cshift_oversized_shift()) all_passed = .false.
    if (.not. test_cshift_explicit_dim_one()) all_passed = .false.
    if (.not. test_cshift_rank2_dim_two()) all_passed = .false.
    if (.not. test_cshift_rank2_vector_shift()) all_passed = .false.
    if (.not. test_eoshift_scalar_boundary()) all_passed = .false.
    if (.not. test_eoshift_real_boundary()) all_passed = .false.
    if (.not. test_eoshift_rank2_dim_two_boundary()) all_passed = .false.
    if (.not. test_invalid_dim_rejected()) all_passed = .false.
    if (.not. test_nonconformable_shift_rejected()) all_passed = .false.
    if (.not. test_nonconformable_boundary_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: cshift/eoshift honour dim, boundary, and shift conformance'

contains

    ! An oversized shift normalizes modulo the extent: cshift(a, 6) on a size-4
    ! array behaves like cshift(a, 2).
    logical function test_cshift_oversized_shift()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  r = cshift(a, 6)'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_cshift_oversized_shift = expect_output( &
            source, '           3'//new_line('a')// &
            '           4'//new_line('a')// &
            '           1'//new_line('a')// &
            '           2'//new_line('a'), &
            '/tmp/ffc_session_shift_over_test')
    end function test_cshift_oversized_shift

    ! An explicit dim=1 on a rank-1 array is the intrinsic default.
    logical function test_cshift_explicit_dim_one()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: r(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [7, 8, 9]'//new_line('a')// &
            '  r = cshift(a, -1, 1)'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_cshift_explicit_dim_one = expect_output( &
            source, '           9'//new_line('a')// &
            '           7'//new_line('a')// &
            '           8'//new_line('a'), &
            '/tmp/ffc_session_shift_dim1_test')
    end function test_cshift_explicit_dim_one

    ! cshift along dim=2 rotates whole columns of a rank-2 array.
    logical function test_cshift_rank2_dim_two()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 3)'//new_line('a')// &
            '  integer :: r(2, 3)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        a(i, j) = 10 * i + j'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  r = cshift(a, 1, 2)'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_cshift_rank2_dim_two = expect_output( &
            source, '          12'//new_line('a')// &
            '          22'//new_line('a')// &
            '          13'//new_line('a')// &
            '          23'//new_line('a')// &
            '          11'//new_line('a')// &
            '          21'//new_line('a'), &
            '/tmp/ffc_session_shift_r2dim2_test')
    end function test_cshift_rank2_dim_two

    ! A rank-reduced (rank-1) shift gives each column its own rotation.
    logical function test_cshift_rank2_vector_shift()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3, 2)'//new_line('a')// &
            '  integer :: r(3, 2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 3'//new_line('a')// &
            '        a(i, j) = 10 * j + i'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  r = cshift(a, [1, -1], 1)'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 3'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_cshift_rank2_vector_shift = expect_output( &
            source, '          12'//new_line('a')// &
            '          13'//new_line('a')// &
            '          11'//new_line('a')// &
            '          23'//new_line('a')// &
            '          21'//new_line('a')// &
            '          22'//new_line('a'), &
            '/tmp/ffc_session_shift_vecshift_test')
    end function test_cshift_rank2_vector_shift

    ! eoshift fills the vacated positions from a scalar boundary.
    logical function test_eoshift_scalar_boundary()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  r = eoshift(a, 2, -9)'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_eoshift_scalar_boundary = expect_output( &
            source, '           3'//new_line('a')// &
            '           4'//new_line('a')// &
            '          -9'//new_line('a')// &
            '          -9'//new_line('a'), &
            '/tmp/ffc_session_shift_bnd_test')
    end function test_eoshift_scalar_boundary

    ! A real boundary fills a real array on a negative shift.
    logical function test_eoshift_real_boundary()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(3)'//new_line('a')// &
            '  real :: r(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
            '  r = eoshift(a, -1, 5.5)'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_eoshift_real_boundary = expect_output( &
            source, '   5.50000000    '//new_line('a')// &
            '   1.00000000    '//new_line('a')// &
            '   2.00000000    '//new_line('a'), &
            '/tmp/ffc_session_shift_realbnd_test')
    end function test_eoshift_real_boundary

    ! eoshift along dim=2 with a boundary fills whole vacated columns.
    logical function test_eoshift_rank2_dim_two_boundary()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2)'//new_line('a')// &
            '  integer :: r(2, 2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        a(i, j) = 10 * i + j'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  r = eoshift(a, 1, 7, 2)'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_eoshift_rank2_dim_two_boundary = expect_output( &
            source, '          12'//new_line('a')// &
            '          22'//new_line('a')// &
            '           7'//new_line('a')// &
            '           7'//new_line('a'), &
            '/tmp/ffc_session_shift_r2bnd_test')
    end function test_eoshift_rank2_dim_two_boundary

    ! dim must lie within 1..rank of the source array.
    logical function test_invalid_dim_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2)'//new_line('a')// &
            '  integer :: r(2, 2)'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  r = cshift(a, 1, 3)'//new_line('a')// &
            '  print *, r(1, 1)'//new_line('a')// &
            'end program main'
        test_invalid_dim_rejected = expect_error_contains( &
            source, 'dim', '/tmp/ffc_session_shift_baddim_test')
    end function test_invalid_dim_rejected

    ! A rank-reduced shift must conform to the remaining extent.
    logical function test_nonconformable_shift_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3, 2)'//new_line('a')// &
            '  integer :: r(3, 2)'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  r = cshift(a, [1, 2, 3], 1)'//new_line('a')// &
            '  print *, r(1, 1)'//new_line('a')// &
            'end program main'
        test_nonconformable_shift_rejected = expect_error_contains( &
            source, 'shift', '/tmp/ffc_session_shift_badshift_test')
    end function test_nonconformable_shift_rejected

    ! A rank-1 boundary is not conformable with a rank-1 eoshift source.
    logical function test_nonconformable_boundary_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: r(3)'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  r = eoshift(a, 1, [1, 2, 3])'//new_line('a')// &
            '  print *, r(1)'//new_line('a')// &
            'end program main'
        test_nonconformable_boundary_rejected = expect_error_contains( &
            source, 'boundary', '/tmp/ffc_session_shift_badbnd_test')
    end function test_nonconformable_boundary_rejected

end program test_session_array_shift_intrinsics_compiler
