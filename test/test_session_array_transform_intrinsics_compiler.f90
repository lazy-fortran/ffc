program test_session_array_transform_intrinsics_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session array transform intrinsics compiler test ==='

    all_passed = .true.
    if (.not. test_merge_scalar_mask_true()) all_passed = .false.
    if (.not. test_merge_scalar_mask_variable()) all_passed = .false.
    if (.not. test_merge_rank2_array_mask()) all_passed = .false.
    if (.not. test_merge_rank3_array_mask()) all_passed = .false.
    if (.not. test_pack_unpack_roundtrip()) all_passed = .false.
    if (.not. test_spread_real_dim1()) all_passed = .false.
    if (.not. test_spread_invalid_dim()) all_passed = .false.
    if (.not. test_spread_invalid_ncopies()) all_passed = .false.
    if (.not. test_merge_nonconformable()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array transform intrinsics lower through direct LIRIC session'

contains

    ! merge accepts a scalar mask: a .true. scalar selects tsource everywhere.
    logical function test_merge_scalar_mask_true()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: t(3)'//new_line('a')// &
            '  integer :: f(3)'//new_line('a')// &
            '  integer :: r(3)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  t = [1, 2, 3]'//new_line('a')// &
            '  f = [7, 8, 9]'//new_line('a')// &
            '  r = merge(t, f, .true.)'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_merge_scalar_mask_true = expect_output( &
            source, '           1'//new_line('a')// &
            '           2'//new_line('a')// &
            '           3'//new_line('a'), &
            '/tmp/ffc_session_transform_merge_scalar_true')
    end function test_merge_scalar_mask_true

    ! A scalar logical variable mask selects fsource when it is .false.
    logical function test_merge_scalar_mask_variable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: t(2)'//new_line('a')// &
            '  real :: f(2)'//new_line('a')// &
            '  real :: r(2)'//new_line('a')// &
            '  logical :: m'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  t = [1.5, 2.5]'//new_line('a')// &
            '  f = [3.5, 4.5]'//new_line('a')// &
            '  m = .false.'//new_line('a')// &
            '  r = merge(t, f, m)'//new_line('a')// &
            '  do i = 1, 2'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_merge_scalar_mask_variable = expect_output( &
            source, '   3.50000000    '//new_line('a')// &
            '   4.50000000    '//new_line('a'), &
            '/tmp/ffc_session_transform_merge_scalar_var')
    end function test_merge_scalar_mask_variable

    ! A rank-2 array mask selects each element in Fortran column-major order.
    logical function test_merge_rank2_array_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: t(2, 2)'//new_line('a')// &
            '  integer :: f(2, 2)'//new_line('a')// &
            '  integer :: r(2, 2)'//new_line('a')// &
            '  logical :: m(2, 2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  t(1, 1) = 1'//new_line('a')// &
            '  t(2, 1) = 2'//new_line('a')// &
            '  t(1, 2) = 3'//new_line('a')// &
            '  t(2, 2) = 4'//new_line('a')// &
            '  f(1, 1) = 10'//new_line('a')// &
            '  f(2, 1) = 20'//new_line('a')// &
            '  f(1, 2) = 30'//new_line('a')// &
            '  f(2, 2) = 40'//new_line('a')// &
            '  m(1, 1) = .true.'//new_line('a')// &
            '  m(2, 1) = .false.'//new_line('a')// &
            '  m(1, 2) = .false.'//new_line('a')// &
            '  m(2, 2) = .true.'//new_line('a')// &
            '  r = merge(t, f, m)'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_merge_rank2_array_mask = expect_output( &
            source, '           1'//new_line('a')// &
            '          20'//new_line('a')// &
            '          30'//new_line('a')// &
            '           4'//new_line('a'), &
            '/tmp/ffc_session_transform_merge_rank2')
    end function test_merge_rank2_array_mask

    ! A rank-3 array mask is traversed in Fortran column-major order.
    logical function test_merge_rank3_array_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: t(2, 2, 2)'//new_line('a')// &
            '  integer :: f(2, 2, 2)'//new_line('a')// &
            '  integer :: r(2, 2, 2)'//new_line('a')// &
            '  logical :: m(2, 2, 2)'//new_line('a')// &
            '  integer :: i, j, k, n'//new_line('a')// &
            '  n = 0'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '     do j = 1, 2'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '           n = n + 1'//new_line('a')// &
            '           t(i, j, k) = n'//new_line('a')// &
            '           f(i, j, k) = 100 + n'//new_line('a')// &
            '           m(i, j, k) = mod(n, 2) == 1'//new_line('a')// &
            '        end do'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  r = merge(t, f, m)'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '     do j = 1, 2'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '           print *, r(i, j, k)'//new_line('a')// &
            '        end do'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_merge_rank3_array_mask = expect_output( &
            source, '           1'//new_line('a')// &
            '         102'//new_line('a')// &
            '           3'//new_line('a')// &
            '         104'//new_line('a')// &
            '           5'//new_line('a')// &
            '         106'//new_line('a')// &
            '           7'//new_line('a')// &
            '         108'//new_line('a'), &
            '/tmp/ffc_session_transform_merge_rank3')
    end function test_merge_rank3_array_mask

    ! pack then unpack with the same mask restores the masked positions.
    logical function test_pack_unpack_roundtrip()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: p(2)'//new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  logical :: m(4)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  m = [.false., .true., .false., .true.]'//new_line('a')// &
            '  p = pack(a, m)'//new_line('a')// &
            '  r = unpack(p, m, 0)'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     print *, r(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_pack_unpack_roundtrip = expect_output( &
            source, '           0'//new_line('a')// &
            '           2'//new_line('a')// &
            '           0'//new_line('a')// &
            '           4'//new_line('a'), &
            '/tmp/ffc_session_transform_pack_unpack')
    end function test_pack_unpack_roundtrip

    ! spread(a, 1, n) on a real source replicates down the new leading dim.
    logical function test_spread_real_dim1()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            '  real :: r(2, 2)'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  a = [1.5, 2.5]'//new_line('a')// &
            '  r = spread(a, 1, 2)'//new_line('a')// &
            '  do j = 1, 2'//new_line('a')// &
            '     do i = 1, 2'//new_line('a')// &
            '        print *, r(i, j)'//new_line('a')// &
            '     end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'
        test_spread_real_dim1 = expect_output( &
            source, '   1.50000000    '//new_line('a')// &
            '   1.50000000    '//new_line('a')// &
            '   2.50000000    '//new_line('a')// &
            '   2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_transform_spread_real_dim1')
    end function test_spread_real_dim1

    ! A DIM outside 1..rank+1 is rejected with a diagnostic.
    logical function test_spread_invalid_dim()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: r(3, 2)'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  r = spread(a, 3, 2)'//new_line('a')// &
            '  print *, r(1, 1)'//new_line('a')// &
            'end program main'
        test_spread_invalid_dim = expect_error_contains( &
            source, 'spread dim', &
            '/tmp/ffc_session_transform_spread_bad_dim')
    end function test_spread_invalid_dim

    ! A negative NCOPIES is rejected with a diagnostic.
    logical function test_spread_invalid_ncopies()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  integer :: r(3, 2)'//new_line('a')// &
            '  a = [1, 2, 3]'//new_line('a')// &
            '  r = spread(a, 2, -1)'//new_line('a')// &
            '  print *, r(1, 1)'//new_line('a')// &
            'end program main'
        test_spread_invalid_ncopies = expect_error_contains( &
            source, 'spread ncopies', &
            '/tmp/ffc_session_transform_spread_bad_ncopies')
    end function test_spread_invalid_ncopies

    ! Nonconformable merge operands are rejected with a diagnostic.
    logical function test_merge_nonconformable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: t(3)'//new_line('a')// &
            '  integer :: f(2)'//new_line('a')// &
            '  logical :: m(3)'//new_line('a')// &
            '  integer :: r(3)'//new_line('a')// &
            '  t = [1, 2, 3]'//new_line('a')// &
            '  f = [7, 8]'//new_line('a')// &
            '  m = [.true., .false., .true.]'//new_line('a')// &
            '  r = merge(t, f, m)'//new_line('a')// &
            '  print *, r(1)'//new_line('a')// &
            'end program main'
        test_merge_nonconformable = expect_error_contains( &
            source, 'merge operands must share the result shape', &
            '/tmp/ffc_session_transform_merge_nonconformable')
    end function test_merge_nonconformable

end program test_session_array_transform_intrinsics_compiler
