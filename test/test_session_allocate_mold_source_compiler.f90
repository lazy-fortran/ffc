program test_session_allocate_mold_source
    ! ALLOCATE with MOLD= and SOURCE= on intrinsic allocatable arrays (#2820).
    ! MOLD copies the source's shape (contents are undefined). SOURCE copies
    ! both shape and element values.
    use ffc_test_support, only: expect_exit_status, expect_output, &
        expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session allocate mold/source compiler test ==='

    all_passed = .true.
    if (.not. test_mold_copies_shape()) all_passed = .false.
    if (.not. test_source_copies_values()) all_passed = .false.
    if (.not. test_rank2_mold_copies_shape()) all_passed = .false.
    if (.not. test_rank2_source_copies_values()) all_passed = .false.
    if (.not. test_rank3_mold_source_matches_gfortran()) all_passed = .false.
    if (.not. test_rank4_mold_source_matches_gfortran()) all_passed = .false.
    if (.not. test_issue_2820_roundtrip()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocate mold=/source= lowers through LIRIC'

contains

    logical function test_mold_copies_shape()
        ! allocate(b, mold=a) gives b the same extent as a; size(b) = 3.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  allocate(b, mold=a)'//new_line('a')// &
            '  stop size(b)'//new_line('a')// &
            'end program main'

        test_mold_copies_shape = expect_exit_status( &
            source, 3, '/tmp/ffc_alloc_mold_shape')
    end function test_mold_copies_shape

    logical function test_source_copies_values()
        ! allocate(c, source=a) copies a's values; sum(c) = 6.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), c(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = 2'//new_line('a')// &
            '  allocate(c, source=a)'//new_line('a')// &
            '  stop c(1) + c(2) + c(3)'//new_line('a')// &
            'end program main'

        test_source_copies_values = expect_exit_status( &
            source, 6, '/tmp/ffc_alloc_source_values')
    end function test_source_copies_values

    logical function test_rank2_mold_copies_shape()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:), b(:,:)'//new_line('a')// &
            '  allocate(a(2,3))'//new_line('a')// &
            '  allocate(b, mold=a)'//new_line('a')// &
            '  stop size(b) + 10 * (size(b, 1) - 2) + '// &
            '       100 * (size(b, 2) - 3)'//new_line('a')// &
            'end program main'

        test_rank2_mold_copies_shape = expect_exit_status( &
            source, 6, '/tmp/ffc_alloc_mold_rank2_shape')
    end function test_rank2_mold_copies_shape

    logical function test_rank2_source_copies_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:), c(:,:)'//new_line('a')// &
            '  allocate(a(2,3))'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  a(2,2) = 4'//new_line('a')// &
            '  a(2,3) = 6'//new_line('a')// &
            '  allocate(c, source=a)'//new_line('a')// &
            '  stop c(1,1) + 10 * c(2,2) + 100 * c(2,3)'//new_line('a')// &
            'end program main'

        test_rank2_source_copies_values = expect_exit_status( &
            source, 129, '/tmp/ffc_alloc_source_rank2_values')
    end function test_rank2_source_copies_values

    logical function test_rank3_mold_source_matches_gfortran()
        ! Runtime descriptor extents and the values distinguish column-major
        ! storage from a mistaken row-major copy.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:,:), b(:,:,:), c(:,:,:)'//new_line('a')// &
            '  integer :: n, m, p'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  m = 2'//new_line('a')// &
            '  p = 3'//new_line('a')// &
            '  allocate(a(n,m,p))'//new_line('a')// &
            '  a(1,1,1) = 11'//new_line('a')// &
            '  a(2,1,1) = 12'//new_line('a')// &
            '  a(1,2,1) = 21'//new_line('a')// &
            '  a(2,2,1) = 22'//new_line('a')// &
            '  a(1,1,2) = 31'//new_line('a')// &
            '  a(2,1,2) = 32'//new_line('a')// &
            '  a(1,2,2) = 41'//new_line('a')// &
            '  a(2,2,2) = 42'//new_line('a')// &
            '  a(1,1,3) = 51'//new_line('a')// &
            '  a(2,1,3) = 52'//new_line('a')// &
            '  a(1,2,3) = 61'//new_line('a')// &
            '  a(2,2,3) = 62'//new_line('a')// &
            '  allocate(b, mold=a)'//new_line('a')// &
            '  allocate(c, source=a)'//new_line('a')// &
            '  print *, size(b,1), size(b,2), size(b,3)'//new_line('a')// &
            '  print *, c(1,1,1), c(2,1,1), c(1,2,1), c(2,2,1), '// &
            'c(1,1,2), c(2,1,2), c(1,2,2), c(2,2,2), '// &
            'c(1,1,3), c(2,1,3), c(1,2,3), c(2,2,3)'//new_line('a')// &
            'end program main'

        test_rank3_mold_source_matches_gfortran = expect_output_matches_gfortran( &
            source, 'alloc_mold_source_rank3')
    end function test_rank3_mold_source_matches_gfortran

    logical function test_rank4_mold_source_matches_gfortran()
        ! Runtime descriptor extents and column-major values exercise all four
        ! dimensions of MOLD= and SOURCE=.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:)'// &
            new_line('a')// &
            '  integer :: n, m, p, q, i, j, k, l, value'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  m = 2'//new_line('a')// &
            '  p = 2'//new_line('a')// &
            '  q = 2'//new_line('a')// &
            '  allocate(a(n,m,p,q))'//new_line('a')// &
            '  value = 101'//new_line('a')// &
            '  do l = 1, q'//new_line('a')// &
            '    do k = 1, p'//new_line('a')// &
            '      do j = 1, m'//new_line('a')// &
            '        do i = 1, n'//new_line('a')// &
            '          a(i,j,k,l) = value'//new_line('a')// &
            '          value = value + 1'//new_line('a')// &
            '        end do'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  allocate(b, mold=a)'//new_line('a')// &
            '  allocate(c, source=a)'//new_line('a')// &
            '  print *, size(b,1), size(b,2), size(b,3), size(b,4)'// &
            new_line('a')// &
            '  print *, c(1,1,1,1), c(2,1,1,1), c(1,2,1,1), '// &
            'c(2,2,1,1), c(1,1,2,1), c(2,1,2,1), c(1,2,2,1), '// &
            'c(2,2,2,1)'//new_line('a')// &
            '  print *, c(1,1,1,2), c(2,1,1,2), c(1,2,1,2), '// &
            'c(2,2,1,2), c(1,1,2,2), c(2,1,2,2), c(1,2,2,2), '// &
            'c(2,2,2,2)'//new_line('a')// &
            'end program main'

        test_rank4_mold_source_matches_gfortran = expect_output_matches_gfortran( &
            source, 'alloc_mold_source_rank4')
    end function test_rank4_mold_source_matches_gfortran

    logical function test_issue_2820_roundtrip()
        ! The corpus program: print size(b) then c (all ones).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:), c(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  allocate(b, mold=a)'//new_line('a')// &
            '  allocate(c, source=a)'//new_line('a')// &
            '  print *, size(b), c'//new_line('a')// &
            'end program main'

        test_issue_2820_roundtrip = expect_output( &
            source, '           3           1           1           1'// &
            new_line('a'), '/tmp/ffc_alloc_mold_source_roundtrip')
    end function test_issue_2820_roundtrip

end program test_session_allocate_mold_source
