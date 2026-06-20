program test_session_allocate_mold_source
    ! ALLOCATE with MOLD= and SOURCE= on 1-D integer allocatables (#2820).
    ! MOLD copies the source's shape (contents are undefined). SOURCE copies
    ! both shape and element values.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session allocate mold/source compiler test ==='

    all_passed = .true.
    if (.not. test_mold_copies_shape()) all_passed = .false.
    if (.not. test_source_copies_values()) all_passed = .false.
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
