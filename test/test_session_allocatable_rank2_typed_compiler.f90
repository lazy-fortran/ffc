program test_session_allocatable_rank2_typed
    ! Rank-2 real(8) and logical allocatables (cluster 4 slice): allocate,
    ! element write/read, and fill-and-sum through the process exit status.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-2 typed allocatable compiler test ==='

    all_passed = .true.
    if (.not. test_real64_element_write_read()) all_passed = .false.
    if (.not. test_real64_fill_and_sum()) all_passed = .false.
    if (.not. test_logical_element_write_read()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: rank-2 real(8) and logical allocatables lower through LIRIC'

contains

    logical function test_real64_element_write_read()
        ! a(2,3) = 17.0d0; stop int(a(2,3))
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8), allocatable :: a(:,:)'//new_line('a')// &
            '  allocate(a(3,4))'//new_line('a')// &
            '  a(2,3) = 17.0d0'//new_line('a')// &
            '  stop int(a(2,3))'//new_line('a')// &
            'end program main'

        test_real64_element_write_read = expect_exit_status( &
            source, 17, '/tmp/ffc_alloc2d_r64_single')
    end function test_real64_element_write_read

    logical function test_real64_fill_and_sum()
        ! Fill a(i,j) = i*3 + j for 2x3, sum = 4+5+6+7+8+9 = 39.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8), allocatable :: a(:,:)'//new_line('a')// &
            '  integer :: i, j, total'//new_line('a')// &
            '  allocate(a(2,3))'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      a(i,j) = real(i * 3 + j, 8)'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      total = total + int(a(i,j))'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_real64_fill_and_sum = expect_exit_status( &
            source, 39, '/tmp/ffc_alloc2d_r64_sum')
    end function test_real64_fill_and_sum

    logical function test_logical_element_write_read()
        ! a(2,3) = .true.; stop 17 when read back true.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical, allocatable :: a(:,:)'//new_line('a')// &
            '  allocate(a(2,3))'//new_line('a')// &
            '  a(2,3) = .true.'//new_line('a')// &
            '  if (a(2,3)) then'//new_line('a')// &
            '    stop 17'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_logical_element_write_read = expect_exit_status( &
            source, 17, '/tmp/ffc_alloc2d_logical_single')
    end function test_logical_element_write_read

end program test_session_allocatable_rank2_typed
