program test_session_allocatable_rank2
    ! Rank-2 integer allocatable: allocate(a(m,n)), a(i,j) element access
    ! (#244 slice B2e). Fill in loops, read back, sum, and check via stop code.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable rank-2 compiler test ==='

    all_passed = .true.
    if (.not. test_element_write_read()) all_passed = .false.
    if (.not. test_fill_and_sum()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: rank-2 integer allocatable lowers through LIRIC'

contains

    logical function test_element_write_read()
        ! a(2,3) = 17; stop a(2,3)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:)'//new_line('a')// &
            '  allocate(a(3,4))'//new_line('a')// &
            '  a(2,3) = 17'//new_line('a')// &
            '  stop a(2,3)'//new_line('a')// &
            'end program main'

        test_element_write_read = expect_exit_status( &
            source, 17, '/tmp/ffc_alloc2d_single')
    end function test_element_write_read

    logical function test_fill_and_sum()
        ! Fill a(i,j) = i*3 + j for 2x3 matrix, sum = sum over all elements.
        ! a(1,1)=4 a(1,2)=5 a(1,3)=6  a(2,1)=7 a(2,2)=8 a(2,3)=9 -> sum=39
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:,:)'//new_line('a')// &
            '  integer :: i, j, total'//new_line('a')// &
            '  allocate(a(2,3))'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      a(i,j) = i * 3 + j'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      total = total + a(i,j)'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_fill_and_sum = expect_exit_status( &
            source, 39, '/tmp/ffc_alloc2d_sum')
    end function test_fill_and_sum

end program test_session_allocatable_rank2
