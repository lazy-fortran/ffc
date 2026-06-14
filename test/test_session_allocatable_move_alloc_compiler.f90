program test_session_allocatable_move_alloc
    ! move_alloc intrinsic on integer 1-D allocatables (#244 slice B2d).
    ! Transfers ownership: source becomes unallocated, destination holds the data.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable move_alloc compiler test ==='

    all_passed = .true.
    if (.not. test_move_alloc_basic()) all_passed = .false.
    if (.not. test_move_alloc_preserves_data()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: move_alloc on 1-D integer allocatable lowers through LIRIC'

contains

    logical function test_move_alloc_basic()
        ! Allocate src, move to dst, read dst(2) = 7.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: src(:), dst(:)'//new_line('a')// &
            '  allocate(src(3))'//new_line('a')// &
            '  src(2) = 7'//new_line('a')// &
            '  call move_alloc(src, dst)'//new_line('a')// &
            '  stop dst(2)'//new_line('a')// &
            'end program main'

        test_move_alloc_basic = expect_exit_status( &
            source, 7, '/tmp/ffc_move_alloc_basic')
    end function test_move_alloc_basic

    logical function test_move_alloc_preserves_data()
        ! Fill src with 1..4, move to dst, sum dst = 10.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: src(:), dst(:)'//new_line('a')// &
            '  integer :: i, total'//new_line('a')// &
            '  allocate(src(4))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    src(i) = i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call move_alloc(src, dst)'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    total = total + dst(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_move_alloc_preserves_data = expect_exit_status( &
            source, 10, '/tmp/ffc_move_alloc_data')
    end function test_move_alloc_preserves_data

end program test_session_allocatable_move_alloc
