program test_session_assumed_size_array
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session assumed-size array compiler test ==='

    all_passed = .true.
    if (.not. test_rank1_bare_star()) all_passed = .false.
    if (.not. test_rank2_last_dim_star()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: assumed-size dummies lower through direct LIRIC'

contains

    logical function test_rank1_bare_star()
        ! Rank-1 assumed-size dummy a(*): element access and writes by loop
        ! index read and write the caller's contiguous storage in place.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: arr(4)'//new_line('a')// &
            'arr = [3, 6, 9, 12]'//new_line('a')// &
            'call bump(arr, 4)'//new_line('a')// &
            'print *, arr(1), arr(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine bump(a, n)'//new_line('a')// &
            'integer, intent(inout) :: a(*)'//new_line('a')// &
            'integer, intent(in) :: n'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'do i = 1, n'//new_line('a')// &
            'a(i) = a(i) + 1'//new_line('a')// &
            'end do'//new_line('a')// &
            'end subroutine bump'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4          13'//new_line('a')

        test_rank1_bare_star = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_size_r1')
    end function test_rank1_bare_star

    logical function test_rank2_last_dim_star()
        ! Rank-2 dummy a(10, *): the leading dimension folds to a compile-time
        ! stride; the trailing asterisk carries no extent. lbound/ubound on
        ! the bounded dimensions and element read/write through it work.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real :: arr(10, 20)'//new_line('a')// &
            'call fill(arr)'//new_line('a')// &
            'print *, arr(1,1), arr(10,2)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine fill(a)'//new_line('a')// &
            'real :: a(10, *)'//new_line('a')// &
            'if (lbound(a, 1) /= 1) error stop'//new_line('a')// &
            'if (ubound(a, 1) /= 10) error stop'//new_line('a')// &
            'if (lbound(a, 2) /= 1) error stop'//new_line('a')// &
            'a(1,1) = 2.5'//new_line('a')// &
            'a(10,2) = 7.5'//new_line('a')// &
            'end subroutine'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   2.50000000       7.50000000    '//new_line('a')

        test_rank2_last_dim_star = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_size_r2')
    end function test_rank2_last_dim_star

end program test_session_assumed_size_array
