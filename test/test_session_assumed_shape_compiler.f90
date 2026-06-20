program test_session_assumed_shape
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session assumed-shape array compiler test ==='

    all_passed = .true.
    if (.not. test_rank1_integer()) all_passed = .false.
    if (.not. test_rank1_real_sum()) all_passed = .false.
    if (.not. test_rank2_integer()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: assumed-shape dummies lower through direct LIRIC'

contains

    logical function test_rank1_integer()
        ! Whole array passed to a(:); size() and element access by loop index
        ! read the caller's contiguous storage in place.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: arr(4)'//new_line('a')// &
            'arr = [3, 6, 9, 12]'//new_line('a')// &
            'call show(arr)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'integer, intent(in) :: a(:)'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'do i = 1, size(a)'//new_line('a')// &
            'print *, a(i)'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, sum(a)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4'//new_line('a')// &
            '           3'//new_line('a')// &
            '           6'//new_line('a')// &
            '           9'//new_line('a')// &
            '          12'//new_line('a')// &
            '          30'//new_line('a')

        test_rank1_integer = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_r1i')
    end function test_rank1_integer

    logical function test_rank1_real_sum()
        ! Assumed-shape real dummy: size() and sum() over the caller extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real :: arr(3)'//new_line('a')// &
            'arr = [1.5, 2.5, 4.0]'//new_line('a')// &
            'call show(arr)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'real, intent(in) :: a(:)'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'print *, sum(a)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           3'//new_line('a')// &
            '   8.00000000    '//new_line('a')

        test_rank1_real_sum = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_r1f')
    end function test_rank1_real_sum

    logical function test_rank2_integer()
        ! Rank-2 assumed-shape dummy a(:,:): size() and column-major element
        ! addressing use the caller-derived extents.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: m(2,2)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'do j = 1, 2'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'm(i,j) = i*10 + j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call show(m)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'print *, a(1,1)'//new_line('a')// &
            'print *, a(2,2)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4'//new_line('a')// &
            '          11'//new_line('a')// &
            '          22'//new_line('a')

        test_rank2_integer = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_r2i')
    end function test_rank2_integer

end program test_session_assumed_shape
