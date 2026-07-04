program test_session_assumed_shape_section
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session assumed-shape array-section actual test ==='

    all_passed = .true.
    if (.not. test_rank1_integer_section()) all_passed = .false.
    if (.not. test_rank2_column_section()) all_passed = .false.
    if (.not. test_rank1_real_section()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array-section actuals bind rank-1 assumed-shape dummies'

contains

    logical function test_rank1_integer_section()
        ! a(2:4) passed to a rank-1 assumed-shape dummy: the callee's size() and
        ! element access read the contiguous slice of the caller's storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(6), s'//new_line('a')// &
            'a = [10, 20, 30, 40, 50, 60]'//new_line('a')// &
            'call sum_sec(a(2:4), s)'//new_line('a')// &
            'print *, s'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine sum_sec(v, total)'//new_line('a')// &
            'integer, intent(in) :: v(:)'//new_line('a')// &
            'integer, intent(out) :: total'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do i = 1, size(v)'//new_line('a')// &
            'total = total + v(i)'//new_line('a')// &
            'end do'//new_line('a')// &
            'end subroutine sum_sec'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          90'//new_line('a')

        test_rank1_integer_section = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_section_r1i')
    end function test_rank1_integer_section

    logical function test_rank2_column_section()
        ! A whole column y(:,2) of a rank-2 array is contiguous, so it binds a
        ! rank-1 dummy through its base pointer; size() reports the column
        ! extent and element access reads the caller's column in place.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: y(2,3), i, j'//new_line('a')// &
            'do j = 1, 3'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'y(i,j) = i + 10*j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call show(y(:,2))'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(v)'//new_line('a')// &
            'integer, intent(in) :: v(:)'//new_line('a')// &
            'print *, size(v), v(1), v(2)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2          21          22'//new_line('a')

        test_rank2_column_section = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_section_r2c')
    end function test_rank2_column_section

    logical function test_rank1_real_section()
        ! A stride-1 real section r(2:3) binds a rank-1 real assumed-shape
        ! dummy; sum() folds over the caller-derived extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real :: r(4)'//new_line('a')// &
            'r = [1.5, 2.5, 3.5, 4.5]'//new_line('a')// &
            'call rshow(r(2:3))'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine rshow(v)'//new_line('a')// &
            'real, intent(in) :: v(:)'//new_line('a')// &
            'print *, size(v), sum(v)'//new_line('a')// &
            'end subroutine rshow'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2   6.00000000    '//new_line('a')

        test_rank1_real_section = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_section_r1f')
    end function test_rank1_real_section

end program test_session_assumed_shape_section
