! fo: dispatcher
module ffc_case_test_session_assumed_shape_section_compiler
    implicit none
    private
    public :: case_test_session_assumed_shape_section_compiler
    interface
        subroutine case_test_session_assumed_shape_section_compiler()
        end subroutine case_test_session_assumed_shape_section_compiler
    end interface
end module ffc_case_test_session_assumed_shape_section_compiler

subroutine case_test_session_assumed_shape_section_compiler()
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none
    save

    logical :: all_passed

    print *, '=== direct session assumed-shape array-section actual test ==='

    all_passed = .true.
    if (.not. test_rank1_integer_section()) all_passed = .false.
    if (.not. test_rank2_column_section()) all_passed = .false.
    if (.not. test_rank1_real_section()) all_passed = .false.
    if (.not. test_live_stride_and_shifted_bounds()) all_passed = .false.
    if (.not. test_pointer_to_borrowed_view()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array-section actuals bind rank-1 assumed-shape dummies'

contains

    logical function test_live_stride_and_shifted_bounds() result(ok)
        ! One callee sees changing positive/negative strides and extents. Its
        ! shifted bounds and writes must preserve the caller's borrowed view.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(8), b(2,3), i, j, lo, hi, step'//new_line('a')// &
            'a = [10,20,30,40,50,60,70,80]'//new_line('a')// &
            'lo = 8'//new_line('a')// &
            'hi = 2'//new_line('a')// &
            'step = -2'//new_line('a')// &
            'call inspect(a(lo:hi:step))'//new_line('a')// &
            'print *, a'//new_line('a')// &
            'lo = 2'//new_line('a')// &
            'hi = 8'//new_line('a')// &
            'step = 3'//new_line('a')// &
            'call inspect(a(lo:hi:step))'//new_line('a')// &
            'print *, a'//new_line('a')// &
            'do j = 1, 3'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'b(i,j) = 100*j + i'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call inspect(b(:,2))'//new_line('a')// &
            'print *, b'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine inspect(v)'//new_line('a')// &
            'integer, intent(inout) :: v(-3:)'//new_line('a')// &
            'integer :: k, total'//new_line('a')// &
            'print *, lbound(v,1), ubound(v,1), size(v)'//new_line('a')// &
            'total = 0'//new_line('a')// &
            'do k = -3, ubound(v,1)'//new_line('a')// &
            'total = total + v(k)'//new_line('a')// &
            'v(k) = v(k) + 1'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, total'//new_line('a')// &
            'end subroutine inspect'//new_line('a')// &
            'end program main'

        ok = expect_output_matches_gfortran(source, 'descriptor_stride_bounds')
    end function test_live_stride_and_shifted_bounds

    logical function test_pointer_to_borrowed_view() result(ok)
        ! Pointer association must obtain its source stride from the borrowed
        ! descriptor after the dummy's separate stride operand is retired.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, target :: a(8)'//new_line('a')// &
            'a = [10,20,30,40,50,60,70,80]'//new_line('a')// &
            'call touch(a(8:2:-2))'//new_line('a')// &
            'print *, a'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine touch(v)'//new_line('a')// &
            'integer, target, intent(inout) :: v(:)'//new_line('a')// &
            'integer, pointer :: p(:)'//new_line('a')// &
            'p => v'//new_line('a')// &
            'print *, p(1), p(2), p(3), p(4)'//new_line('a')// &
            'p(2) = 99'//new_line('a')// &
            'print *, v(2)'//new_line('a')// &
            'end subroutine touch'//new_line('a')// &
            'end program main'

        ok = expect_output_matches_gfortran(source, 'pointer_borrowed_stride')
    end function test_pointer_to_borrowed_view

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
            'integer, intent(in) :: v(0:)'//new_line('a')// &
            'print *, lbound(v,1), ubound(v,1), size(v), v(0), v(1)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           0           1           2          21          22'//new_line('a')

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

end subroutine case_test_session_assumed_shape_section_compiler
