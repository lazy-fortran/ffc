program test_session_dummy_array_bound_call_site
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session dummy array-bound call-site fold compiler test ==='

    all_passed = .true.
    if (.not. test_size_actual_same_name_as_dummy()) all_passed = .false.
    if (.not. test_size_dim_actual_same_name_as_dummy()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: adjustable-array dummy bound folds via call-site size() '// &
        'even when the actual shares its name with a callee dummy'

contains

    logical function test_size_actual_same_name_as_dummy()
        ! An adjustable-array dummy bound (`arr(n)`) is folded from the
        ! call-site actual `size(a)`. When the caller's array and the
        ! callee's own array dummy share the same name ("a"), the callee's
        ! own not-yet-resolved dummy must not shadow the caller's array when
        ! recovering its declared extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real :: a(3), total'//new_line('a')// &
            'a(1) = 3'//new_line('a')// &
            'a(2) = 2'//new_line('a')// &
            'a(3) = 1'//new_line('a')// &
            'call mysum(size(a), a, total)'//new_line('a')// &
            'print *, total'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine mysum(na1, a, r)'//new_line('a')// &
            'integer, intent(in) :: na1'//new_line('a')// &
            'real, intent(in) :: a(na1)'//new_line('a')// &
            'real, intent(out) :: r'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'r = 0'//new_line('a')// &
            'do i = 1, size(a)'//new_line('a')// &
            'r = r + a(i)'//new_line('a')// &
            'end do'//new_line('a')// &
            'end subroutine mysum'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   6.00000000    '//new_line('a')

        test_size_actual_same_name_as_dummy = expect_output(source, expected, &
            '/tmp/ffc_session_dummy_array_bound_r1')
    end function test_size_actual_same_name_as_dummy

    logical function test_size_dim_actual_same_name_as_dummy()
        ! Same collision, with a two-argument size(a, dim=1) actual.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real :: a(3), total'//new_line('a')// &
            'a(1) = 3'//new_line('a')// &
            'a(2) = 2'//new_line('a')// &
            'a(3) = 1'//new_line('a')// &
            'call mysum(size(a, dim=1), a, total)'//new_line('a')// &
            'print *, total'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine mysum(na1, a, r)'//new_line('a')// &
            'integer, intent(in) :: na1'//new_line('a')// &
            'real, intent(in) :: a(na1)'//new_line('a')// &
            'real, intent(out) :: r'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'r = 0'//new_line('a')// &
            'do i = 1, size(a)'//new_line('a')// &
            'r = r + a(i)'//new_line('a')// &
            'end do'//new_line('a')// &
            'end subroutine mysum'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   6.00000000    '//new_line('a')

        test_size_dim_actual_same_name_as_dummy = expect_output(source, expected, &
            '/tmp/ffc_session_dummy_array_bound_r2')
    end function test_size_dim_actual_same_name_as_dummy

end program test_session_dummy_array_bound_call_site
