program test_session_assumed_shape_module
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session assumed-shape module compiler test ==='

    all_passed = .true.
    if (.not. test_module_rank1_real()) all_passed = .false.
    if (.not. test_module_rank2_inquiry()) all_passed = .false.
    if (.not. test_param_dim_actual()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module assumed-shape dummies lower through direct LIRIC'

contains

    logical function test_module_rank1_real()
        ! A module procedure's assumed-shape dummy: FortFront leaves
        ! is_array_access unset on element uses inside a module procedure, so
        ! the symbol-table fallback must still recognise a(2) as element access.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'real, intent(in) :: a(:)'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'print *, a(2)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            'use m'//new_line('a')// &
            'real :: arr(3)'//new_line('a')// &
            'arr = [1.5, 2.5, 4.0]'//new_line('a')// &
            'call show(arr)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           3'//new_line('a')// &
            '   2.50000000    '//new_line('a')

        test_module_rank1_real = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_mod_r1f')
    end function test_module_rank1_real

    logical function test_module_rank2_inquiry()
        ! Rank-2 assumed-shape dummy in a module procedure: size(a, dim),
        ! lbound/ubound per dimension, and column-major element read, all driven
        ! by the caller-derived extents. The program also writes x(i,j) with the
        ! same is_array_access fallback on the assignment target.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'print *, size(a,1)'//new_line('a')// &
            'print *, size(a,2)'//new_line('a')// &
            'print *, lbound(a,1), ubound(a,1)'//new_line('a')// &
            'print *, lbound(a,2), ubound(a,2)'//new_line('a')// &
            'print *, a(2,3)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            'use m'//new_line('a')// &
            'integer :: x(3,4), i, j'//new_line('a')// &
            'do j = 1, 4'//new_line('a')// &
            'do i = 1, 3'//new_line('a')// &
            'x(i,j) = i*10 + j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call show(x)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          12'//new_line('a')// &
            '           3'//new_line('a')// &
            '           4'//new_line('a')// &
            '           1           3'//new_line('a')// &
            '           1           4'//new_line('a')// &
            '          23'//new_line('a')

        test_module_rank2_inquiry = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_mod_r2i')
    end function test_module_rank2_inquiry

    logical function test_param_dim_actual()
        ! The actual array's extent is a named constant declared in the caller's
        ! scope (dimension(n)); the callee resolves the assumed-shape extent by
        ! folding that constant from its PARAMETER declaration in the unit.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, parameter :: n = 4'//new_line('a')// &
            'integer :: x(n)'//new_line('a')// &
            'x = [10, 20, 30, 40]'//new_line('a')// &
            'print *, ssum(x)'//new_line('a')// &
            'contains'//new_line('a')// &
            'function ssum(a) result(s)'//new_line('a')// &
            'integer, intent(in) :: a(:)'//new_line('a')// &
            'integer :: s'//new_line('a')// &
            's = a(1) + a(2)'//new_line('a')// &
            'end function ssum'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          30'//new_line('a')

        test_param_dim_actual = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_param_dim')
    end function test_param_dim_actual

end program test_session_assumed_shape_module
