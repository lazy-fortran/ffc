program test_session_assumed_shape_runtime_extent
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session assumed-shape runtime-extent compiler test ==='

    all_passed = .true.
    if (.not. test_allocatable_integer_actual()) all_passed = .false.
    if (.not. test_allocatable_real_actual()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: assumed-shape dummies read a runtime allocatable extent'

contains

    logical function test_allocatable_integer_actual()
        ! An allocatable actual has no compile-time-foldable shape, so the
        ! assumed-shape dummy's extent travels as a hidden i64 argument (W2).
        ! size()/ubound()/element access/do-loop bound/sum() all read it.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, allocatable :: arr(:)'//new_line('a')// &
            'allocate(arr(5))'//new_line('a')// &
            'arr = [1, 2, 3, 4, 5]'//new_line('a')// &
            'call show(arr)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'integer, intent(in) :: a(:)'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'print *, ubound(a, 1)'//new_line('a')// &
            'do i = 1, size(a)'//new_line('a')// &
            'print *, a(i)'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, sum(a)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           5'//new_line('a')// &
            '           5'//new_line('a')// &
            '           1'//new_line('a')// &
            '           2'//new_line('a')// &
            '           3'//new_line('a')// &
            '           4'//new_line('a')// &
            '           5'//new_line('a')// &
            '          15'//new_line('a')

        test_allocatable_integer_actual = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_runtime_r1i')
    end function test_allocatable_integer_actual

    logical function test_allocatable_real_actual()
        ! Same runtime-extent ABI for a real assumed-shape dummy: size(),
        ! ubound(), element access, and a do-loop bound by size().
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real, allocatable :: arr(:)'//new_line('a')// &
            'allocate(arr(3))'//new_line('a')// &
            'arr = [1.5, 2.5, 4.0]'//new_line('a')// &
            'call show(arr)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'real, intent(in) :: a(:)'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'print *, ubound(a, 1)'//new_line('a')// &
            'do i = 1, size(a)'//new_line('a')// &
            'print *, a(i)'//new_line('a')// &
            'end do'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           3'//new_line('a')// &
            '           3'//new_line('a')// &
            '   1.50000000    '//new_line('a')// &
            '   2.50000000    '//new_line('a')// &
            '   4.00000000    '//new_line('a')

        test_allocatable_real_actual = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_runtime_r1f')
    end function test_allocatable_real_actual

end program test_session_assumed_shape_runtime_extent
