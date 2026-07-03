program test_session_assumed_shape_runtime_extent
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session assumed-shape runtime-extent compiler test ==='

    all_passed = .true.
    if (.not. test_allocatable_integer_actual()) all_passed = .false.
    if (.not. test_allocatable_real_actual()) all_passed = .false.
    if (.not. test_type_bound_call_actual()) all_passed = .false.

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

    logical function test_type_bound_call_actual()
        ! A type-bound subroutine call `call me%worker(x, res)` inserts the
        ! passed-object receiver ("me") ahead of the explicit call arguments
        ! at the callee's dummy position, so an explicit argument's call-site
        ! position does not equal its callee dummy position. The hidden
        ! runtime-extent argument lookup must use the dummy position, not the
        ! call-site position, or it mistakes a later scalar argument for the
        ! assumed-shape actual.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: holder'//new_line('a')// &
            '    integer :: base'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: worker => holder_worker'//new_line('a')// &
            '  end type holder'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine holder_worker(this, x, res)'//new_line('a')// &
            '    class(holder), intent(in) :: this'//new_line('a')// &
            '    real, intent(in) :: x(:)'//new_line('a')// &
            '    real, intent(out) :: res'//new_line('a')// &
            '    res = x(1) + real(this%base)'//new_line('a')// &
            '  end subroutine holder_worker'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(holder) :: h'//new_line('a')// &
            '  real, allocatable :: x(:)'//new_line('a')// &
            '  real :: res'//new_line('a')// &
            '  h%base = 5'//new_line('a')// &
            '  allocate(x(3))'//new_line('a')// &
            '  x = [1.0, 2.0, 3.0]'//new_line('a')// &
            '  call h%worker(x, res)'//new_line('a')// &
            '  print *, res'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   6.00000000    '//new_line('a')

        test_type_bound_call_actual = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_runtime_typebound')
    end function test_type_bound_call_actual

end program test_session_assumed_shape_runtime_extent
