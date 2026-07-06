program test_session_runtime_local_array
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session runtime-sized local array compiler test ==='

    all_passed = .true.
    if (.not. test_integer_automatic_array()) all_passed = .false.
    if (.not. test_real_broadcast_and_copy()) all_passed = .false.
    if (.not. test_lower_bound_array()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime-sized local automatic arrays lower end to end'

contains

    logical function test_integer_automatic_array()
        ! integer :: b(m) with m a dummy value only known at run time. Storage is
        ! a dynamic alloca; element write, size(), sum(), and whole-array print
        ! all walk the runtime element count.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 4'//new_line('a')// &
            'call go(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine go(m)'//new_line('a')// &
            'integer, intent(in) :: m'//new_line('a')// &
            'integer :: b(m)'//new_line('a')// &
            'integer :: j'//new_line('a')// &
            'do j = 1, m'//new_line('a')// &
            'b(j) = j * 10'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, size(b)'//new_line('a')// &
            'print *, sum(b)'//new_line('a')// &
            'print *, b'//new_line('a')// &
            'end subroutine go'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4'//new_line('a')// &
            '         100'//new_line('a')// &
            '          10          20          30          40'//new_line('a')

        test_integer_automatic_array = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_i')
    end function test_integer_automatic_array

    logical function test_real_broadcast_and_copy()
        ! real :: a(m), b(m): whole-array scalar broadcast (a = 2.5) and a
        ! whole-array copy (b = a), both driven by a runtime loop.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 3'//new_line('a')// &
            'call go(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine go(m)'//new_line('a')// &
            'integer, intent(in) :: m'//new_line('a')// &
            'real :: a(m)'//new_line('a')// &
            'real :: b(m)'//new_line('a')// &
            'a = 2.5'//new_line('a')// &
            'b = a'//new_line('a')// &
            'print *, a'//new_line('a')// &
            'print *, b'//new_line('a')// &
            'end subroutine go'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   2.50000000       2.50000000       2.50000000    '//new_line('a')// &
            '   2.50000000       2.50000000       2.50000000    '//new_line('a')

        test_real_broadcast_and_copy = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_r')
    end function test_real_broadcast_and_copy

    logical function test_lower_bound_array()
        ! real(8) :: d(0:m): a runtime upper bound with a constant lower bound.
        ! lbound/ubound/size honour the lower bound; element access indexes from
        ! the declared lower bound.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: n'//new_line('a')// &
            'n = 5'//new_line('a')// &
            'call go(n)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine go(m)'//new_line('a')// &
            'integer, intent(in) :: m'//new_line('a')// &
            'real(8) :: d(0:m)'//new_line('a')// &
            'integer :: j'//new_line('a')// &
            'do j = 0, m'//new_line('a')// &
            'd(j) = j * 1.0d0'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, size(d), lbound(d, 1), ubound(d, 1)'//new_line('a')// &
            'print *, d(0), d(5)'//new_line('a')// &
            'end subroutine go'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           6           0           5'//new_line('a')// &
            '   0.0000000000000000        5.0000000000000000     '//new_line('a')

        test_lower_bound_array = expect_output(source, expected, &
            '/tmp/ffc_session_runtime_local_array_lb')
    end function test_lower_bound_array

end program test_session_runtime_local_array
