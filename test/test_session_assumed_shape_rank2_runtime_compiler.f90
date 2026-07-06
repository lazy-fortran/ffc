program test_session_assumed_shape_rank2_runtime
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-2 assumed-shape runtime-extent test ==='

    all_passed = .true.
    if (.not. test_integer_rank2_actual()) all_passed = .false.
    if (.not. test_real_rank2_write()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: rank-2 assumed-shape dummies read runtime allocatable extents'

contains

    logical function test_integer_rank2_actual()
        ! A rank-2 allocatable actual has no compile-time-foldable shape, so both
        ! per-dimension extents travel as hidden i64 arguments. size()/size(dim)/
        ! ubound(dim) and column-major element access all read the runtime extent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, allocatable :: m(:,:)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'allocate(m(2,3))'//new_line('a')// &
            'do j = 1, 3'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'm(i,j) = i*10 + j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call use_it(m)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine use_it(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a,1), size(a,2), size(a)'//new_line('a')// &
            'print *, ubound(a,1), ubound(a,2)'//new_line('a')// &
            'print *, a(1,1), a(2,3), a(2,1)'//new_line('a')// &
            'end subroutine use_it'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2           3           6'//new_line('a')// &
            '           2           3'//new_line('a')// &
            '          11          23          21'//new_line('a')

        test_integer_rank2_actual = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_rank2_int')
    end function test_integer_rank2_actual

    logical function test_real_rank2_write()
        ! An intent(inout) rank-2 assumed-shape dummy filled with a runtime
        ! column-major element write, then read back through the same runtime
        ! stride in the caller after the call returns.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real, allocatable :: b(:,:)'//new_line('a')// &
            'allocate(b(3,4))'//new_line('a')// &
            'call fill(b)'//new_line('a')// &
            'print *, b(1,1), b(3,4), b(2,3)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine fill(a)'//new_line('a')// &
            'real, intent(inout) :: a(:,:)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'do j = 1, size(a,2)'//new_line('a')// &
            'do i = 1, size(a,1)'//new_line('a')// &
            'a(i,j) = real(i*10 + j)'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'end subroutine fill'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   11.0000000       34.0000000       23.0000000    '// &
            new_line('a')

        test_real_rank2_write = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_rank2_real')
    end function test_real_rank2_write

end program test_session_assumed_shape_rank2_runtime
