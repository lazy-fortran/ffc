program test_session_assumed_shape_rank2_runtime
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session rank-2 assumed-shape runtime-extent test ==='

    all_passed = .true.
    if (.not. test_integer_rank2_actual()) all_passed = .false.
    if (.not. test_real_rank2_write()) all_passed = .false.
    if (.not. test_two_call_sites_differ()) all_passed = .false.
    if (.not. test_rank2_actual_extents_and_last_element()) all_passed = .false.
    if (.not. test_nonunit_actual_bounds_rebind_to_one()) all_passed = .false.
    if (.not. test_rank1_actual_to_rank2_dummy_rejected()) all_passed = .false.

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

    logical function test_two_call_sites_differ()
        ! Two calls to the same procedure with differently sized actuals. Each
        ! call builds its own descriptor, so each callee invocation sees its own
        ! extent. Before #334 the callee folded ONE call site's compile-time
        ! shape for every call, so the second call reported the first call's
        ! size.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: p(3), q(5)'//new_line('a')// &
            'p = 1'//new_line('a')// &
            'q = 2'//new_line('a')// &
            'call show(p)'//new_line('a')// &
            'call show(q)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'integer, intent(in) :: a(:)'//new_line('a')// &
            'print *, size(a), a(1), sum(a)'//new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           3           1           3'//new_line('a')// &
            '           5           2          10'//new_line('a')

        test_two_call_sites_differ = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_two_sites')
    end function test_two_call_sites_differ

    logical function test_rank2_actual_extents_and_last_element()
        ! A runtime-sized rank-2 allocatable actual: the callee reports both
        ! extents and the last element, all read from the caller''s descriptor.
        ! Column-major order is preserved, so element (n1, n2) is the last one.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, allocatable :: m(:,:)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'allocate(m(4, 3))'//new_line('a')// &
            'do j = 1, 3'//new_line('a')// &
            'do i = 1, 4'//new_line('a')// &
            'm(i,j) = i*10 + j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call report(m)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine report(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a,1), size(a,2)'//new_line('a')// &
            'print *, a(size(a,1), size(a,2))'//new_line('a')// &
            'end subroutine report'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4           3'//new_line('a')// &
            '          43'//new_line('a')

        test_rank2_actual_extents_and_last_element = expect_output(source, &
            expected, '/tmp/ffc_session_assumed_shape_rank2_last')
    end function test_rank2_actual_extents_and_last_element

    logical function test_nonunit_actual_bounds_rebind_to_one()
        ! An actual allocated with non-unit lower bounds still presents lower
        ! bound 1 to an assumed-shape dummy (F2018 8.5.8.3), while extents and
        ! element identity are unchanged: a(1,1) is the actual''s first element
        ! in column-major order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: m(0:1, -1:1)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'do j = -1, 1'//new_line('a')// &
            'do i = 0, 1'//new_line('a')// &
            'm(i,j) = (i+1)*10 + (j+2)'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call bounds(m)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine bounds(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, lbound(a,1), lbound(a,2)'//new_line('a')// &
            'print *, ubound(a,1), ubound(a,2)'//new_line('a')// &
            'print *, a(1,1), a(2,3)'//new_line('a')// &
            'end subroutine bounds'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           1           1'//new_line('a')// &
            '           2           3'//new_line('a')// &
            '          11          23'//new_line('a')

        test_nonunit_actual_bounds_rebind_to_one = expect_output(source, &
            expected, '/tmp/ffc_session_assumed_shape_nonunit')
    end function test_nonunit_actual_bounds_rebind_to_one

    logical function test_rank1_actual_to_rank2_dummy_rejected()
        ! Negative: a rank-1 actual bound to a rank-2 assumed-shape dummy is a
        ! rank mismatch (F2018 15.5.2.4), diagnosed while the caller builds the
        ! descriptor rather than lowered into a wrong-shaped call.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: v(6)'//new_line('a')// &
            'v = 1'//new_line('a')// &
            'call take2(v)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine take2(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a)'//new_line('a')// &
            'end subroutine take2'//new_line('a')// &
            'end program main'

        test_rank1_actual_to_rank2_dummy_rejected = expect_error_contains( &
            source, 'Rank mismatch in argument to take2', &
            '/tmp/ffc_session_assumed_shape_rank_mismatch')
    end function test_rank1_actual_to_rank2_dummy_rejected

end program test_session_assumed_shape_rank2_runtime
