program test_session_allocatable_dummy_array_compiler
    ! An allocatable array dummy argument (W11): the parameter symbol is
    ! pre-registered generically at call-entry binding, so a naive
    ! re-declaration used to hit a false-positive "duplicate allocatable
    ! declaration" error. The callee now aliases the caller's own descriptor,
    ! so allocate/deallocate and element writes inside the callee land back
    ! in the caller.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable dummy array compiler test ==='

    all_passed = .true.
    if (.not. test_allocate_intent_out_dummy()) all_passed = .false.
    if (.not. test_intent_inout_dummy_read()) all_passed = .false.
    if (.not. test_default_intent_dummy_roundtrip()) all_passed = .false.
    if (.not. test_rank2_allocatable_as_assumed_shape()) all_passed = .false.
    if (.not. test_double_deallocate_is_not_a_double_free()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable array dummies alias the caller descriptor'

contains

    logical function test_allocate_intent_out_dummy()
        ! allocate() inside an intent(out) callee is visible to the caller.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  call fill(a)'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill(x)'//new_line('a')// &
            '    integer, allocatable, intent(out) :: x(:)'//new_line('a')// &
            '    allocate(x(3))'//new_line('a')// &
            '    x(1) = 1'//new_line('a')// &
            '    x(2) = 2'//new_line('a')// &
            '    x(3) = 3'//new_line('a')// &
            '  end subroutine fill'//new_line('a')// &
            'end program main'

        test_allocate_intent_out_dummy = expect_output( &
            source, '           1           2           3'//new_line('a'), &
            '/tmp/ffc_alloc_dummy_out_test')
    end function test_allocate_intent_out_dummy

    logical function test_intent_inout_dummy_read()
        ! Element writes/reads inside a plain (default-intent) callee land in
        ! the caller's own storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a(1) = 42'//new_line('a')// &
            '  call touch(a)'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine touch(x)'//new_line('a')// &
            '    integer, allocatable, intent(inout) :: x(:)'//new_line('a')// &
            '    integer :: v'//new_line('a')// &
            '    v = x(1)'//new_line('a')// &
            '    x(1) = v + 1'//new_line('a')// &
            '  end subroutine touch'//new_line('a')// &
            'end program main'

        test_intent_inout_dummy_read = expect_output( &
            source, '          43'//new_line('a'), &
            '/tmp/ffc_alloc_dummy_inout_test')
    end function test_intent_inout_dummy_read

    logical function test_default_intent_dummy_roundtrip()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  call fill(a)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill(x)'//new_line('a')// &
            '    integer, allocatable :: x(:)'//new_line('a')// &
            '    allocate(x(2))'//new_line('a')// &
            '  end subroutine fill'//new_line('a')// &
            'end program main'

        test_default_intent_dummy_roundtrip = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_dummy_default_test')
    end function test_default_intent_dummy_roundtrip

    logical function test_rank2_allocatable_as_assumed_shape()
        ! Descriptor interoperability across the two migrated paths: a rank-2
        ! allocatable, whose shape now lives in a canonical array descriptor
        ! (#336), is passed to an assumed-shape dummy, which is bound through a
        ! canonical descriptor too (#334). Both agree on extents, on
        ! column-major order, and on the address of the last element, which is
        ! only true if exactly one layout is in use on both sides.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'integer, allocatable :: m(:,:)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'allocate(m(2,3))'//new_line('a')// &
            'do j = 1, 3'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'm(i,j) = i*10 + j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call show(m)'//new_line('a')// &
            'print *, size(m), m(2,3), m(1,1)'//new_line('a')// &
            'deallocate(m)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine show(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a), size(a,1), size(a,2), a(size(a,1), size(a,2))'// &
            new_line('a')// &
            'end subroutine show'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           6           2           3          23'//new_line('a')// &
            '           6          23          11'//new_line('a')

        test_rank2_allocatable_as_assumed_shape = expect_output(source, &
            expected, '/tmp/ffc_alloc_rank2_assumed_shape_test')
    end function test_rank2_allocatable_as_assumed_shape

    logical function test_double_deallocate_is_not_a_double_free()
        ! Negative control for the descriptor's allocation state. DEALLOCATE
        ! returns the descriptor to the unallocated state (null base, cleared
        ! flags, zero extents), so a second DEALLOCATE frees nothing rather
        ! than handing the same heap block to the deallocator twice.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'integer, allocatable :: v(:)'//new_line('a')// &
            'allocate(v(4))'//new_line('a')// &
            'v(1) = 7'//new_line('a')// &
            'print *, size(v), v(1)'//new_line('a')// &
            'deallocate(v)'//new_line('a')// &
            'deallocate(v)'//new_line('a')// &
            'print *, 42'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           4           7'//new_line('a')// &
            '          42'//new_line('a')

        test_double_deallocate_is_not_a_double_free = expect_output(source, &
            expected, '/tmp/ffc_alloc_double_deallocate_test')
    end function test_double_deallocate_is_not_a_double_free

end program test_session_allocatable_dummy_array_compiler
