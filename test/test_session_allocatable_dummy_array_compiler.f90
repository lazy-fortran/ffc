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

end program test_session_allocatable_dummy_array_compiler
