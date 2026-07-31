program test_session_derived_pointer_component
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    print *, '=== direct session derived pointer component compiler test ==='

    ! A scalar pointer component associates with a target, mutates it, and
    ! reports association until nullify.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'type box_t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'end type box_t'//new_line('a')// &
        'type(box_t) :: b'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        't = 5'//new_line('a')// &
        'b%p => t'//new_line('a')// &
        'if (.not. associated(b%p)) stop 1'//new_line('a')// &
        'b%p = 42'//new_line('a')// &
        'nullify(b%p)'//new_line('a')// &
        'if (associated(b%p)) stop 2'//new_line('a')// &
        'stop t'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_session_derived_pointer_component_test')) stop 1

    ! A read through the component observes a later write to the target.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'type box_t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'end type box_t'//new_line('a')// &
        'type(box_t) :: b'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        'b%p => t'//new_line('a')// &
        't = 9'//new_line('a')// &
        'stop b%p'//new_line('a')// &
        'end program main', 9, &
        '/tmp/ffc_session_derived_pointer_component_read')) stop 1

    ! Intrinsic assignment of the derived object copies association, not
    ! pointee storage: c%p aliases the same target as b%p.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'type box_t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'end type box_t'//new_line('a')// &
        'type(box_t) :: b, c'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        't = 1'//new_line('a')// &
        'b%p => t'//new_line('a')// &
        'c = b'//new_line('a')// &
        'c%p = 33'//new_line('a')// &
        'stop t'//new_line('a')// &
        'end program main', 33, &
        '/tmp/ffc_session_derived_pointer_component_assign')) stop 1

    ! Association of a pointer component to a non-TARGET is rejected.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'type box_t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'end type box_t'//new_line('a')// &
        'type(box_t) :: b'//new_line('a')// &
        'integer :: i'//new_line('a')// &
        'b%p => i'//new_line('a')// &
        'end program main', 'right of => is not a target or pointer', &
        '/tmp/ffc_session_derived_pointer_component_negative')) stop 1

    print *, 'PASS: scalar derived-type pointer components'
end program test_session_derived_pointer_component
