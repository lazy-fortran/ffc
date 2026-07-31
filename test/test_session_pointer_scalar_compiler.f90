program test_session_pointer_scalar
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    print *, '=== direct session scalar pointer/target compiler test ==='

    ! p => t then a write through p mutates t; associated(p) is true until
    ! nullify(p). Final stop returns t, which the pointer write set to 42.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        't = 5'//new_line('a')// &
        'p => t'//new_line('a')// &
        'p = 42'//new_line('a')// &
        'if (.not. associated(p)) stop 1'//new_line('a')// &
        'nullify(p)'//new_line('a')// &
        'if (associated(p)) stop 2'//new_line('a')// &
        'stop t'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_session_pointer_scalar_test')) stop 1

    ! A read through the pointer observes a later write to the target.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'p => t'//new_line('a')// &
        't = 9'//new_line('a')// &
        'stop p'//new_line('a')// &
        'end program main', 9, &
        '/tmp/ffc_session_pointer_read_test')) stop 1

    ! A bare POINTER statement uses the implicit integer type rule for iptr.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: i'//new_line('a')// &
        'pointer :: iptr'//new_line('a')// &
        'i = 7'//new_line('a')// &
        'iptr => i'//new_line('a')// &
        'iptr = 11'//new_line('a')// &
        'stop i'//new_line('a')// &
        'end program main', 11, &
        '/tmp/ffc_session_pointer_statement_test')) stop 1

    ! A bare pointer still rejects association to a non-TARGET entity.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer :: i'//new_line('a')// &
        'pointer :: iptr'//new_line('a')// &
        'iptr => i'//new_line('a')// &
        'end program main', 'right of => is not a target or pointer', &
        '/tmp/ffc_session_pointer_statement_negative')) stop 1

    print *, 'PASS: scalar integer pointer/target, => , associated, nullify'
end program test_session_pointer_scalar
