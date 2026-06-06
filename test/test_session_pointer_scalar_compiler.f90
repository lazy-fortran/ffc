program test_session_pointer_scalar
    use ffc_test_support, only: expect_exit_status
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

    print *, 'PASS: scalar integer pointer/target, => , associated, nullify'
end program test_session_pointer_scalar
