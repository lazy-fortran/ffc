program test_session_pointer_scalar_kinds
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    print *, '=== direct session real/logical scalar pointer compiler test ==='

    ! A real pointer adopts the target slot; a later write to the target is
    ! observed when the pointer is dereferenced in an expression. int(p + 0.5)
    ! folds the read value to an exit code (matches gfortran).
    if (.not. expect_exit_status( &
         'program main'//new_line('a')// &
         'real, target :: t'//new_line('a')// &
         'real, pointer :: p'//new_line('a')// &
         't = 4.0'//new_line('a')// &
         'p => t'//new_line('a')// &
         't = 7.0'//new_line('a')// &
         'if (.not. associated(p)) stop 1'//new_line('a')// &
         'stop int(p + 0.5)'//new_line('a')// &
         'end program main', 7, &
         '/tmp/ffc_session_pointer_real_test')) stop 1

    ! A write through a real pointer mutates the target slot.
    if (.not. expect_exit_status( &
         'program main'//new_line('a')// &
         'real, target :: t'//new_line('a')// &
         'real, pointer :: p'//new_line('a')// &
         'p => t'//new_line('a')// &
         'p = 9.0'//new_line('a')// &
         'stop int(t)'//new_line('a')// &
         'end program main', 9, &
         '/tmp/ffc_session_pointer_real_write_test')) stop 2

    ! A logical pointer dereferenced in print observes the target value.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         'logical, target :: b'//new_line('a')// &
         'logical, pointer :: lp'//new_line('a')// &
         'b = .true.'//new_line('a')// &
         'lp => b'//new_line('a')// &
         'print *, lp'//new_line('a')// &
         'end program main', ' T'//new_line('a'), &
         '/tmp/ffc_session_pointer_logical_test')) stop 3

    print *, 'PASS: real and logical scalar pointer => , deref, associated'
end program test_session_pointer_scalar_kinds
