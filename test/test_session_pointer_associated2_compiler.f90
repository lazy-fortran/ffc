program test_session_pointer_associated2
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session two-argument associated compiler test ==='

    ! B3c: associated(p, t) is true when p => t in straight-line code.
    if (.not. expect_exit_status( &
         'program main'//new_line('a')// &
         'implicit none'//new_line('a')// &
         'integer, target :: t'//new_line('a')// &
         'integer, target :: u'//new_line('a')// &
         'integer, pointer :: p'//new_line('a')// &
         't = 5'//new_line('a')// &
         'u = 7'//new_line('a')// &
         'p => t'//new_line('a')// &
         'if (.not. associated(p, t)) stop 1'//new_line('a')// &
         'if (associated(p, u)) stop 2'//new_line('a')// &
         'stop 0'//new_line('a')// &
         'end program main', 0, &
         '/tmp/ffc_associated2_test')) stop 1

    print *, 'PASS: two-argument associated(p, t)'
end program test_session_pointer_associated2
