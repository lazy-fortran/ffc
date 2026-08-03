program test_session_pointer_function_result
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    print *, '=== direct session pointer function result compiler test ==='

    ! A pointer result returns the target address without copying the pointee:
    ! writing through the caller's pointer mutates the original target.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        't = 5'//new_line('a')// &
        'p => pick(t)'//new_line('a')// &
        'if (.not. associated(p)) stop 1'//new_line('a')// &
        'p = 42'//new_line('a')// &
        'stop t'//new_line('a')// &
        'contains'//new_line('a')// &
        'function pick(x) result(r)'//new_line('a')// &
        'integer, target, intent(in) :: x'//new_line('a')// &
        'integer, pointer :: r'//new_line('a')// &
        'r => x'//new_line('a')// &
        'end function pick'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_session_pointer_result_target')) stop 1

    ! A null pointer result is disassociated in the caller.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        't = 9'//new_line('a')// &
        'p => nulled(t)'//new_line('a')// &
        'if (associated(p)) stop 1'//new_line('a')// &
        'p => pick(t)'//new_line('a')// &
        'if (.not. associated(p)) stop 2'//new_line('a')// &
        'stop p'//new_line('a')// &
        'contains'//new_line('a')// &
        'function nulled(x) result(r)'//new_line('a')// &
        'integer, target, intent(in) :: x'//new_line('a')// &
        'integer, pointer :: r'//new_line('a')// &
        'r => null()'//new_line('a')// &
        'end function nulled'//new_line('a')// &
        'function pick(x) result(r)'//new_line('a')// &
        'integer, target, intent(in) :: x'//new_line('a')// &
        'integer, pointer :: r'//new_line('a')// &
        'r => x'//new_line('a')// &
        'end function pick'//new_line('a')// &
        'end program main', 9, &
        '/tmp/ffc_session_pointer_result_null')) stop 1

    ! A pointer result may allocate its own scalar target before returning.
    ! The caller must receive that target address, not a copied/temporary value.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'p => make(ll(40))'//new_line('a')// &
        'if (.not. associated(p)) stop 1'//new_line('a')// &
        'stop p'//new_line('a')// &
        'contains'//new_line('a')// &
        'elemental function ll(i)'//new_line('a')// &
        'integer, intent(in) :: i'//new_line('a')// &
        'integer :: ll'//new_line('a')// &
        'll = i + 1'//new_line('a')// &
        'end function ll'//new_line('a')// &
        'function make(i) result(r)'//new_line('a')// &
        'integer, intent(in) :: i'//new_line('a')// &
        'integer, pointer :: r'//new_line('a')// &
        'allocate (r)'//new_line('a')// &
        'r = i'//new_line('a')// &
        'end function make'//new_line('a')// &
        'end program main', 41, &
        '/tmp/ffc_session_pointer_result_allocated')) stop 1

    ! Returning a pointer to a local automatic target is statically detectable
    ! and rejected: that storage dies with the function invocation.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'p => stale()'//new_line('a')// &
        'contains'//new_line('a')// &
        'function stale() result(r)'//new_line('a')// &
        'integer, pointer :: r'//new_line('a')// &
        'integer, target :: local'//new_line('a')// &
        'local = 3'//new_line('a')// &
        'r => local'//new_line('a')// &
        'end function stale'//new_line('a')// &
        'end program main', 'does not outlive', &
        '/tmp/ffc_session_pointer_result_stale')) stop 1

    print *, 'PASS: scalar data pointer function result'
end program test_session_pointer_function_result
