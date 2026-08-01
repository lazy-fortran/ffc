program test_session_pointer_intent_in_target_arg
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    print *, '=== direct session TARGET actual for INTENT(IN) POINTER dummy ==='

    ! F2008 12.5.2.7: a POINTER dummy declared INTENT(IN) may be associated
    ! with a non-pointer actual that has the TARGET attribute. The callee
    ! reads through the dummy and reports the target's value.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        't = 42'//new_line('a')// &
        'call show(t)'//new_line('a')// &
        'contains'//new_line('a')// &
        'subroutine show(p)'//new_line('a')// &
        'integer, pointer, intent(in) :: p'//new_line('a')// &
        'stop p'//new_line('a')// &
        'end subroutine show'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_pointer_intent_in_target_arg')) stop 1

    ! Same rule for a derived-type TARGET actual (lfortran class_22.f90).
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'type :: inner_t'//new_line('a')// &
        'integer :: value'//new_line('a')// &
        'end type inner_t'//new_line('a')// &
        'type(inner_t), target :: x'//new_line('a')// &
        'x%value = 50'//new_line('a')// &
        'call expect_pointer(x)'//new_line('a')// &
        'contains'//new_line('a')// &
        'subroutine expect_pointer(p)'//new_line('a')// &
        'type(inner_t), pointer, intent(in) :: p'//new_line('a')// &
        'stop p%value'//new_line('a')// &
        'end subroutine expect_pointer'//new_line('a')// &
        'end program main', 50, &
        '/tmp/ffc_pointer_intent_in_target_derived')) stop 1

    ! Negative control: without the TARGET attribute the actual still cannot
    ! associate with a POINTER dummy.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer :: t'//new_line('a')// &
        't = 1'//new_line('a')// &
        'call show(t)'//new_line('a')// &
        'contains'//new_line('a')// &
        'subroutine show(p)'//new_line('a')// &
        'integer, pointer, intent(in) :: p'//new_line('a')// &
        'stop p'//new_line('a')// &
        'end subroutine show'//new_line('a')// &
        'end program main', 'must be a pointer', &
        '/tmp/ffc_pointer_intent_in_no_target')) stop 1

    ! Negative control: INTENT(OUT)/INTENT(INOUT) POINTER dummies still
    ! require a pointer actual, TARGET attribute notwithstanding.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer, target :: t'//new_line('a')// &
        't = 1'//new_line('a')// &
        'call reset(t)'//new_line('a')// &
        'contains'//new_line('a')// &
        'subroutine reset(p)'//new_line('a')// &
        'integer, pointer, intent(inout) :: p'//new_line('a')// &
        'nullify(p)'//new_line('a')// &
        'end subroutine reset'//new_line('a')// &
        'end program main', 'must be a pointer', &
        '/tmp/ffc_pointer_intent_inout_target')) stop 1

    print *, 'PASS: TARGET actual accepted for INTENT(IN) POINTER dummy'
end program test_session_pointer_intent_in_target_arg
