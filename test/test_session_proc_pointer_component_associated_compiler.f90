program test_session_proc_pointer_component_associated
    ! #372: associated(obj%p) and associated(obj%p, target) for procedure
    ! pointer components. A fresh component reads as disassociated, a bound
    ! component reads as associated, and the two-argument form compares the
    ! stored callee address against the named procedure.
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    print *, '=== procedure pointer component associated compiler test ==='

    ! Null component: associated(obj%p) is false before any => binding.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'type :: handler_t'//new_line('a')// &
        '  procedure(), pointer :: p => null()'//new_line('a')// &
        'end type handler_t'//new_line('a')// &
        'type(handler_t) :: obj'//new_line('a')// &
        'integer :: code'//new_line('a')// &
        'code = 0'//new_line('a')// &
        'if (.not. associated(obj%p)) code = code + 7'//new_line('a')// &
        'stop code'//new_line('a')// &
        'end program main', 7, &
        '/tmp/ffc_proc_comp_assoc_null_test')) stop 1

    ! Bound component: one-argument and two-argument forms.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'type :: handler_t'//new_line('a')// &
        '  procedure(), pointer :: p => null()'//new_line('a')// &
        'end type handler_t'//new_line('a')// &
        'type(handler_t) :: obj'//new_line('a')// &
        'integer :: code'//new_line('a')// &
        'code = 0'//new_line('a')// &
        'obj%p => double_it'//new_line('a')// &
        'if (associated(obj%p)) code = code + 1'//new_line('a')// &
        'if (associated(obj%p, double_it)) code = code + 2'//new_line('a')// &
        'if (.not. associated(obj%p, triple_it)) code = code + 4'//new_line('a')// &
        'stop code'//new_line('a')// &
        'contains'//new_line('a')// &
        'integer function double_it(x)'//new_line('a')// &
        'integer, intent(in) :: x'//new_line('a')// &
        'double_it = x*2'//new_line('a')// &
        'end function double_it'//new_line('a')// &
        'integer function triple_it(x)'//new_line('a')// &
        'integer, intent(in) :: x'//new_line('a')// &
        'triple_it = x*3'//new_line('a')// &
        'end function triple_it'//new_line('a')// &
        'end program main', 7, &
        '/tmp/ffc_proc_comp_assoc_bound_test')) stop 1

    ! pr66465 shape: a scalar procedure pointer compared against a null
    ! procedure pointer component. Both are disassociated, so the result is
    ! false (F2018 16.9.20: a disassociated POINTER argument never matches).
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'type :: handler_t'//new_line('a')// &
        '  procedure(), pointer :: handler_out => null()'//new_line('a')// &
        'end type handler_t'//new_line('a')// &
        'type(handler_t) :: this'//new_line('a')// &
        'procedure(), pointer :: proc_ptr => null()'//new_line('a')// &
        'integer :: code'//new_line('a')// &
        'code = 0'//new_line('a')// &
        'if (.not. associated(proc_ptr, this%handler_out)) code = 11'// &
        new_line('a')// &
        'stop code'//new_line('a')// &
        'end program main', 11, &
        '/tmp/ffc_proc_comp_assoc_pr66465_test')) stop 1

    ! Two components bound to the same and to different procedures.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'type :: handler_t'//new_line('a')// &
        '  procedure(), pointer :: p => null()'//new_line('a')// &
        '  procedure(), pointer :: q => null()'//new_line('a')// &
        'end type handler_t'//new_line('a')// &
        'type(handler_t) :: obj'//new_line('a')// &
        'integer :: code'//new_line('a')// &
        'code = 0'//new_line('a')// &
        'obj%p => double_it'//new_line('a')// &
        'obj%q => double_it'//new_line('a')// &
        'if (associated(obj%p, obj%q)) code = code + 5'//new_line('a')// &
        'obj%q => null()'//new_line('a')// &
        'if (.not. associated(obj%q)) code = code + 6'//new_line('a')// &
        'if (associated(obj%p)) code = code + 8'//new_line('a')// &
        'stop code'//new_line('a')// &
        'contains'//new_line('a')// &
        'integer function double_it(x)'//new_line('a')// &
        'integer, intent(in) :: x'//new_line('a')// &
        'double_it = x*2'//new_line('a')// &
        'end function double_it'//new_line('a')// &
        'end program main', 19, &
        '/tmp/ffc_proc_comp_assoc_pair_test')) stop 1

    ! Negative: a data object has no procedure interface, so binding it to a
    ! procedure-pointer component is diagnosed rather than silently lowered.
    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'type :: handler_t'//new_line('a')// &
        '  procedure(), pointer :: p => null()'//new_line('a')// &
        'end type handler_t'//new_line('a')// &
        'type(handler_t) :: obj'//new_line('a')// &
        'integer :: n'//new_line('a')// &
        'n = 1'//new_line('a')// &
        'obj%p => n'//new_line('a')// &
        'stop 0'//new_line('a')// &
        'end program main', &
        'procedure pointer target is not a procedure', &
        '/tmp/ffc_proc_comp_assoc_bad_target_test')) stop 1

    print *, 'PASS: procedure pointer component associated'
end program test_session_proc_pointer_component_associated
