program test_session_scalar_pointer_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    print *, '=== direct session scalar data-pointer compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: i'//new_line('a')// &
        'integer, pointer :: ip'//new_line('a')// &
        'real, target :: r'//new_line('a')// &
        'real, pointer :: rp'//new_line('a')// &
        'logical, target :: flag'//new_line('a')// &
        'logical, pointer :: fp'//new_line('a')// &
        'i = 4'//new_line('a')// &
        'r = 5.5'//new_line('a')// &
        'flag = .true.'//new_line('a')// &
        'ip => i'//new_line('a')// &
        'rp => r'//new_line('a')// &
        'fp => flag'//new_line('a')// &
        'ip = 8'//new_line('a')// &
        'r = 6.5'//new_line('a')// &
        'if (.not. fp) stop 1'//new_line('a')// &
        'if (.not. associated(ip, i)) stop 2'//new_line('a')// &
        'if (.not. associated(rp, r)) stop 3'//new_line('a')// &
        'if (.not. associated(fp, flag)) stop 4'//new_line('a')// &
        'nullify(ip, rp, fp)'//new_line('a')// &
        'if (associated(ip) .or. associated(rp) .or. associated(fp)) stop 5'// &
        new_line('a')// &
        'if (.not. flag) stop 6'//new_line('a')// &
        'stop i + int(r) + 1'//new_line('a')// &
        'end program main', 15, &
        '/tmp/ffc_session_scalar_pointer_numeric')) stop 1

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'integer, target :: value'//new_line('a')// &
        'value = 3'//new_line('a')// &
        'call bump(value)'//new_line('a')// &
        'stop value'//new_line('a')// &
        'contains'//new_line('a')// &
        'subroutine bump(target_value)'//new_line('a')// &
        'integer, target, intent(inout) :: target_value'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'p => target_value'//new_line('a')// &
        'p = 9'//new_line('a')// &
        'end subroutine bump'//new_line('a')// &
        'end program main', 9, &
        '/tmp/ffc_session_scalar_pointer_dummy')) stop 2

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'character(len=3), target :: text'//new_line('a')// &
        'character(len=3), pointer :: ptext'//new_line('a')// &
        "text = 'abc'"//new_line('a')// &
        'ptext => text'//new_line('a')// &
        "ptext = 'xy'"//new_line('a')// &
        "if (text /= 'xy ') stop 1"//new_line('a')// &
        'if (.not. associated(ptext, text)) stop 2'//new_line('a')// &
        'nullify(ptext)'//new_line('a')// &
        'if (associated(ptext)) stop 3'//new_line('a')// &
        'stop len_trim(text)'//new_line('a')// &
        'end program main', 2, &
        '/tmp/ffc_session_scalar_pointer_character')) stop 3

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'type :: pair_t'//new_line('a')// &
        '  integer :: left'//new_line('a')// &
        '  integer :: right'//new_line('a')// &
        'end type pair_t'//new_line('a')// &
        'type(pair_t), target :: value'//new_line('a')// &
        'type(pair_t), pointer :: pvalue'//new_line('a')// &
        'type(pair_t) :: copy'//new_line('a')// &
        'value = pair_t(2, 4)'//new_line('a')// &
        'pvalue => value'//new_line('a')// &
        'pvalue%left = 7'//new_line('a')// &
        'pvalue%right = 3'//new_line('a')// &
        'if (.not. associated(pvalue, value)) stop 1'//new_line('a')// &
        'copy = pvalue'//new_line('a')// &
        'nullify(pvalue)'//new_line('a')// &
        'if (associated(pvalue)) stop 2'//new_line('a')// &
        'stop copy%left + copy%right'//new_line('a')// &
        'end program main', 10, &
        '/tmp/ffc_session_scalar_pointer_derived')) stop 4

    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer :: value'//new_line('a')// &
        'integer, pointer :: p'//new_line('a')// &
        'p => value'//new_line('a')// &
        'end program main', 'right of => is not a target or pointer', &
        '/tmp/ffc_session_scalar_pointer_non_target')) stop 5

    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'integer, target :: value'//new_line('a')// &
        'real, pointer :: p'//new_line('a')// &
        'p => value'//new_line('a')// &
        'end program main', 'pointer and target scalar kinds do not match', &
        '/tmp/ffc_session_scalar_pointer_incompatible')) stop 6

    print *, 'PASS: scalar data-pointer storage, aliasing, queries, and diagnostics'
end program test_session_scalar_pointer_compiler
