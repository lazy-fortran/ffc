program test_session_external_statement_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== bare EXTERNAL statement ==='

    ! An EXTERNAL statement names a procedure, not a variable, and carries the
    ! placeholder type "external". ffc skips it instead of rejecting it as an
    ! unsupported scalar type; the call resolves to the top-level subroutine.
    if (.not. expect_exit_status( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        '    external :: bar'//new_line('a')// &
        '    n = 41'//new_line('a')// &
        '    call bar(n)'//new_line('a')// &
        '    if (n /= 42) error stop'//new_line('a')// &
        'end program'//new_line('a')// &
        'subroutine bar(n)'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer, intent(inout) :: n'//new_line('a')// &
        '    n = n + 1'//new_line('a')// &
        'end subroutine', 0, &
        '/tmp/ffc_session_external_statement_test')) stop 1

    print *, 'PASS: bare EXTERNAL statement lowers and call resolves'
end program test_session_external_statement_compiler
