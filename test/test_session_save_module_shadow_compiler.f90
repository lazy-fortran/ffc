program test_session_save_module_shadow_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status, &
        expect_output
    implicit none

    print *, '=== saved local shadowing a module variable ==='

    ! A contained procedure's saved local shares its name with a host-associated
    ! module variable. The local shadows the module global with its own
    ! persistent storage; regression guard for the "duplicate declaration of
    ! saved local" error.
    if (.not. expect_exit_status( &
        'module save_shadow_mod'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    real :: y = 100.0'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine bump(expected)'//new_line('a')// &
        '        real, intent(in) :: expected'//new_line('a')// &
        '        real, save :: y = 5.0'//new_line('a')// &
        '        y = y + 1.0'//new_line('a')// &
        '        if (abs(y - expected) > 0.001) error stop'//new_line('a')// &
        '    end subroutine'//new_line('a')// &
        'end module'//new_line('a')// &
        'program p'//new_line('a')// &
        '    use save_shadow_mod'//new_line('a')// &
        '    call bump(6.0)'//new_line('a')// &
        '    call bump(7.0)'//new_line('a')// &
        'end program', 0, &
        '/tmp/ffc_session_save_module_shadow_test')) stop 1

    ! The first contained procedure reads the program's host variable. The
    ! sibling declares a distinct local with the same spelling; FortFront's
    ! binding identities must keep the two references separate.
    if (.not. expect_output( &
        'program p'//new_line('a')// &
        '    integer :: x'//new_line('a')// &
        '    x = 3'//new_line('a')// &
        '    call read_host()'//new_line('a')// &
        '    call shadow_local()'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine read_host()'//new_line('a')// &
        '        print *, x'//new_line('a')// &
        '    end subroutine read_host'//new_line('a')// &
        '    subroutine shadow_local()'//new_line('a')// &
        '        integer :: x'//new_line('a')// &
        '        x = 7'//new_line('a')// &
        '        print *, x'//new_line('a')// &
        '    end subroutine shadow_local'//new_line('a')// &
        'end program p', &
        '           3'//new_line('a')//'           7'//new_line('a'), &
        '/tmp/ffc_session_host_assoc_shadow')) stop 1

    ! An internal procedure may not synthesize a symbol for an unresolved
    ! reference: retain the frontend's undeclared-name diagnostic.
    if (.not. expect_error_contains( &
        'program p'//new_line('a')// &
        '    call bad()'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine bad()'//new_line('a')// &
        '        print *, absent_name'//new_line('a')// &
        '    end subroutine bad'//new_line('a')// &
        'end program p', 'not declared', &
        '/tmp/ffc_session_host_assoc_missing')) stop 1

    print *, 'PASS: saved local and host association bindings are distinct'
end program test_session_save_module_shadow_compiler
