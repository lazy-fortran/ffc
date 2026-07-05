program test_session_save_module_shadow_compiler
    use ffc_test_support, only: expect_exit_status
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

    print *, 'PASS: saved local shadows module variable and persists'
end program test_session_save_module_shadow_compiler
