program test_session_module_multi_name_var_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== multi-name module variable declaration ==='

    ! FortFront keeps a multi-name module declaration (integer :: i, j) as one
    ! node. ffc emits one global per declared name, host-associates each in a
    ! module procedure, and imports each into the using program; regression
    ! guard for the "stores only scalar single-name module variables" rejection.
    if (.not. expect_exit_status( &
        'module multi_name_mod'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer :: i, j'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine fill()'//new_line('a')// &
        '        i = 3'//new_line('a')// &
        '        j = 7'//new_line('a')// &
        '    end subroutine'//new_line('a')// &
        'end module'//new_line('a')// &
        'program p'//new_line('a')// &
        '    use multi_name_mod'//new_line('a')// &
        '    call fill()'//new_line('a')// &
        '    if (i /= 3) error stop'//new_line('a')// &
        '    if (j /= 7) error stop'//new_line('a')// &
        '    if (i + j /= 10) error stop'//new_line('a')// &
        'end program', 0, &
        '/tmp/ffc_session_module_multi_name_var_test')) stop 1

    print *, 'PASS: multi-name module variables emit and import per name'
end program test_session_module_multi_name_var_compiler
