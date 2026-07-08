program test_session_pointer_proc
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    print *, '=== direct session procedure pointer compiler test ==='

    if (.not. expect_error_contains( &
        'program main'//new_line('a')// &
        'interface'//new_line('a')// &
        '  subroutine c_target()'//new_line('a')// &
        '    !GCC$ ATTRIBUTES CDECL :: c_target'//new_line('a')// &
        '  end subroutine c_target'//new_line('a')// &
        '  subroutine std_target()'//new_line('a')// &
        '    !GCC$ ATTRIBUTES STDCALL :: std_target'//new_line('a')// &
        '  end subroutine std_target'//new_line('a')// &
        'end interface'//new_line('a')// &
        '!GCC$ ATTRIBUTES CDECL :: fp'//new_line('a')// &
        'procedure(), pointer :: fp'//new_line('a')// &
        'fp => std_target'//new_line('a')// &
        'end program main', &
        'calling convention', &
        '/tmp/ffc_proc_ptr_callconv_mismatch_test')) stop 1

    ! B3d: procedure pointer to a contained integer function; call through it
    ! and verify the result via stop code.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'procedure(), pointer :: fp'//new_line('a')// &
        'fp => double_it'//new_line('a')// &
        'stop fp(21)'//new_line('a')// &
        'contains'//new_line('a')// &
        'integer function double_it(x)'//new_line('a')// &
        'integer, intent(in) :: x'//new_line('a')// &
        'double_it = x * 2'//new_line('a')// &
        'end function double_it'//new_line('a')// &
        'end program main', 42, &
        '/tmp/ffc_proc_ptr_func_test')) stop 1

    ! B3d: procedure pointer to a contained subroutine; call mutates a variable.
    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        'implicit none'//new_line('a')// &
        'integer :: result'//new_line('a')// &
        'procedure(), pointer :: sp'//new_line('a')// &
        'result = 0'//new_line('a')// &
        'sp => add_ten'//new_line('a')// &
        'call sp(result)'//new_line('a')// &
        'stop result'//new_line('a')// &
        'contains'//new_line('a')// &
        'subroutine add_ten(x)'//new_line('a')// &
        'integer, intent(inout) :: x'//new_line('a')// &
        'x = x + 10'//new_line('a')// &
        'end subroutine add_ten'//new_line('a')// &
        'end program main', 10, &
        '/tmp/ffc_proc_ptr_sub_test')) stop 1

    print *, 'PASS: procedure pointer to function and subroutine'
end program test_session_pointer_proc
