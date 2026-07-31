program test_session_interface_external_call_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== external function call through an explicit interface ==='

    ! An explicit interface block declares the signature of a top-level
    ! (external) function while the real definition supplies the body. Both
    ! carry the same name, so registration sees the name twice; the signature
    ! must reconcile with the definition instead of rejecting it as a duplicate.
    ! The call then resolves across every scalar result kind.
    if (.not. expect_exit_status( &
        'integer function isq(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    isq = n * n'//new_line('a')// &
        'end function'//new_line('a')// &
        'real function rdbl(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    rdbl = real(n) * 2.0'//new_line('a')// &
        'end function'//new_line('a')// &
        'logical function pos(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    pos = n > 0'//new_line('a')// &
        'end function'//new_line('a')// &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    interface'//new_line('a')// &
        '        integer function isq(n)'//new_line('a')// &
        '            integer, intent(in) :: n'//new_line('a')// &
        '        end function'//new_line('a')// &
        '        real function rdbl(n)'//new_line('a')// &
        '            integer, intent(in) :: n'//new_line('a')// &
        '        end function'//new_line('a')// &
        '        logical function pos(n)'//new_line('a')// &
        '            integer, intent(in) :: n'//new_line('a')// &
        '        end function'//new_line('a')// &
        '    end interface'//new_line('a')// &
        '    if (isq(5) /= 25) error stop'//new_line('a')// &
        '    if (abs(rdbl(3) - 6.0) > 1e-6) error stop'//new_line('a')// &
        '    if (.not. pos(2)) error stop'//new_line('a')// &
        '    if (pos(-1)) error stop'//new_line('a')// &
        'end program', 0, &
        '/tmp/ffc_session_interface_external_call_test')) stop 1

    ! An INTERFACE block inside the specification part of a module procedure
    ! only declares signatures; it must not terminate the procedure. The
    ! declarations and executable statements that follow it stay part of the
    ! body, and the program unit after the module is still compiled.
    if (.not. expect_exit_status( &
        'module bench_mod'//new_line('a')// &
        'contains'//new_line('a')// &
        '    subroutine run(x)'//new_line('a')// &
        '        interface'//new_line('a')// &
        '            integer function f()'//new_line('a')// &
        '            end function f'//new_line('a')// &
        '        end interface'//new_line('a')// &
        '        integer :: x'//new_line('a')// &
        '        if (x /= 1) error stop'//new_line('a')// &
        '        print *, x'//new_line('a')// &
        '    end subroutine run'//new_line('a')// &
        '    integer function twice(y)'//new_line('a')// &
        '        interface'//new_line('a')// &
        '            subroutine sub(a)'//new_line('a')// &
        '                integer, intent(in) :: a'//new_line('a')// &
        '            end subroutine sub'//new_line('a')// &
        '        end interface'//new_line('a')// &
        '        integer, intent(in) :: y'//new_line('a')// &
        '        twice = 2 * y'//new_line('a')// &
        '    end function twice'//new_line('a')// &
        'end module bench_mod'//new_line('a')// &
        'program p'//new_line('a')// &
        '    use bench_mod'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    call run(1)'//new_line('a')// &
        '    if (twice(3) /= 6) error stop'//new_line('a')// &
        'end program p', 0, &
        '/tmp/ffc_session_interface_unused_body_test')) stop 1

    print *, 'PASS: external functions called through explicit interfaces'
end program test_session_interface_external_call_compiler
