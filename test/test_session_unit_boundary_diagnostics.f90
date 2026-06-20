program test_session_unit_boundary_diagnostics
    use ffc_test_support, only: expect_exit_status, &
                                expect_cli_no_error, &
                                expect_no_error
    implicit none

    logical :: all_passed

    print *, '=== direct session program unit boundary diagnostic test ==='

    all_passed = .true.
    if (.not. test_explicit_interface_call_runs()) all_passed = .false.
    if (.not. test_cli_explicit_interface_no_error()) all_passed = .false.
    if (.not. test_use_intrinsic_module_no_error()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: program unit boundary diagnostics are targeted'

contains

    logical function test_use_intrinsic_module_no_error()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  use iso_fortran_env'//new_line('a')// &
                                       'end program main'

        ! USE of intrinsic module should be silently ignored (no error)
        test_use_intrinsic_module_no_error = expect_no_error( &
            source, '/tmp/ffc_session_use_intrinsic_test')
    end function test_use_intrinsic_module_no_error

    logical function test_explicit_interface_call_runs()
        ! A plain explicit interface for an external subroutine lowers: the
        ! CALL type-checks and resolves to the separately-defined subroutine.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine bump(x)'//new_line('a')// &
            '      integer, intent(inout) :: x'//new_line('a')// &
            '    end subroutine bump'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  v = 3'//new_line('a')// &
            '  call bump(v)'//new_line('a')// &
            '  stop v'//new_line('a')// &
            'end program main'//new_line('a')// &
            'subroutine bump(x)'//new_line('a')// &
            '  integer, intent(inout) :: x'//new_line('a')// &
            '  x = x + 1'//new_line('a')// &
            'end subroutine bump'

        test_explicit_interface_call_runs = expect_exit_status( &
            source, 4, '/tmp/ffc_session_interface_block_test')
    end function test_explicit_interface_call_runs

    logical function test_cli_explicit_interface_no_error()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine bump(x)'//new_line('a')// &
            '      integer, intent(inout) :: x'//new_line('a')// &
            '    end subroutine bump'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  v = 3'//new_line('a')// &
            '  call bump(v)'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            'end program main'//new_line('a')// &
            'subroutine bump(x)'//new_line('a')// &
            '  integer, intent(inout) :: x'//new_line('a')// &
            '  x = x + 1'//new_line('a')// &
            'end subroutine bump'

        test_cli_explicit_interface_no_error = expect_cli_no_error( &
            source, '/tmp/ffc_cli_interface_block_test')
    end function test_cli_explicit_interface_no_error

end program test_session_unit_boundary_diagnostics
