program test_session_unit_boundary_diagnostics
    use ffc_test_support, only: expect_error_contains, &
                                expect_cli_error_contains, &
                                expect_no_error
    implicit none

    logical :: all_passed

    print *, '=== direct session program unit boundary diagnostic test ==='

    all_passed = .true.
    if (.not. test_interface_block_diagnostic()) all_passed = .false.
    if (.not. test_cli_interface_block_diagnostic()) all_passed = .false.
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

    logical function test_interface_block_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  interface'//new_line('a')// &
                                       '  end interface'//new_line('a')// &
                                       'end program main'

        test_interface_block_diagnostic = expect_error_contains( &
                                          source, &
                                          'unsupported interface block', &
                                          '/tmp/ffc_session_interface_block_test')
    end function test_interface_block_diagnostic

    logical function test_cli_interface_block_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  interface'//new_line('a')// &
                                       '  end interface'//new_line('a')// &
                                       'end program main'

        test_cli_interface_block_diagnostic = expect_cli_error_contains( &
                                              source, &
                                              'unsupported interface block', &
                                              '/tmp/ffc_cli_interface_block_test')
    end function test_cli_interface_block_diagnostic

end program test_session_unit_boundary_diagnostics
