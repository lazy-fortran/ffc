module ffc_suite_cases_00
    !! Dispatcher cases, folded from the standalone wrapper test
    !! programs of the same names. One link of libffc instead of
    !! one per test; the case name is the old test name.
    implicit none
    public :: case_test_session_empty_program_compiler
contains

    subroutine case_test_session_empty_program_compiler()
        use ffc_test_support, only: expect_exit_status
        implicit none

    print *, '=== direct session empty program compiler test ==='

    if (.not. expect_exit_status( 'program main'//new_line('a')// 'end program main', 0, '/tmp/ffc_session_empty_program_test')) stop 1

    print *, 'PASS: empty program compiles and runs through direct LIRIC session'
    end subroutine case_test_session_empty_program_compiler

end module ffc_suite_cases_00
