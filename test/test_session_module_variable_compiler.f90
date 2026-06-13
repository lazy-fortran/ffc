program test_session_module_variable_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session module-variable compiler test ==='

    all_passed = .true.
    if (.not. test_module_variable_set_and_read()) all_passed = .false.
    if (.not. test_module_variable_incremented()) all_passed = .false.
    if (.not. test_module_variable_only_clause()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module-level integer variables persist across use'

contains

    logical function test_module_variable_set_and_read()
        ! B7a: a module variable written in one subroutine is readable
        ! in the program via use.
        character(len=*), parameter :: source = &
            'module counters'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            'end module counters'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use counters'//new_line('a')// &
            '  total = 42'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_module_variable_set_and_read = expect_exit_status( &
            source, 42, '/tmp/ffc_session_modvar_set_read')
    end function test_module_variable_set_and_read

    logical function test_module_variable_incremented()
        character(len=*), parameter :: source = &
            'module state'//new_line('a')// &
            '  integer :: count'//new_line('a')// &
            'end module state'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use state'//new_line('a')// &
            '  count = 3'//new_line('a')// &
            '  count = count + 4'//new_line('a')// &
            '  stop count'//new_line('a')// &
            'end program main'

        test_module_variable_incremented = expect_exit_status( &
            source, 7, '/tmp/ffc_session_modvar_incr')
    end function test_module_variable_incremented

    logical function test_module_variable_only_clause()
        character(len=*), parameter :: source = &
            'module vals'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  integer :: y'//new_line('a')// &
            'end module vals'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use vals, only: x'//new_line('a')// &
            '  x = 99'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_module_variable_only_clause = expect_exit_status( &
            source, 99, '/tmp/ffc_session_modvar_only')
    end function test_module_variable_only_clause

end program test_session_module_variable_compiler
