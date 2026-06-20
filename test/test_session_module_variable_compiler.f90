program test_session_module_variable_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session module-variable compiler test ==='

    all_passed = .true.
    if (.not. test_module_variable_set_and_read()) all_passed = .false.
    if (.not. test_module_variable_incremented()) all_passed = .false.
    if (.not. test_module_variable_only_clause()) all_passed = .false.
    if (.not. test_module_procedure_reads_and_writes()) all_passed = .false.
    if (.not. test_logical_module_variable()) all_passed = .false.
    if (.not. test_protected_module_variable()) all_passed = .false.

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

    logical function test_module_procedure_reads_and_writes()
        ! #263: a scalar integer module variable with an initializer, written by
        ! a module procedure and again by the program, then printed. Exercises
        ! host association of the module variable inside the procedure body.
        character(len=*), parameter :: source = &
            'module counters'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: counter = 10'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump(n)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    counter = counter + n'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end module counters'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use counters'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call bump(5)'//new_line('a')// &
            '  counter = counter + 1'//new_line('a')// &
            '  print *, counter'//new_line('a')// &
            'end program main'

        test_module_procedure_reads_and_writes = expect_output( &
            source, '          16'//new_line('a'), &
            '/tmp/ffc_session_modvar_proc')
    end function test_module_procedure_reads_and_writes

    logical function test_logical_module_variable()
        ! A logical module variable keeps its initializer and is writable from a
        ! module procedure the program calls before printing it.
        character(len=*), parameter :: source = &
            'module sw'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  logical :: flag = .true.'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine off()'//new_line('a')// &
            '    flag = .false.'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end module sw'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use sw'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call off()'//new_line('a')// &
            '  print *, flag'//new_line('a')// &
            'end program main'

        test_logical_module_variable = expect_output( &
            source, ' F'//new_line('a'), &
            '/tmp/ffc_session_modvar_logical')
    end function test_logical_module_variable

    logical function test_protected_module_variable()
        ! A protected module variable is writable from a module procedure and
        ! readable from the using program (#274). PROTECTED only restricts
        ! external writes, which the frontend enforces; ffc emits storage.
        character(len=*), parameter :: source = &
            'module pmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, protected :: value = 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine set_value(v)'//new_line('a')// &
            '    integer, intent(in) :: v'//new_line('a')// &
            '    value = v'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end module pmod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use pmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call set_value(33)'//new_line('a')// &
            '  stop value'//new_line('a')// &
            'end program main'

        test_protected_module_variable = expect_exit_status( &
            source, 33, '/tmp/ffc_session_modvar_protected')
    end function test_protected_module_variable

end program test_session_module_variable_compiler
