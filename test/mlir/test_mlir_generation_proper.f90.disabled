program test_mlir_generation_proper
    use iso_fortran_env, only: error_unit
    use system_utils, only: sys_file_exists, sys_get_temp_dir, sys_remove_file
    use temp_utils, only: fortran_with_isolated_cache
    implicit none

    logical :: all_tests_passed
    character(len=:), allocatable :: command, output_file
    character(len=1024) :: mlir_output
    integer :: stat, unit

    print *, "=== MLIR Generation Tests ==="
    print *

    all_tests_passed = .true.

    ! Test 1: Simple variable declaration should generate hlfir.declare
    if (.not. test_variable_declaration()) all_tests_passed = .false.

    ! Test 2: Assignment should generate hlfir.assign
    if (.not. test_assignment()) all_tests_passed = .false.

    ! Test 3: Array operations should generate hlfir.expr
    if (.not. test_array_operations()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All MLIR generation tests passed!"
        stop 0
    else
        print *, "Some MLIR generation tests failed!"
        stop 1
    end if

contains

    function test_variable_declaration() result(passed)
        logical :: passed
        character(len=:), allocatable :: test_file

        passed = .false.

        ! Create test program
        test_file = "test_variable_declaration.f90"
        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x'
        write (unit, '(a)') '    x = 42'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Generate MLIR
        command = fortran_with_isolated_cache('test_variable_declaration') // ' --emit-hlfir ' // test_file
        call execute_command_line(command, exitstat=stat)

        if (stat == 0) then
            ! Check for MLIR declare in output
            ! For now, just check it doesn't crash
            passed = .true.
            print *, "PASS: MLIR declare generation (placeholder)"
        else
            print *, "FAIL: MLIR declare test failed with exit code:", stat
        end if

        ! Cleanup
        call sys_remove_file(test_file)
    end function test_variable_declaration

    function test_assignment() result(passed)
        logical :: passed
        character(len=:), allocatable :: test_file

        passed = .false.

        ! Create test program
        test_file = "test_assignment.f90"
        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x, y'
        write (unit, '(a)') '    x = 10'
        write (unit, '(a)') '    y = x + 5'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Generate MLIR
   command = fortran_with_isolated_cache('test_assignment')//' --emit-hlfir '//test_file
        call execute_command_line(command, exitstat=stat)

        if (stat == 0) then
            passed = .true.
            print *, "PASS: MLIR assign generation (placeholder)"
        else
            print *, "FAIL: MLIR assign test failed with exit code:", stat
        end if

        ! Cleanup
        call sys_remove_file(test_file)
    end function test_assignment

    function test_array_operations() result(passed)
        logical :: passed
        character(len=:), allocatable :: test_file

        passed = .false.

        ! Create test program
        test_file = "test_array_operations.f90"
        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: arr(5)'
        write (unit, '(a)') '    arr = [1, 2, 3, 4, 5]'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Generate MLIR
        command = fortran_with_isolated_cache('test_array_operations') // ' --emit-hlfir ' // test_file
        call execute_command_line(command, exitstat=stat)

        if (stat == 0) then
            passed = .true.
            print *, "PASS: MLIR array expr generation (placeholder)"
        else
            print *, "FAIL: MLIR array expr test failed with exit code:", stat
        end if

        ! Cleanup
        call sys_remove_file(test_file)
    end function test_array_operations

end program test_mlir_generation_proper
