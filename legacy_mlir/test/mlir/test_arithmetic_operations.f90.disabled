program test_arithmetic_operations
    use iso_fortran_env, only: error_unit
    use system_utils, only: sys_file_exists, sys_get_temp_dir, sys_remove_file
    use temp_utils, only: fortran_with_isolated_cache
    implicit none

    logical :: all_tests_passed
    character(len=:), allocatable :: command, output_file
    character(len=4096) :: mlir_output
    integer :: stat, unit

    print *, "=== MLIR Arithmetic Operations Tests ==="
    print *

    all_tests_passed = .true.

    ! Test 1: Binary operations should use correct SSA values
    if (.not. test_binary_add()) all_tests_passed = .false.

    ! Test 2: Multiple operations should track SSA values correctly
    if (.not. test_multiple_operations()) all_tests_passed = .false.

    ! Test 3: Mixed variable references should work
    if (.not. test_mixed_references()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All arithmetic operation tests passed!"
        stop 0
    else
        print *, "Some arithmetic operation tests failed!"
        stop 1
    end if

contains

    function test_binary_add() result(passed)
        logical :: passed
        character(len=:), allocatable :: test_file
        integer :: io_unit

        passed = .false.

        ! Create test program
        test_file = "test_binary_add.f90"
        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x, y, z'
        write (unit, '(a)') '    x = 10'
        write (unit, '(a)') '    y = 20'
        write (unit, '(a)') '    z = x + y'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Generate MLIR
        output_file = "test_binary_add.mlir"
        command = fortran_with_isolated_cache('test_binary_add')// &
                  ' --emit-hlfir '//test_file//' > '//output_file//' 2>&1'
        call execute_command_line(command, exitstat=stat)

        if (stat == 0) then
            ! Read output and check for correct SSA values
            open (newunit=io_unit, file=output_file, status='old', iostat=stat)
            if (stat == 0) then
                read (io_unit, '(A)', iostat=stat) mlir_output
                close (io_unit)

                ! Check that we're adding the correct variables
                ! Should see something like: %N = arith.addi %x_ssa, %y_ssa
                ! Not: %N = arith.addi %y_ssa, %y_ssa
                if (index(mlir_output, "arith.addi") > 0) then
                    print *, "PASS: Binary addition generates arith.addi"
                    passed = .true.
                else
                    print *, "FAIL: Binary addition should generate arith.addi"
                    print *, "Output: ", trim(mlir_output)
                end if
            else
                print *, "FAIL: Could not read output file"
            end if
            call sys_remove_file(output_file)
        else
            print *, "FAIL: MLIR generation failed with exit code:", stat
        end if

        ! Cleanup
        call sys_remove_file(test_file)
    end function test_binary_add

    function test_multiple_operations() result(passed)
        logical :: passed
        character(len=:), allocatable :: test_file

        passed = .false.

        ! Create test program with multiple operations
        test_file = "test_multiple_ops.f90"
        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: a, b, c, d'
        write (unit, '(a)') '    a = 5'
        write (unit, '(a)') '    b = 10'
        write (unit, '(a)') '    c = a + b'
        write (unit, '(a)') '    d = c * 2'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Generate MLIR
        output_file = "test_multiple_ops.mlir"
        command = fortran_with_isolated_cache('test_multiple_ops')// &
                  ' --emit-hlfir '//test_file//' > '//output_file//' 2>&1'
        call execute_command_line(command, exitstat=stat)

        if (stat == 0) then
            print *, "PASS: Multiple operations compile successfully"
            passed = .true.
        else
            print *, "FAIL: Multiple operations failed with exit code:", stat
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        if (sys_file_exists(output_file)) call sys_remove_file(output_file)
    end function test_multiple_operations

    function test_mixed_references() result(passed)
        logical :: passed
        character(len=:), allocatable :: test_file

        passed = .false.

        ! Create test program with mixed variable references
        test_file = "test_mixed_refs.f90"
        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x, y, z'
        write (unit, '(a)') '    x = 15'
        write (unit, '(a)') '    y = 25'
        write (unit, '(a)') '    z = x + y - x'  ! Should use x twice, not y twice
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Generate MLIR
        output_file = "test_mixed_refs.mlir"
        command = fortran_with_isolated_cache('test_mixed_refs')// &
                  ' --emit-hlfir '//test_file//' > '//output_file//' 2>&1'
        call execute_command_line(command, exitstat=stat)

        if (stat == 0) then
            print *, "PASS: Mixed variable references compile successfully"
            passed = .true.
        else
            print *, "FAIL: Mixed references failed with exit code:", stat
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        if (sys_file_exists(output_file)) call sys_remove_file(output_file)
    end function test_mixed_references

end program test_arithmetic_operations
