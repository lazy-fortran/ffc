program test_multi_argument_print
    use iso_fortran_env, only: error_unit
    use temp_utils, only: fortran_with_isolated_cache
    use system_utils, only: sys_remove_file
    implicit none

    logical :: all_tests_passed
    character(len=:), allocatable :: command, test_file, exe_file, output_file
    integer :: exit_code, unit, io_unit, stat
    character(len=1024) :: line

    print *, "=== Multi-Argument Print Statement Tests ==="
    print *

    all_tests_passed = .true.

    ! Test 1: Multiple integers
    if (.not. test_multiple_integers()) all_tests_passed = .false.

    ! Test 2: Mixed types (integer and variable)
    if (.not. test_mixed_types()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All multi-argument print tests passed!"
        stop 0
    else
        print *, "Some multi-argument print tests failed!"
        stop 1
    end if

contains

    function test_multiple_integers() result(passed)
        logical :: passed, found_10, found_20
        character(len=1024) :: temp_line

        passed = .false.

        ! Create test program
        test_file = "test_multi_int.f90"
        exe_file = "test_multi_int"
        output_file = "test_multi_int_output.txt"

        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x, y'
        write (unit, '(a)') '    x = 10'
        write (unit, '(a)') '    y = 20'
        write (unit, '(a)') '    print *, x, y'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Compile to executable
        command = fortran_with_isolated_cache('test_multi_int')// &
                  ' --compile -o '//exe_file//' '//test_file
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            ! Run and capture output
            command = './'//exe_file//' > '//output_file//' 2>&1'
            call execute_command_line(command, exitstat=exit_code)

            if (exit_code == 0) then
                ! Check output contains both 10 and 20
                open (newunit=io_unit, file=output_file, status='old', iostat=stat)
                if (stat == 0) then
                    ! Read all content to check for both values
                    found_10 = .false.
                    found_20 = .false.
                    do
                        read (io_unit, '(A)', iostat=stat) temp_line
                        if (stat /= 0) exit
                        if (index(temp_line, "10") > 0) found_10 = .true.
                        if (index(temp_line, "20") > 0) found_20 = .true.
                    end do
                    close (io_unit)

                    if (found_10 .and. found_20) then
                        print *, "PASS: Multiple integers print works"
                        passed = .true.
                    else
                        print *, "FAIL: Multiple integers output incorrect"
                        print *, "  Expected: both 10 and 20"
                        print *, "  Got: ", trim(line)
                    end if
                else
                    print *, "FAIL: Could not read multiple integers output file"
                end if
            else
                print *, "FAIL: Multiple integers executable failed to run"
            end if
        else
            print *, "FAIL: Multiple integers compilation failed"
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        call sys_remove_file(exe_file)
        call sys_remove_file(output_file)
    end function test_multiple_integers

    function test_mixed_types() result(passed)
        logical :: passed, found_42, found_99
        character(len=1024) :: temp_line

        passed = .false.

        ! Create test program
        test_file = "test_mixed_types.f90"
        exe_file = "test_mixed_types"
        output_file = "test_mixed_types_output.txt"

        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x'
        write (unit, '(a)') '    x = 42'
        write (unit, '(a)') '    print *, x, 99'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Compile to executable
        command = fortran_with_isolated_cache('test_mixed_types')// &
                  ' --compile -o '//exe_file//' '//test_file
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            ! Run and capture output
            command = './'//exe_file//' > '//output_file//' 2>&1'
            call execute_command_line(command, exitstat=exit_code)

            if (exit_code == 0) then
                ! Check output contains both 42 and 99
                open (newunit=io_unit, file=output_file, status='old', iostat=stat)
                if (stat == 0) then
                    ! Read all content to check for both values
                    found_42 = .false.
                    found_99 = .false.
                    do
                        read (io_unit, '(A)', iostat=stat) temp_line
                        if (stat /= 0) exit
                        if (index(temp_line, "42") > 0) found_42 = .true.
                        if (index(temp_line, "99") > 0) found_99 = .true.
                    end do
                    close (io_unit)

                    if (found_42 .and. found_99) then
                        print *, "PASS: Mixed types print works"
                        passed = .true.
                    else
                        print *, "FAIL: Mixed types output incorrect"
                        print *, "  Expected: both 42 and 99"
                        print *, "  Got: ", trim(line)
                    end if
                else
                    print *, "FAIL: Could not read mixed types output file"
                end if
            else
                print *, "FAIL: Mixed types executable failed to run"
            end if
        else
            print *, "FAIL: Mixed types compilation failed"
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        call sys_remove_file(exe_file)
        call sys_remove_file(output_file)
    end function test_mixed_types

end program test_multi_argument_print
