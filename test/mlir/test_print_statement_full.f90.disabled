program test_print_statement_full
    use iso_fortran_env, only: error_unit
    use system_utils, only: sys_file_exists, sys_remove_file
    use temp_utils, only: fortran_with_isolated_cache
    implicit none

    logical :: all_tests_passed
    character(len=:), allocatable :: command, test_file, exe_file, output_file
    integer :: stat, unit, io_unit, exit_code
    character(len=1024) :: line

    print *, "=== Full Print Statement Tests ==="
    print *

    all_tests_passed = .true.

    ! Test 1: Integer print should work
    if (.not. test_integer_print()) all_tests_passed = .false.

    ! Test 2: Expression print should work
    if (.not. test_expression_print()) all_tests_passed = .false.

    ! Test 3: Mixed string and integer print should work
    if (.not. test_mixed_print()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All full print statement tests passed!"
        stop 0
    else
        print *, "Some full print statement tests failed!"
        stop 1
    end if

contains

    function test_integer_print() result(passed)
        logical :: passed

        passed = .false.

        ! Create test program
        test_file = "test_int_print.f90"
        exe_file = "test_int_print"
        output_file = "test_int_print_output.txt"

        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x'
        write (unit, '(a)') '    x = 42'
        write (unit, '(a)') '    print *, x'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Compile to executable
        command = fortran_with_isolated_cache('test_int_print')// &
                  ' --compile -o '//exe_file//' '//test_file
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            ! Run and capture output
            command = './'//exe_file//' > '//output_file//' 2>&1'
            call execute_command_line(command, exitstat=exit_code)

            if (exit_code == 0) then
                ! Check output contains 42
                open (newunit=io_unit, file=output_file, status='old', iostat=stat)
                if (stat == 0) then
                    read (io_unit, '(A)', iostat=stat) line
                    close (io_unit)

                    if (stat == 0 .and. index(line, "42") > 0) then
                        print *, "PASS: Integer print works"
                        passed = .true.
                    else
                        print *, "FAIL: Integer print output incorrect"
                        print *, "  Expected: 42"
                        print *, "  Got: ", trim(line)
                    end if
                else
                    print *, "FAIL: Could not read output file"
                end if
            else
                print *, "FAIL: Executable failed to run"
            end if
        else
            print *, "FAIL: Integer print compilation failed"
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        call sys_remove_file(exe_file)
        call sys_remove_file(output_file)
    end function test_integer_print

    function test_expression_print() result(passed)
        logical :: passed

        passed = .false.

        ! Create test program
        test_file = "test_expr_print.f90"
        exe_file = "test_expr_print"
        output_file = "test_expr_print_output.txt"

        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: x, y'
        write (unit, '(a)') '    x = 10'
        write (unit, '(a)') '    y = 20'
        write (unit, '(a)') '    print *, x + y'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Compile to executable
        command = fortran_with_isolated_cache('test_expr_print')// &
                  ' --compile -o '//exe_file//' '//test_file
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            ! Run and capture output
            command = './'//exe_file//' > '//output_file//' 2>&1'
            call execute_command_line(command, exitstat=exit_code)

            if (exit_code == 0) then
                ! Check output contains 30
                open (newunit=io_unit, file=output_file, status='old', iostat=stat)
                if (stat == 0) then
                    read (io_unit, '(A)', iostat=stat) line
                    close (io_unit)

                    if (stat == 0 .and. index(line, "30") > 0) then
                        print *, "PASS: Expression print works"
                        passed = .true.
                    else
                        print *, "FAIL: Expression print output incorrect"
                        print *, "  Expected: 30"
                        print *, "  Got: ", trim(line)
                    end if
                else
                    print *, "FAIL: Could not read expression output file"
                end if
            else
                print *, "FAIL: Expression executable failed to run"
            end if
        else
            print *, "FAIL: Expression print compilation failed"
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        call sys_remove_file(exe_file)
        call sys_remove_file(output_file)
    end function test_expression_print

    function test_mixed_print() result(passed)
        logical :: passed

        passed = .false.

        ! Create test program
        test_file = "test_mixed_print.f90"
        exe_file = "test_mixed_print"
        output_file = "test_mixed_print_output.txt"

        open (newunit=unit, file=test_file, status='replace')
        write (unit, '(a)') 'program test'
        write (unit, '(a)') '    integer :: result'
        write (unit, '(a)') '    result = 50'
        write (unit, '(a)') '    print *, "Result:", result'
        write (unit, '(a)') 'end program test'
        close (unit)

        ! Compile to executable
        command = fortran_with_isolated_cache('test_mixed_print')// &
                  ' --compile -o '//exe_file//' '//test_file
        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            ! Run and capture output
            command = './'//exe_file//' > '//output_file//' 2>&1'
            call execute_command_line(command, exitstat=exit_code)

            if (exit_code == 0) then
                ! Check output contains both "Result:" and "50"
                open (newunit=io_unit, file=output_file, status='old', iostat=stat)
                if (stat == 0) then
                    read (io_unit, '(A)', iostat=stat) line
                    close (io_unit)

        if (stat == 0 .and. index(line, "Result:") > 0 .and. index(line, "50") > 0) then
                        print *, "PASS: Mixed print works"
                        passed = .true.
                    else
                        print *, "FAIL: Mixed print output incorrect"
                        print *, "  Expected: Result: 50"
                        print *, "  Got: ", trim(line)
                    end if
                else
                    print *, "FAIL: Could not read mixed output file"
                end if
            else
                print *, "FAIL: Mixed executable failed to run"
            end if
        else
            print *, "FAIL: Mixed print compilation failed"
        end if

        ! Cleanup
        call sys_remove_file(test_file)
        call sys_remove_file(exe_file)
        call sys_remove_file(output_file)
    end function test_mixed_print

end program test_print_statement_full
