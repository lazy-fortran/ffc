program test_io_runtime_library
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing I/O Runtime Library Implementation ==="
    print *, ""

    all_passed = all_passed .and. test_runtime_library_linking()
    all_passed = all_passed .and. test_actual_io_operations()

    if (all_passed) then
        print *, ""
        print *, "All I/O runtime library tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some I/O runtime library tests failed!"
        stop 1
    end if

contains

    function test_runtime_library_linking() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: command
        character(len=256) :: result_file
        integer :: exit_code

        print *, "Testing I/O runtime library linking..."

        passed = .false.

        ! Create temporary directory for test
        call temp_mgr%create('io_runtime_link')

        ! Create test program that uses I/O operations
        result_file = temp_mgr%get_file_path('test_program.F90')
        open (unit=10, file=result_file, status='replace')
        write (10, '(A)') 'program test_io'
        write (10, '(A)') '    implicit none'
        write (10, '(A)') '    integer :: x = 42'
        write (10, '(A)') '    print *, "Value:", x'
        write (10, '(A)') 'end program test_io'
        close (10)

        ! Try to compile to executable (should fail due to missing runtime)
        command = fortran_with_isolated_cache('io_runtime_link')// &
                  ' --compile -o '//temp_mgr%get_file_path('test_exe')// &
                  ' '//result_file

        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            print *, "FAIL: Compilation succeeded but runtime functions not implemented"
            print *, "  Expected: Link failure due to undefined _FortranAio* symbols"
print *, "  Got: Successful compilation (indicates symbols may be incorrectly resolved)"

            ! Try to run the executable to see if it actually works
       call execute_command_line(temp_mgr%get_file_path('test_exe'), exitstat=exit_code)
            if (exit_code /= 0) then
       print *, "  Note: Executable fails at runtime, confirming missing implementation"
                passed = .false.  ! This demonstrates the limitation
            else
        print *, "  Note: Executable runs successfully - may be using different runtime"
                passed = .false.  ! Still not what we want
            end if
        else
            print *, "EXPECTED: Compilation failed due to missing runtime library"
            print *, "  This demonstrates the current limitation"
            passed = .false.  ! This is the failing test we expect
        end if

    end function test_runtime_library_linking

    function test_actual_io_operations() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: command
        character(len=256) :: source_file, exe_file
        integer :: exit_code

        print *, "Testing actual I/O operations with runtime library..."

        passed = .false.

        ! Create temporary directory for test
        call temp_mgr%create('io_runtime_ops')

        ! Create test program with various I/O operations
        source_file = temp_mgr%get_file_path('io_test.F90')
        open (unit=11, file=source_file, status='replace')
        write (11, '(A)') 'program io_operations'
        write (11, '(A)') '    implicit none'
        write (11, '(A)') '    integer :: a, b'
        write (11, '(A)') '    real :: x'
        write (11, '(A)') '    logical :: flag'
        write (11, '(A)') '    '
        write (11, '(A)') '    ! Test various print operations'
        write (11, '(A)') '    a = 123'
        write (11, '(A)') '    print *, "Integer:", a'
        write (11, '(A)') '    '
        write (11, '(A)') '    x = 3.14'
        write (11, '(A)') '    print *, "Real:", x'
        write (11, '(A)') '    '
        write (11, '(A)') '    flag = .true.'
        write (11, '(A)') '    print *, "Logical:", flag'
        write (11, '(A)') '    '
        write (11, '(A)') '    ! Test write statement'
        write (11, '(A)') '    write(*,*) "Write statement test"'
        write (11, '(A)') 'end program io_operations'
        close (11)

        exe_file = temp_mgr%get_file_path('io_exe')

        ! Try to compile the I/O test program
        command = fortran_with_isolated_cache('io_runtime_ops')// &
                  ' --compile -o '//exe_file//' '//source_file

        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            print *, "Compilation succeeded - checking runtime execution..."

            ! Try to run the executable
            call execute_command_line(exe_file, exitstat=exit_code)
            if (exit_code == 0) then
                print *, "UNEXPECTED: All I/O operations work correctly"
              print *, "  This suggests runtime library is already properly implemented"
                passed = .true.  ! This would mean the task is already done
            else
                print *, "EXPECTED: Runtime execution failed"
      print *, "  Confirms that _FortranAio* functions are declared but not implemented"
                passed = .false.  ! This is the failing test we expect
            end if
        else
            print *, "EXPECTED: Compilation failed due to missing runtime symbols"
            print *, "  This demonstrates the current limitation we need to fix"
            passed = .false.  ! This is the failing test we expect
        end if

    end function test_actual_io_operations

end program test_io_runtime_library
