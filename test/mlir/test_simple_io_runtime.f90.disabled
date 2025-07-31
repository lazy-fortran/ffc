program test_simple_io_runtime
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Simple I/O Runtime Library ==="
    print *, ""

    all_passed = all_passed .and. test_simple_literal_io()

    if (all_passed) then
        print *, ""
        print *, "Simple I/O runtime library test passed!"
        stop 0
    else
        print *, ""
        print *, "Simple I/O runtime library test failed!"
        stop 1
    end if

contains

    function test_simple_literal_io() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: command
        character(len=256) :: source_file, exe_file
        integer :: exit_code

        print *, "Testing simple literal I/O with runtime library..."

        passed = .false.

        ! Create temporary directory for test
        call temp_mgr%create('simple_io_runtime')

        ! Create simple test program with only literal print statements
        source_file = temp_mgr%get_file_path('simple_io.F90')
        open (unit=11, file=source_file, status='replace')
        write (11, '(A)') 'program simple_io'
        write (11, '(A)') '    implicit none'
        write (11, '(A)') '    print *, "Hello World"'
        write (11, '(A)') '    print *, 42'
        write (11, '(A)') 'end program simple_io'
        close (11)

        exe_file = temp_mgr%get_file_path('simple_exe')

        ! Try to compile the simple I/O test program
        command = fortran_with_isolated_cache('simple_io_runtime')// &
                  ' --compile -o '//exe_file//' '//source_file

        call execute_command_line(command, exitstat=exit_code)

        if (exit_code == 0) then
            print *, "Compilation succeeded - checking runtime execution..."

            ! Try to run the executable
            call execute_command_line(exe_file, exitstat=exit_code)
            if (exit_code == 0) then
                print *, "SUCCESS: Simple I/O operations work with runtime library"
                passed = .true.
            else
                print *, "FAIL: Runtime execution failed"
                passed = .false.
            end if
        else
            print *, "FAIL: Compilation failed"
            passed = .false.
        end if

    end function test_simple_literal_io

end program test_simple_io_runtime
