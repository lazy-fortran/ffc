program test_hlfir_generation_simple
    use temp_utils, only: temp_dir_manager, create_test_cache_dir
    use system_utils, only: sys_run_command, sys_file_exists
    implicit none
    
    type(temp_dir_manager) :: temp_mgr
    character(len=:), allocatable :: test_file, command, cmd_output, cache_dir
    character(len=512) :: output_file
    integer :: exit_code, unit
    logical :: all_passed
    
    all_passed = .true.
    print *, "=== Simple HLFIR Generation Tests ==="
    
    ! Test 1: Basic HLFIR generation
    call test_basic_hlfir()
    
    ! Test 2: Compilation test
    call test_compilation()
    
    if (.not. all_passed) then
        print *, "Some HLFIR tests FAILED!"
        stop 1
    else
        print *, "All HLFIR tests PASSED!"
    end if
    
contains
    
    subroutine test_basic_hlfir()
        character(len=1024) :: line
        integer :: ios
        
        print *, "TEST: Basic HLFIR generation"
        
        call temp_mgr%create('hlfir_basic_test')
        cache_dir = create_test_cache_dir('hlfir_basic')
        
        test_file = temp_mgr%get_file_path('basic.f90')
        call write_test_program(test_file, &
            "program basic" // new_line('a') // &
            "    integer :: x, y" // new_line('a') // &
            "    x = 10" // new_line('a') // &
            "    y = 20" // new_line('a') // &
            "    print *, x + y" // new_line('a') // &
            "    do i = 1, 3" // new_line('a') // &
            "        print *, i" // new_line('a') // &
            "    end do" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('hlfir_output.txt')
        ! Use absolute path since fpm changes directory
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // &
                  trim(test_file) // ' > ' // trim(output_file) // ' 2>&1'
        call sys_run_command(command, cmd_output, exit_code)
        
        print *, "  Debug: Output file is:", trim(output_file)
        print *, "  Debug: Exit code was:", exit_code
        
        ! Read output file
        open(newunit=unit, file=trim(output_file), status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print *, "  ERROR: Could not open output file"
            all_passed = .false.
            return
        end if
        cmd_output = ""
        do
            read(unit, '(A)', iostat=ios) line
            if (ios /= 0) exit
            cmd_output = cmd_output // trim(line) // new_line('a')
        end do
        close(unit)
        
        print *, "  Debug: First 200 chars of output:", cmd_output(1:min(200,len(cmd_output)))
        
        ! Check for HLFIR features
        if (index(cmd_output, 'hlfir.declare') > 0) then
            print *, "  PASS: hlfir.declare found"
        else
            print *, "  FAIL: Expected hlfir.declare"
            all_passed = .false.
        end if
        
        if (index(cmd_output, 'fir.do_loop') > 0) then
            print *, "  PASS: fir.do_loop found"
        else
            print *, "  FAIL: Expected fir.do_loop"
            all_passed = .false.
        end if
        
        if (index(cmd_output, 'hlfir.print') > 0) then
            print *, "  PASS: hlfir.print found"
        else
            print *, "  FAIL: Expected hlfir.print"
            all_passed = .false.
        end if
        
        if (index(cmd_output, '!fir.int<4>') > 0) then
            print *, "  PASS: !fir.int<4> types found"
        else
            print *, "  FAIL: Expected !fir.int<4> types"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_compilation()
        print *, "TEST: HLFIR compilation pipeline"
        
        call temp_mgr%create('hlfir_compile_test')
        cache_dir = create_test_cache_dir('hlfir_compile')
        
        test_file = temp_mgr%get_file_path('hello.f90')
        call write_test_program(test_file, &
            "program hello" // new_line('a') // &
            "    print *, 'Hello from HLFIR!'" // new_line('a') // &
            "end program")
        
        ! Test --compile flag
        command = 'cd ' // temp_mgr%path // ' && FORTRAN_CACHE_DIR=' // cache_dir // &
                  ' fpm run fortran -- --compile ' // trim(test_file) // ' 2>&1'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0) then
            print *, "  PASS: Compilation succeeded"
            
            ! Check if executable exists
            if (sys_file_exists(temp_mgr%get_file_path('hello'))) then
                print *, "  PASS: Executable generated"
                
                ! Try to run it
                command = 'cd ' // temp_mgr%path // ' && ./hello'
                call sys_run_command(command, cmd_output, exit_code)
                
                if (exit_code == 0 .and. index(cmd_output, 'Hello from HLFIR!') > 0) then
                    print *, "  PASS: Executable runs correctly"
                else
                    print *, "  FAIL: Executable did not produce expected output"
                    all_passed = .false.
                end if
            else
                print *, "  FAIL: Executable not found"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: Compilation failed"
            print *, "  Error output:", trim(cmd_output)
            all_passed = .false.
        end if
    end subroutine
    
    subroutine write_test_program(filename, content)
        character(len=*), intent(in) :: filename, content
        integer :: unit
        
        open(newunit=unit, file=filename, action='write', status='replace')
        write(unit, '(A)') content
        close(unit)
    end subroutine write_test_program
    
end program test_hlfir_generation_simple