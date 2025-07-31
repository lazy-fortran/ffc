program test_hlfir_generation
    use temp_utils, only: temp_dir_manager, create_test_cache_dir
    use system_utils, only: sys_run_command
    implicit none
    
    type(temp_dir_manager) :: temp_mgr
    character(len=:), allocatable :: test_file, command, cmd_output, cache_dir
    integer :: exit_code
    logical :: all_passed
    
    all_passed = .true.
    print *, "=== HLFIR Generation Tests ==="
    
    ! Test 1: Variable declarations use hlfir.declare
    call test_hlfir_declarations()
    
    ! Test 2: Loop constructs use fir.do_loop
    call test_fir_loops()
    
    ! Test 3: Print statements use hlfir.print
    call test_hlfir_print()
    
    ! Test 4: String literals use fir.string_lit
    call test_fir_string_literals()
    
    ! Test 5: Types use !fir.* notation
    call test_fir_types()
    
    ! Test 6: HLFIR to FIR lowering with flang-opt-19
    call test_hlfir_to_fir_lowering()
    
    ! Test 7: FIR to LLVM lowering with tco-19
    call test_fir_to_llvm_lowering()
    
    ! Test 8: Complete compilation and execution
    call test_hlfir_compile_and_execute()
    
    if (.not. all_passed) then
        print *, "Some HLFIR generation tests FAILED!"
        stop 1
    else
        print *, "All HLFIR generation tests PASSED!"
    end if
    
contains
    
    subroutine test_hlfir_declarations()
        print *, "TEST: Variable declarations use hlfir.declare"
        
        call temp_mgr%create('hlfir_decl_test')
        cache_dir = create_test_cache_dir('hlfir_decl')
        
        test_file = temp_mgr%get_file_path('decl_test.f90')
        call write_test_program(test_file, &
            "program decl_test" // new_line('a') // &
            "    integer :: x, y, z" // new_line('a') // &
            "    real :: a, b" // new_line('a') // &
            "    logical :: flag" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0 .and. index(cmd_output, 'hlfir.declare') > 0 .and. &
            index(cmd_output, '{var_name="x"}') > 0 .and. &
            index(cmd_output, '{var_name="y"}') > 0 .and. &
            index(cmd_output, '{var_name="z"}') > 0) then
            print *, "  PASS: hlfir.declare found for all variables"
        else
            print *, "  FAIL: Expected hlfir.declare for all variables"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_fir_loops()
        print *, "TEST: Loop constructs use fir.do_loop"
        
        call temp_mgr%create('fir_loop_test')
        cache_dir = create_test_cache_dir('fir_loop')
        
        test_file = temp_mgr%get_file_path('loop_test.f90')
        call write_test_program(test_file, &
            "program loop_test" // new_line('a') // &
            "    integer :: i, sum" // new_line('a') // &
            "    sum = 0" // new_line('a') // &
            "    do i = 1, 10" // new_line('a') // &
            "        sum = sum + i" // new_line('a') // &
            "    end do" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0 .and. index(cmd_output, 'fir.do_loop') > 0) then
            print *, "  PASS: fir.do_loop found"
            if (index(cmd_output, 'scf.for') == 0) then
                print *, "  PASS: No obsolete scf.for found"
            else
                print *, "  FAIL: Found obsolete scf.for"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: Expected fir.do_loop"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_hlfir_print()
        print *, "TEST: Print statements use hlfir.print"
        
        call temp_mgr%create('hlfir_print_test')
        cache_dir = create_test_cache_dir('hlfir_print')
        
        test_file = temp_mgr%get_file_path('print_test.f90')
        call write_test_program(test_file, &
            "program print_test" // new_line('a') // &
            '    print *, "Hello HLFIR!"' // new_line('a') // &
            "    print *, 42" // new_line('a') // &
            "    print *, 3.14" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0 .and. index(cmd_output, 'hlfir.print') > 0) then
            print *, "  PASS: hlfir.print operations found"
        else
            print *, "  FAIL: Expected hlfir.print operations"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_fir_string_literals()
        print *, "TEST: String literals use fir.string_lit"
        
        call temp_mgr%create('fir_string_test')
        cache_dir = create_test_cache_dir('fir_string')
        
        test_file = temp_mgr%get_file_path('string_test.f90')
        call write_test_program(test_file, &
            "program string_test" // new_line('a') // &
            '    print *, "Test string"' // new_line('a') // &
            '    print *, "Another string"' // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0 .and. index(cmd_output, 'fir.string_lit') > 0) then
            print *, "  PASS: fir.string_lit found for string literals"
        else
            print *, "  FAIL: Expected fir.string_lit for string literals"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_fir_types()
        print *, "TEST: Types use !fir.* notation"
        
        call temp_mgr%create('fir_types_test')
        cache_dir = create_test_cache_dir('fir_types')
        
        test_file = temp_mgr%get_file_path('types_test.f90')
        call write_test_program(test_file, &
            "program types_test" // new_line('a') // &
            "    integer :: i" // new_line('a') // &
            "    real :: r" // new_line('a') // &
            "    logical :: b" // new_line('a') // &
            "    character(len=10) :: s" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0) then
            if (index(cmd_output, '!fir.int<4>') > 0) then
                print *, "  PASS: !fir.int<4> found for integer"
            else
                print *, "  FAIL: Expected !fir.int<4> for integer"
                all_passed = .false.
            end if
            
            if (index(cmd_output, '!fir.real<4>') > 0) then
                print *, "  PASS: !fir.real<4> found for real"
            else
                print *, "  FAIL: Expected !fir.real<4> for real"
                all_passed = .false.
            end if
            
            if (index(cmd_output, '!fir.logical<1>') > 0) then
                print *, "  PASS: !fir.logical<1> found for logical"
            else
                print *, "  FAIL: Expected !fir.logical<1> for logical"
                all_passed = .false.
            end if
            
            if (index(cmd_output, '!fir.char<1,10>') > 0) then
                print *, "  PASS: !fir.char<1,10> found for character"
            else
                print *, "  FAIL: Expected !fir.char<1,10> for character"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: HLFIR generation failed"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_hlfir_to_fir_lowering()
        print *, "TEST: HLFIR to FIR lowering with flang-opt-19"
        
        call temp_mgr%create('hlfir_to_fir_test')
        cache_dir = create_test_cache_dir('hlfir_to_fir')
        
        test_file = temp_mgr%get_file_path('simple.f90')
        call write_test_program(test_file, &
            "program simple" // new_line('a') // &
            "    integer :: x" // new_line('a') // &
            "    x = 42" // new_line('a') // &
            "    print *, x" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-fir ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        ! Check for FIR-specific operations after lowering
        if (exit_code == 0) then
            print *, "  PASS: FIR lowering succeeded"
            ! After lowering, HLFIR operations should be lowered to FIR operations
            if (index(cmd_output, 'fir.') > 0) then
                print *, "  PASS: FIR operations found after lowering"
            else
                print *, "  FAIL: Expected FIR operations after lowering"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: FIR lowering failed"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_fir_to_llvm_lowering()
        print *, "TEST: FIR to LLVM lowering with tco-19"
        
        call temp_mgr%create('fir_to_llvm_test')
        cache_dir = create_test_cache_dir('fir_to_llvm')
        
        test_file = temp_mgr%get_file_path('simple.f90')
        call write_test_program(test_file, &
            "program simple" // new_line('a') // &
            "    integer :: x" // new_line('a') // &
            "    x = 42" // new_line('a') // &
            "    print *, x" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-llvm ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        ! Check for LLVM IR after lowering
        if (exit_code == 0) then
            print *, "  PASS: LLVM lowering succeeded"
            ! After lowering to LLVM, should see LLVM IR constructs
            if (index(cmd_output, 'define') > 0 .or. index(cmd_output, '@main') > 0) then
                print *, "  PASS: LLVM IR generated"
            else
                print *, "  FAIL: Expected LLVM IR"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: LLVM lowering failed"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_hlfir_compile_and_execute()
        use system_utils, only: sys_file_exists
        print *, "TEST: Complete compilation and execution through HLFIR pipeline"
        
        call temp_mgr%create('hlfir_compile_test')
        cache_dir = create_test_cache_dir('hlfir_compile')
        
        test_file = temp_mgr%get_file_path('hello.f90')
        call write_test_program(test_file, &
            "program hello" // new_line('a') // &
            "    integer :: sum, i" // new_line('a') // &
            "    sum = 0" // new_line('a') // &
            "    do i = 1, 10" // new_line('a') // &
            "        sum = sum + i" // new_line('a') // &
            "    end do" // new_line('a') // &
            "    print *, 'Sum is:', sum" // new_line('a') // &
            "end program")
        
        ! Test object file generation
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --compile -c ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0) then
            print *, "  PASS: Object file compilation succeeded"
            if (sys_file_exists('hello.o')) then
                print *, "  PASS: hello.o generated"
                call sys_run_command('rm -f hello.o', cmd_output, exit_code)
            else
                print *, "  FAIL: hello.o not found"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: Object file compilation failed"
            all_passed = .false.
        end if
        
        ! Test executable generation and execution
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --compile -o hello ' // test_file // &
                  ' 2>&1 | grep -v "DEBUG:" | grep -v "Project is up to date" | grep -v "STOP 0"'
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0) then
            print *, "  PASS: Executable compilation succeeded"
            if (sys_file_exists('hello')) then
                print *, "  PASS: hello executable generated"
                
                ! Run the executable
                call sys_run_command('./hello', cmd_output, exit_code)
                if (exit_code == 0 .and. index(cmd_output, 'Sum is:') > 0 .and. index(cmd_output, '55') > 0) then
                    print *, "  PASS: Executable runs and produces correct output"
                else
                    print *, "  FAIL: Executable did not produce expected output"
                    print *, "  Output was:", trim(cmd_output)
                    all_passed = .false.
                end if
                
                call sys_run_command('rm -f hello', cmd_output, exit_code)
            else
                print *, "  FAIL: hello executable not found"
                all_passed = .false.
            end if
        else
            print *, "  FAIL: Executable compilation failed"
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
    
end program test_hlfir_generation