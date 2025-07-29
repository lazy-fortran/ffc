program test_hlfir_compile_command
    use temp_utils
    use system_utils, only: sys_run_command, sys_file_exists
    implicit none
    
    logical :: all_tests_passed
    
    print *, "=== HLFIR --compile Command Tests ==="
    print *, ""
    
    all_tests_passed = .true.
    
    ! Test all HLFIR --compile requirements
    if (.not. test_compile_object_file()) all_tests_passed = .false.
    if (.not. test_loop_constructs()) all_tests_passed = .false.
    if (.not. test_executable_generation()) all_tests_passed = .false.
    if (.not. test_optimization_flags()) all_tests_passed = .false.
    if (.not. test_error_handling()) all_tests_passed = .false.
    if (.not. test_variable_declarations()) all_tests_passed = .false.
    
    print *, ""
    if (all_tests_passed) then
        print *, "All HLFIR --compile tests PASSED!"
    else
        print *, "Some HLFIR --compile tests FAILED!"
        stop 1
    end if
    
contains

    function test_compile_object_file() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: test_file, cache_dir, command, output_file
        character(len=1024) :: cmd_output
        integer :: exit_code
        logical :: file_exists
        
        print *, "TEST: --compile generates .o file using HLFIR pipeline"
        
        call temp_mgr%create('hlfir_compile_test_obj')
        cache_dir = create_test_cache_dir('hlfir_compile_obj')
        
        test_file = temp_mgr%get_file_path('simple.f90')
        call write_test_program(test_file, &
            "program simple" // new_line('a') // &
            "    integer :: x" // new_line('a') // &
            "    x = 42" // new_line('a') // &
            "    print *, x" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('simple.o')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --compile -o ' // &
                  output_file // ' ' // test_file
        
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code /= 0) then
            print *, "  FAIL: Compilation failed: " // trim(cmd_output)
            passed = .false.
            return
        end if
        
        file_exists = sys_file_exists(output_file)
        if (.not. file_exists) then
            print *, "  FAIL: Object file not created: " // output_file
            passed = .false.
            return
        end if
        
        print *, "  PASS: Object file generated successfully"
        passed = .true.
    end function test_compile_object_file

    function test_loop_constructs() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: test_file, cache_dir, command
        character(len=1024) :: cmd_output
        integer :: exit_code
        
        print *, "TEST: Loop constructs use fir.do_loop instead of scf.for"
        
        call temp_mgr%create('hlfir_loop_test')
        cache_dir = create_test_cache_dir('hlfir_loop')
        
        test_file = temp_mgr%get_file_path('loop_test.f90')
        call write_test_program(test_file, &
            "program loop_test" // new_line('a') // &
            "    integer :: i, sum" // new_line('a') // &
            "    sum = 0" // new_line('a') // &
            "    do i = 1, 10" // new_line('a') // &
            "        sum = sum + i" // new_line('a') // &
            "    end do" // new_line('a') // &
            "    print *, sum" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code /= 0) then
            print *, "  FAIL: HLFIR generation failed: " // trim(cmd_output)
            passed = .false.
            return
        end if
        
        if (index(cmd_output, "fir.do_loop") == 0) then
            print *, "  FAIL: Expected fir.do_loop but found: " // cmd_output(1:min(200, len_trim(cmd_output)))
            passed = .false.
            return
        end if
        
        if (index(cmd_output, "scf.for") > 0) then
            print *, "  FAIL: Found obsolete scf.for - must use fir.do_loop"
            passed = .false.
            return
        end if
        
        print *, "  PASS: fir.do_loop generated correctly"
        passed = .true.
    end function test_loop_constructs

    function test_executable_generation() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: test_file, cache_dir, command, output_file
        character(len=1024) :: cmd_output
        integer :: exit_code
        logical :: file_exists
        
        print *, "TEST: --compile generates executable using HLFIR → FIR → LLVM pipeline"
        
        call temp_mgr%create('hlfir_exec_test')
        cache_dir = create_test_cache_dir('hlfir_exec')
        
        test_file = temp_mgr%get_file_path('hello.f90')
        call write_test_program(test_file, &
            "program hello" // new_line('a') // &
            "    print *, 'Hello HLFIR'" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('hello_exec')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --compile -o ' // &
                  output_file // ' ' // test_file
        
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code /= 0) then
            print *, "  FAIL: Executable compilation failed: " // trim(cmd_output)
            passed = .false.
            return
        end if
        
        file_exists = sys_file_exists(output_file)
        if (.not. file_exists) then
            print *, "  FAIL: Executable not created: " // output_file
            passed = .false.
            return
        end if
        
        ! Test that executable runs
        call sys_run_command(output_file, cmd_output, exit_code)
        if (exit_code /= 0) then
            print *, "  FAIL: Executable failed to run: " // trim(cmd_output)
            passed = .false.
            return
        end if
        
        print *, "  PASS: Executable created and runs successfully"
        passed = .true.
    end function test_executable_generation

    function test_optimization_flags() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: test_file, cache_dir, command, output_file
        character(len=1024) :: cmd_output
        integer :: exit_code
        logical :: file_exists
        
        print *, "TEST: Optimization flags use flang-opt for HLFIR optimizations"
        
        call temp_mgr%create('hlfir_opt_test')
        cache_dir = create_test_cache_dir('hlfir_opt')
        
        test_file = temp_mgr%get_file_path('optimize.f90')
        call write_test_program(test_file, &
            "program optimize" // new_line('a') // &
            "    integer :: x, y" // new_line('a') // &
            "    x = 10" // new_line('a') // &
            "    y = x * 2 + 5" // new_line('a') // &
            "    print *, y" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('optimize.o')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --compile -O2 -o ' // &
                  output_file // ' ' // test_file
        
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code /= 0) then
            print *, "  FAIL: Optimized compilation failed: " // trim(cmd_output)
            passed = .false.
            return
        end if
        
        file_exists = sys_file_exists(output_file)
        if (.not. file_exists) then
            print *, "  FAIL: Optimized object file not created"
            passed = .false.
            return
        end if
        
        print *, "  PASS: Optimized object file generated"
        passed = .true.
    end function test_optimization_flags

    function test_error_handling() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: test_file, cache_dir, command
        character(len=1024) :: cmd_output
        integer :: exit_code
        
        print *, "TEST: Error handling understands Fortran semantics in HLFIR context"
        
        call temp_mgr%create('hlfir_error_test')
        cache_dir = create_test_cache_dir('hlfir_error')
        
        test_file = temp_mgr%get_file_path('error_test.f90')
        call write_test_program(test_file, &
            "program error_test" // new_line('a') // &
            "    integer :: x" // new_line('a') // &
            "    x = undeclared_var" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --compile -o error_test.out ' // test_file
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code == 0) then
            print *, "  FAIL: Expected compilation error for undeclared variable"
            passed = .false.
            return
        end if
        
        if (index(cmd_output, "undeclared") > 0 .or. index(cmd_output, "not declared") > 0) then
            print *, "  PASS: Proper Fortran semantic error detected"
            passed = .true.
        else
            print *, "  FAIL: Generic error instead of Fortran semantic error: " // trim(cmd_output)
            passed = .false.
        end if
    end function test_error_handling

    function test_variable_declarations() result(passed)
        logical :: passed
        type(temp_dir_manager) :: temp_mgr
        character(len=:), allocatable :: test_file, cache_dir, command
        character(len=1024) :: cmd_output
        integer :: exit_code
        
        print *, "TEST: Variable declarations use hlfir.declare instead of memref.alloca"
        
        call temp_mgr%create('hlfir_var_test')
        cache_dir = create_test_cache_dir('hlfir_var')
        
        test_file = temp_mgr%get_file_path('variables.f90')
        call write_test_program(test_file, &
            "program variables" // new_line('a') // &
            "    integer :: i" // new_line('a') // &
            "    real :: r" // new_line('a') // &
            "    logical :: l" // new_line('a') // &
            "    i = 42" // new_line('a') // &
            "    r = 3.14" // new_line('a') // &
            "    l = .true." // new_line('a') // &
            "    print *, i, r, l" // new_line('a') // &
            "end program")
        
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // test_file
        call sys_run_command(command, cmd_output, exit_code)
        
        if (exit_code /= 0) then
            print *, "  FAIL: HLFIR generation failed: " // trim(cmd_output)
            passed = .false.
            return
        end if
        
        if (index(cmd_output, "hlfir.declare") == 0) then
            print *, "  FAIL: Expected hlfir.declare for variable declarations"
            passed = .false.
            return
        end if
        
        if (index(cmd_output, "memref.alloca") > 0) then
            print *, "  FAIL: Found obsolete memref.alloca - must use hlfir.declare"
            passed = .false.
            return
        end if
        
        if (index(cmd_output, "!fir.") == 0) then
            print *, "  FAIL: Expected !fir.* types but found none"
            passed = .false.
            return
        end if
        
        print *, "  PASS: hlfir.declare with !fir.* types generated correctly"
        passed = .true.
    end function test_variable_declarations

    subroutine write_test_program(filename, content)
        character(len=*), intent(in) :: filename, content
        integer :: unit
        
        open(newunit=unit, file=filename, action='write', status='replace')
        write(unit, '(A)') content
        close(unit)
    end subroutine write_test_program

end program test_hlfir_compile_command