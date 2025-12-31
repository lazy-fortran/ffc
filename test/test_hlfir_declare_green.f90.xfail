program test_hlfir_declare_green
    ! GREEN Test: Verify hlfir.declare generation works correctly
    implicit none
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: test_passed = .true.
    logical :: found_func = .false.
    logical :: found_x_alloca = .false., found_x_declare = .false.
    logical :: found_y_alloca = .false., found_y_declare = .false.
    logical :: found_arr_alloca = .false., found_arr_declare = .false.
    
    print *, "=== GREEN Test: hlfir.declare generation ==="
    print *, "Expected: This test MUST PASS"
    print *
    
    ! Create a simple Fortran program with variable declarations
    test_program = "test_declare.f90"
    open(newunit=unit, file=test_program, status='replace')
    write(unit, '(A)') 'program test_declare'
    write(unit, '(A)') '    integer :: x'
    write(unit, '(A)') '    real :: y'
    write(unit, '(A)') '    integer, dimension(10) :: arr'
    write(unit, '(A)') '    x = 42'
    write(unit, '(A)') '    y = 3.14'
    write(unit, '(A)') '    arr(1) = 100'
    write(unit, '(A)') 'end program test_declare'
    close(unit)
    
    ! Compile with HLFIR output
    output_file = "test_declare_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code /= 0) then
        print *, "FAIL: Compilation failed with exit code:", exit_code
        test_passed = .false.
    else
        ! Check the generated HLFIR
        open(newunit=unit, file=output_file, status='old', iostat=ios)
        if (ios /= 0) then
            print *, "FAIL: Could not open output file"
            test_passed = .false.
        else
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                
                ! Check for function
                if (index(line, 'func.func @_QQmain()') > 0) found_func = .true.
                
                ! Check for x variable
                if (index(line, 'fir.alloca') > 0 .and. index(line, 'name = "x"') > 0) then
                    found_x_alloca = .true.
                end if
                if (index(line, 'hlfir.declare') > 0 .and. index(line, 'name = "x"') > 0) then
                    found_x_declare = .true.
                end if
                
                ! Check for y variable
                if (index(line, 'fir.alloca') > 0 .and. index(line, 'name = "y"') > 0) then
                    found_y_alloca = .true.
                end if
                if (index(line, 'hlfir.declare') > 0 .and. index(line, 'name = "y"') > 0) then
                    found_y_declare = .true.
                end if
                
                ! Check for arr array
                if (index(line, 'fir.alloca') > 0 .and. index(line, 'name = "arr"') > 0) then
                    found_arr_alloca = .true.
                end if
                if (index(line, 'hlfir.declare') > 0 .and. index(line, 'name = "arr"') > 0) then
                    found_arr_declare = .true.
                end if
            end do
            close(unit)
            
            ! Verify all expected elements
            print *, "Checking generated HLFIR:"
            
            if (found_func) then
                print *, "  PASS: Found func.func @_QQmain()"
            else
                print *, "  FAIL: Missing func.func @_QQmain()"
                test_passed = .false.
            end if
            
            if (found_x_alloca .and. found_x_declare) then
                print *, "  PASS: Found fir.alloca and hlfir.declare for x"
            else
                print *, "  FAIL: Missing alloca/declare for x"
                test_passed = .false.
            end if
            
            if (found_y_alloca .and. found_y_declare) then
                print *, "  PASS: Found fir.alloca and hlfir.declare for y"
            else
                print *, "  FAIL: Missing alloca/declare for y"
                test_passed = .false.
            end if
            
            if (found_arr_alloca .and. found_arr_declare) then
                print *, "  PASS: Found fir.alloca and hlfir.declare for arr"
            else
                print *, "  FAIL: Missing alloca/declare for arr"
                test_passed = .false.
            end if
        end if
    end if
    
    print *
    if (test_passed) then
        print *, "SUCCESS: All GREEN tests passed!"
        print *, "hlfir.declare generation is working correctly"
        stop 0
    else
        print *, "FAILURE: Some tests failed"
        stop 1
    end if
    
end program test_hlfir_declare_green