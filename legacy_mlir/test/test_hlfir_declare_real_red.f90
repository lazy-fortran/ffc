program test_hlfir_declare_real_red
    ! RED Test: Actually compile Fortran with hlfir.declare
    ! This test MUST FAIL until we implement real functionality
    use iso_fortran_env, only: error_unit
    implicit none
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: found_hlfir_declare = .false.
    
    print *, "=== RED Test: Real hlfir.declare compilation ==="
    print *, "Expected: This test MUST FAIL (no real HLFIR generation yet)"
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
    
    ! Try to compile with HLFIR output
    output_file = "test_declare_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code == 0) then
        ! If compilation succeeded, check for hlfir.declare
        open(newunit=unit, file=output_file, status='old', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                if (index(line, 'hlfir.declare') > 0) then
                    found_hlfir_declare = .true.
                    exit
                end if
            end do
            close(unit)
            
            if (found_hlfir_declare) then
                print *, "UNEXPECTED SUCCESS: Found hlfir.declare in output!"
                print *, "This should not happen in RED phase"
                stop 1
            else
                print *, "FAIL: No hlfir.declare found (but compilation succeeded?)"
                stop 1
            end if
        else
            print *, "FAIL: Could not read output file"
            stop 1
        end if
    else
        print *, "EXPECTED FAILURE: Compilation failed (exit code:", exit_code, ")"
        print *, "This is correct for RED phase - no real HLFIR generation yet"
        stop 0
    end if
    
end program test_hlfir_declare_real_red