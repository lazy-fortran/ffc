program test_hlfir_assign_red
    ! RED Test: hlfir.assign with aliasing analysis and semantic validation
    ! This test MUST FAIL until we implement proper hlfir.assign
    implicit none
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: found_assign = .false.
    logical :: found_aliasing_check = .false.
    
    print *, "=== RED Test: hlfir.assign generation ==="
    print *, "Expected: This test MUST FAIL (no proper hlfir.assign yet)"
    print *
    
    ! Create Fortran program with assignments requiring semantic analysis
    test_program = "test_assign.f90"
    open(newunit=unit, file=test_program, status='replace')
    write(unit, '(A)') 'program test_assign'
    write(unit, '(A)') '    real, dimension(10) :: a, b, c'
    write(unit, '(A)') '    real :: x, y'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Simple scalar assignment'
    write(unit, '(A)') '    x = 42.0          ! Should generate hlfir.assign'
    write(unit, '(A)') '    y = x + 1.0       ! Should generate hlfir.assign'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Array assignment (whole array)'
    write(unit, '(A)') '    a = 0.0           ! Should generate hlfir.assign for array'
    write(unit, '(A)') '    b = a             ! Should check for aliasing'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Array section assignment with potential aliasing'
    write(unit, '(A)') '    a(1:5) = a(6:10) ! Should detect no aliasing'
    write(unit, '(A)') '    a(2:8) = a(3:9)  ! Should detect aliasing overlap!'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Assignment with type conversion'
    write(unit, '(A)') '    x = int(y)        ! Should handle type conversion in hlfir.assign'
    write(unit, '(A)') 'end program test_assign'
    close(unit)
    
    ! Try to compile with HLFIR output
    output_file = "test_assign_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code == 0) then
        ! Check for hlfir.assign in output
        open(newunit=unit, file=output_file, status='old', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                if (index(line, 'hlfir.assign') > 0) then
                    found_assign = .true.
                    ! Check if it includes aliasing info
                    if (index(line, 'alias') > 0 .or. index(line, 'overlap') > 0) then
                        found_aliasing_check = .true.
                    end if
                end if
            end do
            close(unit)
            
            if (found_assign .and. found_aliasing_check) then
                print *, "UNEXPECTED SUCCESS: Found hlfir.assign with aliasing!"
                print *, "This should not happen in RED phase"
                stop 1
            else if (found_assign) then
                print *, "PARTIAL: Found hlfir.assign but no aliasing analysis"
                print *, "Still failing as expected"
                stop 0
            else
                print *, "FAIL AS EXPECTED: No proper hlfir.assign found"
                stop 0
            end if
        else
            print *, "FAIL: Could not read output file"
            stop 1
        end if
    else
        print *, "EXPECTED FAILURE: Compilation failed"
        print *, "This is correct for RED phase"
        stop 0
    end if
    
end program test_hlfir_assign_red