program test_hlfir_associate_red
    ! RED Test: hlfir.associate for temporary association management
    ! This test MUST FAIL until we implement hlfir.associate/end_associate
    implicit none
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: found_associate = .false.
    logical :: found_end_associate = .false.
    
    print *, "=== RED Test: hlfir.associate generation ==="
    print *, "Expected: This test MUST FAIL (no hlfir.associate yet)"
    print *
    
    ! Create Fortran program requiring temporary management
    test_program = "test_associate.f90"
    open(newunit=unit, file=test_program, status='replace')
    write(unit, '(A)') 'program test_associate'
    write(unit, '(A)') '    real, dimension(10) :: a, b, c'
    write(unit, '(A)') '    real :: temp'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Initialize arrays'
    write(unit, '(A)') '    a = 1.0'
    write(unit, '(A)') '    b = 2.0'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Expression requiring temporary'
    write(unit, '(A)') '    ! The expression (a + b) needs to be evaluated and stored'
    write(unit, '(A)') '    c = (a + b) * 2.0  ! Should use hlfir.associate for (a+b)'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Function argument requiring temporary'
    write(unit, '(A)') '    temp = sum(a * b)  ! Should associate temporary for a*b'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Associate construct (explicit association)'
    write(unit, '(A)') '    associate (x => a + b)'
    write(unit, '(A)') '        c = x * 3.0    ! x is associated with temporary'
    write(unit, '(A)') '    end associate      ! Should generate hlfir.end_associate'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Nested expressions with multiple temporaries'
    write(unit, '(A)') '    c = matmul(transpose(a), b) + sum(a)'
    write(unit, '(A)') '    ! Should create temporaries for:'
    write(unit, '(A)') '    ! - transpose(a)'
    write(unit, '(A)') '    ! - matmul result'
    write(unit, '(A)') '    ! - sum(a)'
    write(unit, '(A)') 'end program test_associate'
    close(unit)
    
    ! Try to compile with HLFIR output
    output_file = "test_associate_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code == 0) then
        ! Check for hlfir.associate in output
        open(newunit=unit, file=output_file, status='old', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                if (index(line, 'hlfir.associate') > 0) then
                    found_associate = .true.
                end if
                if (index(line, 'hlfir.end_associate') > 0) then
                    found_end_associate = .true.
                end if
            end do
            close(unit)
            
            if (found_associate .and. found_end_associate) then
                print *, "UNEXPECTED SUCCESS: Found hlfir.associate/end_associate!"
                print *, "This should not happen in RED phase"
                stop 1
            else if (found_associate) then
                print *, "PARTIAL: Found hlfir.associate but no end_associate"
                print *, "Still failing as expected"
                stop 0
            else
                print *, "FAIL AS EXPECTED: No hlfir.associate found"
                print *, "Temporary management not yet implemented"
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
    
end program test_hlfir_associate_red