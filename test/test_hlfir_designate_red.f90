program test_hlfir_designate_red
    ! RED Test: hlfir.designate for array sections, components, substrings
    ! This test MUST FAIL until we implement hlfir.designate
    implicit none
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: found_designate = .false.
    
    print *, "=== RED Test: hlfir.designate generation ==="
    print *, "Expected: This test MUST FAIL (no hlfir.designate yet)"
    print *
    
    ! Create Fortran program with array sections and substring operations
    test_program = "test_designate.f90"
    open(newunit=unit, file=test_program, status='replace')
    write(unit, '(A)') 'program test_designate'
    write(unit, '(A)') '    integer, dimension(10) :: arr'
    write(unit, '(A)') '    integer :: x'
    write(unit, '(A)') '    character(len=20) :: str'
    write(unit, '(A)') '    character(len=5) :: substr'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Array element access'
    write(unit, '(A)') '    arr = 0'
    write(unit, '(A)') '    x = arr(5)           ! Should generate hlfir.designate'
    write(unit, '(A)') '    arr(3) = 42          ! Should generate hlfir.designate'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Array section'
    write(unit, '(A)') '    arr(2:8) = 100       ! Should generate hlfir.designate for section'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! String substring'
    write(unit, '(A)') '    str = "Hello, World!"'
    write(unit, '(A)') '    substr = str(1:5)    ! Should generate hlfir.designate for substring'
    write(unit, '(A)') 'end program test_designate'
    close(unit)
    
    ! Try to compile with HLFIR output
    output_file = "test_designate_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code == 0) then
        ! Check for hlfir.designate in output
        open(newunit=unit, file=output_file, status='old', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                if (index(line, 'hlfir.designate') > 0) then
                    found_designate = .true.
                    exit
                end if
            end do
            close(unit)
            
            if (found_designate) then
                print *, "UNEXPECTED SUCCESS: Found hlfir.designate in output!"
                print *, "This should not happen in RED phase"
                stop 1
            else
                print *, "FAIL AS EXPECTED: No hlfir.designate found"
                print *, "Currently only generates hlfir.declare"
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
    
end program test_hlfir_designate_red