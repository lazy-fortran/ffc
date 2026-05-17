program test_hlfir_elemental_red
    ! RED Test: hlfir.elemental for array expressions
    ! This test MUST FAIL until we implement hlfir.elemental
    implicit none
    
    character(len=:), allocatable :: test_program, output_file, compile_cmd
    integer :: exit_code, unit, ios
    character(len=1024) :: line
    logical :: found_elemental = .false.
    logical :: found_index_func = .false.
    
    print *, "=== RED Test: hlfir.elemental generation ==="
    print *, "Expected: This test MUST FAIL (no hlfir.elemental yet)"
    print *
    
    ! Create Fortran program with array expressions
    test_program = "test_elemental.f90"
    open(newunit=unit, file=test_program, status='replace')
    write(unit, '(A)') 'program test_elemental'
    write(unit, '(A)') '    real, dimension(10) :: a, b, c'
    write(unit, '(A)') '    integer, dimension(10) :: idx'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Initialize arrays'
    write(unit, '(A)') '    a = 1.0'
    write(unit, '(A)') '    b = 2.0'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Elemental array expression: element-wise addition'
    write(unit, '(A)') '    c = a + b         ! Should generate hlfir.elemental'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! More complex elemental expression'
    write(unit, '(A)') '    c = 2.0 * a + b / 3.0  ! Should generate nested hlfir.elemental'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Elemental with array sections'
    write(unit, '(A)') '    c(2:8) = a(2:8) * b(2:8)  ! Should use index-based computation'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Elemental with WHERE mask'
    write(unit, '(A)') '    where (a > 0.5)'
    write(unit, '(A)') '        c = sqrt(a) + b    ! Conditional elemental operation'
    write(unit, '(A)') '    end where'
    write(unit, '(A)') '    '
    write(unit, '(A)') '    ! Elemental intrinsic functions'
    write(unit, '(A)') '    c = sin(a) + cos(b)    ! Should map to elemental operations'
    write(unit, '(A)') 'end program test_elemental'
    close(unit)
    
    ! Try to compile with HLFIR output
    output_file = "test_elemental_hlfir.mlir"
    compile_cmd = "./ffc " // test_program // " --emit-hlfir -o " // output_file
    
    print *, "Compiling: ", trim(compile_cmd)
    call execute_command_line(compile_cmd, exitstat=exit_code)
    
    if (exit_code == 0) then
        ! Check for hlfir.elemental in output
        open(newunit=unit, file=output_file, status='old', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                if (index(line, 'hlfir.elemental') > 0) then
                    found_elemental = .true.
                    ! Check for index function (lambda)
                    if (index(line, '^bb') > 0 .or. index(line, 'index') > 0) then
                        found_index_func = .true.
                    end if
                end if
            end do
            close(unit)
            
            if (found_elemental .and. found_index_func) then
                print *, "UNEXPECTED SUCCESS: Found hlfir.elemental with index function!"
                print *, "This should not happen in RED phase"
                stop 1
            else if (found_elemental) then
                print *, "PARTIAL: Found hlfir.elemental but no index function"
                print *, "Still failing as expected"
                stop 0
            else
                print *, "FAIL AS EXPECTED: No hlfir.elemental found"
                print *, "Currently only generates basic operations"
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
    
end program test_hlfir_elemental_red