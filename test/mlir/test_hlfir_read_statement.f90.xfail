program test_hlfir_read_statement
    use iso_fortran_env, only: output_unit, error_unit
    implicit none
    
    logical :: all_passed = .true.
    
    print *, '=== HLFIR Read Statement Tests ==='
    
    call test_basic_read()
    call test_formatted_read()
    call test_read_with_iostat()
    call test_file_read()
    
    if (.not. all_passed) then
        print *, 'Some HLFIR read tests FAILED!'
        stop 1
    else
        print *, 'All HLFIR read tests PASSED!'
    end if
    
contains
    
    subroutine test_basic_read()
        integer :: exit_code
        
        print *, 'TEST: Basic read statement generates HLFIR operations'
        
        ! Create test program
        call execute_command_line('echo "program read_test' // new_line('a') // &
                                 '    integer :: x' // new_line('a') // &
                                 '    read(*,*) x' // new_line('a') // &
                                 'end program" > test_read.f90')
        
        ! Generate HLFIR
        call execute_command_line('./build/gfortran_C7198B2030710F70/app/fortran --emit-hlfir ' // &
                                 'test_read.f90 > out.mlir 2>&1', exitstat=exit_code)
        
        if (exit_code == 0) then
            ! Check for read operations
            call execute_command_line('grep -q "fir.call @_FortranAioBeginExternalListInput" out.mlir', exitstat=exit_code)
            if (exit_code == 0) then
                print *, '  PASS: Found fir.call for read input'
            else
                call execute_command_line('grep -q "fir.call @_FortranAioBeginExternalFormattedInput" out.mlir', exitstat=exit_code)
                if (exit_code == 0) then
                    print *, '  PASS: Found fir.call for formatted read input'
                else
                    print *, '  FAIL: Expected fir.call @_FortranAioBeginExternalListInput or FormattedInput'
                    all_passed = .false.
                end if
            end if
            
            call execute_command_line('grep -q "fir.call @_FortranAioInputInteger" out.mlir', exitstat=exit_code)
            if (exit_code == 0) then
                print *, '  PASS: Found integer input operation'
            else
                print *, '  FAIL: Expected fir.call @_FortranAioInputInteger'
                all_passed = .false.
            end if
        else
            print *, '  FAIL: HLFIR generation failed'
            all_passed = .false.
        end if
        
        ! Cleanup
        call execute_command_line('rm -f test_read.f90 out.mlir')
    end subroutine
    
    subroutine test_formatted_read()
        integer :: exit_code
        
        print *, 'TEST: Formatted read statement with HLFIR'
        
        ! Create test program
        call execute_command_line('echo "program fmt_read_test' // new_line('a') // &
                                 '    integer :: i, j' // new_line('a') // &
                                 '    read(*,''(I3,1X,I3)'') i, j' // new_line('a') // &
                                 'end program" > test_fmt_read.f90')
        
        ! Generate HLFIR
        call execute_command_line('./build/gfortran_C7198B2030710F70/app/fortran --emit-hlfir ' // &
                                 'test_fmt_read.f90 > out.mlir 2>&1', exitstat=exit_code)
        
        if (exit_code == 0) then
            ! Check for formatted read
            call execute_command_line('grep -q "fir.string_lit \"(I3,1X,I3)\"" out.mlir', exitstat=exit_code)
            if (exit_code == 0) then
                print *, '  PASS: Found format string literal'
            else
                print *, '  FAIL: Expected format string as fir.string_lit'
                all_passed = .false.
            end if
            
            call execute_command_line('grep -q "fir.call.*Input" out.mlir', exitstat=exit_code)
            if (exit_code == 0) then
                print *, '  PASS: Found input operations'
            else
                print *, '  FAIL: Expected input operations'
                all_passed = .false.
            end if
        else
            print *, '  FAIL: HLFIR generation failed'
            all_passed = .false.
        end if
        
        ! Cleanup
        call execute_command_line('rm -f test_fmt_read.f90 out.mlir')
    end subroutine
    
    subroutine test_read_with_iostat()
        integer :: exit_code
        
        print *, 'TEST: Read statement with iostat handling'
        
        ! Create test program
        call execute_command_line('echo "program iostat_read_test' // new_line('a') // &
                                 '    integer :: x, ios' // new_line('a') // &
                                 '    read(*,*,iostat=ios) x' // new_line('a') // &
                                 'end program" > test_iostat_read.f90')
        
        ! Generate HLFIR
        call execute_command_line('./build/gfortran_C7198B2030710F70/app/fortran --emit-hlfir ' // &
                                 'test_iostat_read.f90 > out.mlir 2>&1', exitstat=exit_code)
        
        if (exit_code == 0) then
            ! For now, just check basic read generation
            ! iostat is not yet supported in the parser
            print *, '  SKIP: iostat not yet implemented in parser'
        else
            print *, '  FAIL: HLFIR generation failed'
            all_passed = .false.
        end if
        
        ! Cleanup
        call execute_command_line('rm -f test_iostat_read.f90 out.mlir')
    end subroutine
    
    subroutine test_file_read()
        integer :: exit_code
        
        print *, 'TEST: Read from file with HLFIR'
        
        ! Create test program
        call execute_command_line('echo "program file_read_test' // new_line('a') // &
                                 '    integer :: x' // new_line('a') // &
                                 '    open(10, file=''data.txt'')'  // new_line('a') // &
                                 '    read(10,*) x' // new_line('a') // &
                                 '    close(10)' // new_line('a') // &
                                 'end program" > test_file_read.f90')
        
        ! Generate HLFIR
        call execute_command_line('./build/gfortran_C7198B2030710F70/app/fortran --emit-hlfir ' // &
                                 'test_file_read.f90 > out.mlir 2>&1', exitstat=exit_code)
        
        if (exit_code == 0) then
            ! Check for file operations
            call execute_command_line('grep -q "fir.call.*Input" out.mlir', exitstat=exit_code)
            if (exit_code == 0) then
                print *, '  PASS: Generated file read operations'
            else
                print *, '  FAIL: Expected file read operations'
                all_passed = .false.
            end if
        else
            print *, '  FAIL: HLFIR generation failed'
            all_passed = .false.
        end if
        
        ! Cleanup
        call execute_command_line('rm -f test_file_read.f90 out.mlir')
    end subroutine
    
end program test_hlfir_read_statement