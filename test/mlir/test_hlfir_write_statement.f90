program test_hlfir_write_statement
    use temp_utils, only: temp_dir_manager, create_test_cache_dir
    use system_utils, only: sys_run_command, sys_file_exists
    implicit none
    
    type(temp_dir_manager) :: temp_mgr
    character(len=:), allocatable :: test_file, command, cmd_output, cache_dir
    character(len=512) :: output_file
    character(len=1024) :: line
    integer :: exit_code, unit, ios
    logical :: all_passed
    
    all_passed = .true.
    print *, "=== HLFIR Write Statement Tests ==="
    
    ! Test 1: Basic write statement with HLFIR
    call test_basic_write()
    
    ! Test 2: Write with format specifier
    call test_formatted_write()
    
    ! Test 3: Write with iostat handling
    call test_write_with_iostat()
    
    ! Test 4: Write to file
    call test_write_to_file()
    
    if (.not. all_passed) then
        print *, "Some HLFIR write tests FAILED!"
        stop 1
    else
        print *, "All HLFIR write tests PASSED!"
    end if
    
contains
    
    subroutine test_basic_write()
        print *, "TEST: Basic write statement generates HLFIR operations"
        
        call temp_mgr%create('hlfir_write_test')
        cache_dir = create_test_cache_dir('hlfir_write')
        
        test_file = temp_mgr%get_file_path('write_test.f90')
        call write_test_program(test_file, &
            "program write_test" // new_line('a') // &
            "    integer :: x" // new_line('a') // &
            "    x = 42" // new_line('a') // &
            "    write(*,*) 'Value is:', x" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('hlfir_output.txt')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // &
                  trim(test_file) // ' > ' // trim(output_file) // ' 2>&1'
        call sys_run_command(command, cmd_output, exit_code)
        
        ! Read and check output
        cmd_output = ""
        open(newunit=unit, file=trim(output_file), status='old', action='read', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                cmd_output = cmd_output // trim(line) // new_line('a')
            end do
            close(unit)
        end if
        
        ! Check for HLFIR I/O operations
        if (index(cmd_output, 'fir.call @_FortranAioBeginExternalFormattedOutput') > 0 .or. &
            index(cmd_output, 'fir.call @_FortranAioBeginExternalListOutput') > 0) then
            print *, "  PASS: Found fir.call for I/O begin"
        else
            print *, "  FAIL: Expected fir.call @_FortranAioBeginExternalFormattedOutput or ListOutput"
            all_passed = .false.
        end if
        
        if (index(cmd_output, 'fir.string_lit') > 0) then
            print *, "  PASS: String literals use fir.string_lit"
        else
            print *, "  FAIL: Expected fir.string_lit for string literals"
            all_passed = .false.
        end if
        
        if (index(cmd_output, '!fir.ref<!fir.box<none>>') > 0 .or. &
            index(cmd_output, '!fir.') > 0) then
            print *, "  PASS: Uses FIR types"
        else
            print *, "  FAIL: Expected FIR types for I/O operations"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_formatted_write()
        print *, "TEST: Formatted write statement with HLFIR"
        
        call temp_mgr%create('hlfir_fmt_write_test')
        cache_dir = create_test_cache_dir('hlfir_fmt_write')
        
        test_file = temp_mgr%get_file_path('fmt_write_test.f90')
        call write_test_program(test_file, &
            "program fmt_write_test" // new_line('a') // &
            "    integer :: i, j" // new_line('a') // &
            "    i = 10" // new_line('a') // &
            "    j = 20" // new_line('a') // &
            "    write(*,'(A,I3,A,I3)') 'Values: ', i, ' and ', j" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('hlfir_output.txt')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // &
                  trim(test_file) // ' > ' // trim(output_file) // ' 2>&1'
        call sys_run_command(command, cmd_output, exit_code)
        
        ! Read and check output
        cmd_output = ""
        open(newunit=unit, file=trim(output_file), status='old', action='read', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                cmd_output = cmd_output // trim(line) // new_line('a')
            end do
            close(unit)
        end if
        
        ! Check for formatted output operations
        if (index(cmd_output, 'fir.call @_FortranAioBeginExternalFormattedOutput') > 0) then
            print *, "  PASS: Found fir.call for formatted output"
        else
            print *, "  FAIL: Expected fir.call @_FortranAioBeginExternalFormattedOutput"
            all_passed = .false.
        end if
        
        if (index(cmd_output, '(A,I3,A,I3)') > 0) then
            print *, "  PASS: Format string preserved"
        else
            print *, "  FAIL: Expected format string in output"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_write_with_iostat()
        print *, "TEST: Write statement with iostat handling"
        
        call temp_mgr%create('hlfir_iostat_test')
        cache_dir = create_test_cache_dir('hlfir_iostat')
        
        test_file = temp_mgr%get_file_path('iostat_test.f90')
        call write_test_program(test_file, &
            "program iostat_test" // new_line('a') // &
            "    integer :: stat" // new_line('a') // &
            "    write(*,*, iostat=stat) 'Testing iostat'" // new_line('a') // &
            "    if (stat /= 0) then" // new_line('a') // &
            "        print *, 'Write failed'" // new_line('a') // &
            "    end if" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('hlfir_output.txt')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // &
                  trim(test_file) // ' > ' // trim(output_file) // ' 2>&1'
        call sys_run_command(command, cmd_output, exit_code)
        
        ! Read and check output
        cmd_output = ""
        open(newunit=unit, file=trim(output_file), status='old', action='read', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                cmd_output = cmd_output // trim(line) // new_line('a')
            end do
            close(unit)
        end if
        
        ! Check for iostat handling
        if (index(cmd_output, 'fir.call @_FortranAioEndIoStatement') > 0) then
            print *, "  PASS: Found fir.call for I/O end with status"
        else
            print *, "  FAIL: Expected proper iostat handling with HLFIR"
            all_passed = .false.
        end if
    end subroutine
    
    subroutine test_write_to_file()
        print *, "TEST: Write to file with HLFIR"
        
        call temp_mgr%create('hlfir_file_write_test')
        cache_dir = create_test_cache_dir('hlfir_file_write')
        
        test_file = temp_mgr%get_file_path('file_write_test.f90')
        call write_test_program(test_file, &
            "program file_write_test" // new_line('a') // &
            "    integer :: unit" // new_line('a') // &
            "    open(newunit=unit, file='output.txt', status='replace')" // new_line('a') // &
            "    write(unit,*) 'Writing to file'" // new_line('a') // &
            "    write(unit,'(A,I5)') 'Number: ', 12345" // new_line('a') // &
            "    close(unit)" // new_line('a') // &
            "end program")
        
        output_file = temp_mgr%get_file_path('hlfir_output.txt')
        command = 'FORTRAN_CACHE_DIR=' // cache_dir // ' fpm run fortran -- --emit-hlfir ' // &
                  trim(test_file) // ' > ' // trim(output_file) // ' 2>&1'
        call sys_run_command(command, cmd_output, exit_code)
        
        ! Read and check output
        cmd_output = ""
        open(newunit=unit, file=trim(output_file), status='old', action='read', iostat=ios)
        if (ios == 0) then
            do
                read(unit, '(A)', iostat=ios) line
                if (ios /= 0) exit
                cmd_output = cmd_output // trim(line) // new_line('a')
            end do
            close(unit)
        end if
        
        ! Check for file I/O operations
        if (index(cmd_output, 'fir.call') > 0 .and. &
            (index(cmd_output, '!fir.') > 0 .or. index(cmd_output, 'fir.') > 0)) then
            print *, "  PASS: File I/O uses HLFIR operations"
        else
            print *, "  FAIL: Expected HLFIR file I/O operations"
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
    
end program test_hlfir_write_statement