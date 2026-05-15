program test_block_if_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD, &
                         ast_to_json
    use empty_program_lowering, only: lower_empty_program_to_llvm
    use liric_bindings, only: liric_compile_ll_to_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: llvm_ir
    character(len=:), allocatable :: error_msg
    character(len=:), allocatable :: json
    character(len=64) :: output_line
    character(len=*), parameter :: exe_path = '/tmp/ffc_block_if_test'
    character(len=*), parameter :: out_path = '/tmp/ffc_block_if_test.out'
    integer :: exit_stat
    integer :: io_stat
    integer :: unit
    character(len=*), parameter :: source = &
                                   'program main'//new_line('a')// &
                                   '  integer :: x'//new_line('a')// &
                                   '  x = 5'//new_line('a')// &
                                   '  if (x == 5) then'//new_line('a')// &
                                   '    print *, 1'//new_line('a')// &
                                   '  else'//new_line('a')// &
                                   '    print *, 0'//new_line('a')// &
                                   '  end if'//new_line('a')// &
                                   'end program main'

    print *, '=== block if compiler test ==='

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call lower_empty_program_to_llvm(frontend_result%arena, &
                                     frontend_result%root_index, llvm_ir, &
                                     error_msg)
    if (len_trim(error_msg) > 0) then
        call ast_to_json(frontend_result%arena, frontend_result%root_index, json)
        print *, trim(json)
        print *, 'FAIL: lowering failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path//' '//out_path)
    if (.not. liric_compile_ll_to_exe(llvm_ir, exe_path, error_msg)) then
        print *, 'FAIL: LIRIC executable emission failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
    if (exit_stat /= 0) then
        print *, 'FAIL: executable exit status ', exit_stat
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        stop 1
    end if

    open (newunit=unit, file=out_path, status='old', action='read', &
          iostat=io_stat)
    if (io_stat /= 0) then
        print *, 'FAIL: could not open captured output'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        stop 1
    end if
    read (unit, '(A)', iostat=io_stat) output_line
    close (unit)
    call execute_command_line('rm -f '//exe_path//' '//out_path)

    if (io_stat /= 0 .or. trim(adjustl(output_line)) /= '1') then
        print *, 'FAIL: expected output 1, got ', trim(output_line)
        stop 1
    end if

    print *, 'PASS: block if lowering'
end program test_block_if_compiler
