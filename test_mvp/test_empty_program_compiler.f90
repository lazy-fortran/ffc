program test_empty_program_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use empty_program_lowering, only: lower_empty_program_to_llvm
    use liric_bindings, only: liric_compile_ll_to_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: llvm_ir
    character(len=:), allocatable :: error_msg
    character(len=*), parameter :: exe_path = '/tmp/ffc_empty_program_test'
    integer :: exit_stat
    character(len=*), parameter :: source = &
                                   'program main'//new_line('a')// &
                                   'end program main'

    print *, '=== empty program compiler test ==='

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
        print *, 'FAIL: lowering failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path)
    if (.not. liric_compile_ll_to_exe(llvm_ir, exe_path, error_msg)) then
        print *, 'FAIL: LIRIC executable emission failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line(exe_path, exitstat=exit_stat)
    call execute_command_line('rm -f '//exe_path)
    if (exit_stat /= 0) then
        print *, 'FAIL: executable exit status ', exit_stat
        stop 1
    end if

    print *, 'PASS: empty program compiles and runs'
end program test_empty_program_compiler
