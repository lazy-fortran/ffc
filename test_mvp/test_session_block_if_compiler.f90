program test_session_block_if_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: error_msg
    character(len=*), parameter :: exe_path = '/tmp/ffc_session_block_if_test'
    integer :: exit_stat
    integer :: cmd_stat
    character(len=*), parameter :: source = &
                                   'program main'//new_line('a')// &
                                   'if (2 < 3) then'//new_line('a')// &
                                   '    stop 7'//new_line('a')// &
                                   'else'//new_line('a')// &
                                   '    stop 1'//new_line('a')// &
                                   'end if'//new_line('a')// &
                                   'end program main'

    print *, '=== direct session block if compiler test ==='

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path)
    call lower_program_to_liric_exe(frontend_result%arena, &
                                    frontend_result%root_index, exe_path, &
                                    error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
        stop 1
    end if

    call execute_command_line(exe_path, exitstat=exit_stat, cmdstat=cmd_stat)
    call execute_command_line('rm -f '//exe_path)
    if (cmd_stat /= 0) then
        print *, 'FAIL: emitted executable did not run'
        stop 1
    end if
    if (exit_stat /= 7) then
        print *, 'FAIL: executable exit status ', exit_stat
        stop 1
    end if

    print *, 'PASS: block IF lowers through direct LIRIC session'
end program test_session_block_if_compiler
