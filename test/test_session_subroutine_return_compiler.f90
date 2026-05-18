program test_session_subroutine_return_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: error_msg
    character(len=*), parameter :: exe_path = '/tmp/ffc_session_subr_return_test'
    integer :: exit_stat, cmd_stat
    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: x'//new_line('a')// &
        '  x = 0'//new_line('a')// &
        '  call maybe_set(1, x)'//new_line('a')// &
        '  call maybe_set(0, x)'//new_line('a')// &
        '  stop x'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine maybe_set(skip, target)'//new_line('a')// &
        '    integer, intent(in) :: skip'//new_line('a')// &
        '    integer, intent(inout) :: target'//new_line('a')// &
        '    if (skip == 1) then'//new_line('a')// &
        '      return'//new_line('a')// &
        '    else'//new_line('a')// &
        '      target = target + 7'//new_line('a')// &
        '    end if'//new_line('a')// &
        '  end subroutine maybe_set'//new_line('a')// &
        'end program main'

    print *, '=== subroutine early-return compiler test ==='

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
        print *, 'FAIL: exit status ', exit_stat, ' expected 7'
        stop 1
    end if

    print *, 'PASS: subroutine early return lowers through direct LIRIC session'
end program test_session_subroutine_return_compiler
