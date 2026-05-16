program test_session_character_variable_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    type(compiler_frontend_options_t) :: options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=:), allocatable :: error_msg
    character(len=6) :: output_text
    character(len=64) :: output_line
    character(len=*), parameter :: exe_path = '/tmp/ffc_session_char_var_test'
    character(len=*), parameter :: out_path = &
                                   '/tmp/ffc_session_char_var_test.out'
    integer :: exit_stat
    integer :: file_size
    integer :: io_stat
    integer :: unit
    character(len=*), parameter :: source = &
                                   'program main'//new_line('a')// &
                                   '  character(len=5) :: s'//new_line('a')// &
                                   '  s = "hello"'//new_line('a')// &
                                   '  print *, s'//new_line('a')// &
                                   'end program main'

    print *, '=== direct session character variable compiler test ==='

    options = compiler_frontend_options_t()
    options%run_semantics = .true.
    options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_string(source, frontend_result, options)
    if (.not. frontend_result%success()) then
        print *, 'FAIL: FortFront rejected source: ', &
            trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call execute_command_line('rm -f '//exe_path//' '//out_path)
    call lower_program_to_liric_exe(frontend_result%arena, &
                                    frontend_result%root_index, exe_path, &
                                    error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
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

    inquire (file=out_path, size=file_size)
    open (newunit=unit, file=out_path, status='old', access='stream', &
          form='unformatted', action='read', iostat=io_stat)
    if (io_stat /= 0) then
        print *, 'FAIL: could not open captured output as stream'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        stop 1
    end if
    read (unit, iostat=io_stat) output_text
    close (unit)
    call execute_command_line('rm -f '//exe_path//' '//out_path)

    if (io_stat /= 0 .or. file_size /= 6 .or. &
        output_text /= 'hello'//new_line('a')) then
        print *, 'FAIL: expected fixed-length output hello, got ', output_line
        stop 1
    end if

    print *, 'PASS: character variables lower through direct LIRIC session'
end program test_session_character_variable_compiler
