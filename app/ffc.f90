program ffc_main
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe, &
                                        lower_program_to_liric_object
    use ffc_cli_options, only: cli_options_t, parse_arguments, CLI_PATH_LEN
    implicit none

    type(compiler_frontend_options_t) :: frontend_options
    type(compiler_frontend_result_t) :: frontend_result
    type(cli_options_t) :: opts
    character(len=CLI_PATH_LEN), allocatable :: argv(:)
    character(len=:), allocatable :: error_msg
    character(len=CLI_PATH_LEN) :: output_file
    integer :: nargs, i

    nargs = command_argument_count()
    if (nargs < 1) then
        call print_usage()
        stop 1
    end if

    allocate (argv(nargs))
    do i = 1, nargs
        call get_command_argument(i, argv(i))
    end do
    call parse_arguments(argv, opts)
    if (opts%error) then
        print '(A)', opts%error_message
        stop 1
    end if

    frontend_options = compiler_frontend_options_t()
    frontend_options%run_semantics = .true.
    frontend_options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_file(trim(opts%input_file), frontend_result, &
                                    frontend_options)
    if (.not. frontend_result%success()) then
        print '(A)', trim(frontend_result%diagnostic_text)
        stop 1
    end if

    output_file = opts%output_file
    if (len_trim(output_file) == 0) output_file = default_output_name(opts%emit_object)

    if (opts%emit_object) then
        call lower_program_to_liric_object(frontend_result%arena, &
                                           frontend_result%root_index, &
                                           trim(output_file), error_msg, &
                                           opts%include_paths)
    else
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, &
                                        trim(output_file), error_msg, &
                                        opts%include_paths)
    end if
    if (len_trim(error_msg) > 0) then
        print '(A)', trim(error_msg)
        stop 1
    end if

contains

    subroutine print_usage()
        print '(A)', 'Usage: ffc <input.f90> [options]'
        print '(A)', 'Options:'
        print '(A)', '  -o <file>     Output file'
        print '(A)', '  -c            Emit object file'
        print '(A)', '  -I <dir>      Add module/include search directory'
    end subroutine print_usage

    function default_output_name(emit_object) result(name)
        logical, intent(in) :: emit_object
        character(len=CLI_PATH_LEN) :: name

        if (emit_object) then
            name = 'a.o'
        else
            name = 'a.out'
        end if
    end function default_output_name

end program ffc_main
