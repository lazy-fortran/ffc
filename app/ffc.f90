program ffc_main
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe, &
                                        lower_program_to_liric_object
    implicit none

    type(compiler_frontend_options_t) :: frontend_options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=512) :: input_file
    character(len=512) :: output_file
    character(len=:), allocatable :: error_msg
    logical :: emit_object

    if (command_argument_count() < 1) then
        call print_usage()
        stop 1
    end if

    call get_command_argument(1, input_file)
    call parse_command_line(output_file, emit_object)

    frontend_options = compiler_frontend_options_t()
    frontend_options%run_semantics = .true.
    frontend_options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_file(trim(input_file), frontend_result, &
                                    frontend_options)
    if (.not. frontend_result%success()) then
        print '(A)', trim(frontend_result%diagnostic_text)
        stop 1
    end if

    if (len_trim(output_file) == 0) output_file = default_output_name(emit_object)

    if (emit_object) then
        call lower_program_to_liric_object(frontend_result%arena, &
                                           frontend_result%root_index, &
                                           trim(output_file), error_msg)
    else
        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, &
                                        trim(output_file), error_msg)
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
    end subroutine print_usage

    subroutine parse_command_line(out_file, emit_object)
        character(len=*), intent(out) :: out_file
        logical, intent(out) :: emit_object
        integer :: i
        character(len=512) :: arg

        out_file = ''
        emit_object = .false.
        i = 2
        do while (i <= command_argument_count())
            call get_command_argument(i, arg)
            select case (trim(arg))
            case ('-c')
                emit_object = .true.
            case ('-o')
                i = i + 1
                if (i > command_argument_count()) then
                    print '(A)', 'Missing value for -o'
                    stop 1
                end if
                call get_command_argument(i, out_file)
            case default
                print '(A,A)', 'Unknown option: ', trim(arg)
                stop 1
            end select
            i = i + 1
        end do
    end subroutine parse_command_line

    function default_output_name(emit_object) result(name)
        logical, intent(in) :: emit_object
        character(len=512) :: name

        if (emit_object) then
            name = 'a.o'
        else
            name = 'a.out'
        end if
    end function default_output_name

end program ffc_main
