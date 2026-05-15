program ffc_main
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_file, INPUT_MODE_STANDARD
    use empty_program_lowering, only: lower_empty_program_to_llvm
    use liric_bindings, only: liric_compile_ll_to_exe
    implicit none

    type(compiler_frontend_options_t) :: frontend_options
    type(compiler_frontend_result_t) :: frontend_result
    character(len=512) :: input_file
    character(len=512) :: output_file
    character(len=:), allocatable :: llvm_ir
    character(len=:), allocatable :: error_msg
    logical :: emit_llvm

    if (command_argument_count() < 1) then
        call print_usage()
        stop 1
    end if

    call get_command_argument(1, input_file)
    call parse_command_line(output_file, emit_llvm)

    frontend_options = compiler_frontend_options_t()
    frontend_options%run_semantics = .true.
    frontend_options%input_mode = INPUT_MODE_STANDARD

    call compile_frontend_from_file(trim(input_file), frontend_result, &
                                    frontend_options)
    if (.not. frontend_result%success()) then
        print '(A)', trim(frontend_result%diagnostic_text)
        stop 1
    end if

    call lower_empty_program_to_llvm(frontend_result%arena, &
                                     frontend_result%root_index, llvm_ir, &
                                     error_msg)
    if (len_trim(error_msg) > 0) then
        print '(A)', trim(error_msg)
        stop 1
    end if

    if (emit_llvm) then
        if (len_trim(output_file) > 0) then
            call write_to_file(trim(output_file), llvm_ir)
        else
            print '(A)', llvm_ir
        end if
        stop 0
    end if

    if (len_trim(output_file) == 0) output_file = default_output_name()

    if (.not. liric_compile_ll_to_exe(llvm_ir, trim(output_file), error_msg)) then
        print '(A)', trim(error_msg)
        stop 1
    end if

contains

    subroutine print_usage()
        print '(A)', 'Usage: ffc <input.f90> [options]'
        print '(A)', 'Options:'
        print '(A)', '  -o <file>     Output executable or LLVM file'
        print '(A)', '  --emit-llvm   Emit LLVM IR instead of compiling'
    end subroutine print_usage

    subroutine parse_command_line(out_file, emit_ir)
        character(len=*), intent(out) :: out_file
        logical, intent(out) :: emit_ir
        integer :: i
        character(len=512) :: arg

        out_file = ''
        emit_ir = .false.
        i = 2
        do while (i <= command_argument_count())
            call get_command_argument(i, arg)
            select case (trim(arg))
            case ('-o')
                i = i + 1
                if (i > command_argument_count()) then
                    print '(A)', 'Missing value for -o'
                    stop 1
                end if
                call get_command_argument(i, out_file)
            case ('--emit-llvm')
                emit_ir = .true.
            case default
                print '(A,A)', 'Unknown option: ', trim(arg)
                stop 1
            end select
            i = i + 1
        end do
    end subroutine parse_command_line

    function default_output_name() result(name)
        character(len=512) :: name

        name = 'a.out'
    end function default_output_name

    subroutine write_to_file(filename, content)
        character(len=*), intent(in) :: filename
        character(len=*), intent(in) :: content
        integer :: unit

        open (newunit=unit, file=filename, status='replace', action='write')
        write (unit, '(A)') content
        close (unit)
    end subroutine write_to_file

end program ffc_main
