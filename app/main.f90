program ffc_main
    use backend_factory, only: create_backend
    use backend_interface, only: backend_t, backend_options_t
    use ast_core, only: ast_arena_t, create_ast_stack
    use frontend, only: lex_source, parse_tokens, analyze_semantics
    implicit none

    character(len=256) :: input_file, output_file, error_msg
    class(backend_t), allocatable :: backend
    type(backend_options_t) :: options
    type(ast_arena_t) :: arena
    integer :: prog_index, num_args
    character(len=:), allocatable :: code

    ! Get command line arguments
    num_args = command_argument_count()
    if (num_args < 1) then
        print *, "Usage: ffc <input.f90> [options]"
        print *, "Options:"
        print *, "  -o <file>     Output file"
        print *, "  --emit-hlfir  Emit HLFIR code"
        print *, "  --emit-fir    Emit FIR code" 
        print *, "  --emit-llvm   Emit LLVM IR"
        stop 1
    end if

    call get_command_argument(1, input_file)
    
    ! Default options
    options%compile_mode = .true.
    options%optimize = .true.
    options%generate_llvm = .true.
    options%enable_ad = .false.
    options%emit_hlfir = .false.
    options%emit_fir = .false.
    options%emit_llvm = .false.
    options%generate_executable = .true.
    
    ! Parse command line options
    call parse_command_line(options, output_file)
    
    if (len_trim(output_file) > 0) then
        options%output_file = trim(output_file)
        ! Check if generating object file vs executable
        options%generate_executable = index(output_file, '.o') == 0 .and. &
                                     index(output_file, '.obj') == 0
    end if

    ! Create MLIR backend
    call create_backend("mlir", backend, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, "Error creating backend: ", trim(error_msg)
        stop 1
    end if

    ! Compile source file through frontend
    arena = create_ast_stack()
    call compile_source_file(input_file, arena, prog_index, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, "Error: ", trim(error_msg)
        stop 1
    end if

    ! Generate code using backend
    call backend%generate_code(arena, prog_index, options, code, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, "Error generating code: ", trim(error_msg)
        stop 1
    end if

    ! Output result
    if (options%emit_hlfir .or. options%emit_fir .or. options%emit_llvm) then
        if (len_trim(output_file) > 0) then
            call write_to_file(output_file, code)
        else
            print '(a)', code
        end if
    end if

    print *, "Compilation successful"
    if (len_trim(output_file) > 0) then
        print *, "Output written to: ", trim(output_file)
    end if

contains

    subroutine parse_command_line(opts, out_file)
        type(backend_options_t), intent(inout) :: opts
        character(len=*), intent(out) :: out_file
        integer :: i
        character(len=256) :: arg
        
        out_file = ""
        i = 2  ! Start after input file
        
        do while (i <= command_argument_count())
            call get_command_argument(i, arg)
            
            select case (trim(arg))
            case ("-o")
                i = i + 1
                if (i <= command_argument_count()) then
                    call get_command_argument(i, out_file)
                end if
            case ("--emit-hlfir")
                opts%emit_hlfir = .true.
            case ("--emit-fir") 
                opts%emit_fir = .true.
            case ("--emit-llvm")
                opts%emit_llvm = .true.
            end select
            
            i = i + 1
        end do
    end subroutine parse_command_line

    subroutine compile_source_file(filename, arena, prog_idx, err_msg)
        character(len=*), intent(in) :: filename
        type(ast_arena_t), intent(inout) :: arena
        integer, intent(out) :: prog_idx
        character(len=*), intent(out) :: err_msg
        
        ! This would use the frontend to compile
        ! For now, simplified stub
        err_msg = ""
        prog_idx = 1
    end subroutine compile_source_file

    subroutine write_to_file(filename, content)
        character(len=*), intent(in) :: filename, content
        integer :: unit
        
        open(newunit=unit, file=filename, status='replace')
        write(unit, '(a)') content
        close(unit)
    end subroutine write_to_file

end program ffc_main