! MLIR Backend Implementation Module
! This module provides the actual implementation of the MLIR backend
module mlir_backend_impl
    use backend_interface
    use fortfront
    use mlir_backend_types
    use mlir_backend, only: generate_mlir_module
    use mlir_compile, only: compile_mlir_to_output, apply_mlir_lowering_passes
    use logger, only: log_debug, log_info
    implicit none
    private

    public :: mlir_backend_impl_t

    ! Extended type with real implementation
    type, extends(mlir_backend_t) :: mlir_backend_impl_t
    contains
        procedure :: generate_code => mlir_generate_code_impl
    end type mlir_backend_impl_t

contains

    ! Real implementation of generate_code
    subroutine mlir_generate_code_impl(this, arena, prog_index, options, output, error_msg)
        class(mlir_backend_impl_t), intent(inout) :: this
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable, intent(out) :: output
        character(len=*), intent(out) :: error_msg
        
        character(len=:), allocatable :: mlir_code
        logical :: success

        error_msg = ""
        output = ""

        call log_debug("MLIR backend generate_code called")
        call log_debug("compile_mode = " // merge("true ", "false", options%compile_mode))

        ! Generate the MLIR module
        call log_debug("MLIR backend about to generate MLIR module")
        mlir_code = generate_mlir_module(this, arena, prog_index, options)
        block
            character(len=32) :: length_str
            write(length_str, '(I0)') len_trim(mlir_code)
            call log_debug("Generated MLIR length: " // trim(length_str))
        end block

        if (allocated(this%error_messages)) then
            error_msg = this%error_messages
            this%error_messages = ""  ! Clear for next run
            return
        end if

        if (options%compile_mode) then
            ! Compile to object code or executable
            ! Apply lowering passes if standard dialects requested
            if (this%use_standard_dialects) then
                call apply_mlir_lowering_passes(mlir_code, "llvm", output, success, error_msg)
                if (.not. success) then
                    return
                end if
                mlir_code = output  ! Use lowered code for compilation
            end if
            ! Generate object file or executable
            call compile_mlir_to_output(mlir_code, options, success, error_msg)
            if (.not. success) then
                return
            end if
            ! Don't return text output when compiling to file
            output = ""
        else
            ! Default: return the high-level MLIR (--emit-hlfir or no flag)
            output = mlir_code
        end if
    end subroutine mlir_generate_code_impl

end module mlir_backend_impl