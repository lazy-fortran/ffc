module session_program_lowering
    use fortfront, only: ast_arena_t, program_node
    use liric_session_bindings, only: liric_session_t, liric_session_create
    implicit none
    private

    public :: lower_empty_program_to_liric_exe

contains

    subroutine lower_empty_program_to_liric_exe(arena, root_index, output_path, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(liric_session_t) :: session

        call validate_empty_program(arena, root_index, error_msg)
        if (len_trim(error_msg) > 0) return

        call liric_session_create(session, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. session%emit_ret_i32_main_exe(0, output_path, error_msg)) then
            call session%destroy()
            return
        end if

        call session%destroy()
        call set_empty(error_msg)
    end subroutine lower_empty_program_to_liric_exe

    subroutine validate_empty_program(arena, root_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=:), allocatable, intent(out) :: error_msg

        if (root_index <= 0) then
            error_msg = 'FortFront did not return a root program index'
            return
        end if

        if (.not. arena%has_node_at(root_index)) then
            error_msg = 'FortFront root index does not reference an AST node'
            return
        end if

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (allocated(program%body_indices)) then
                if (size(program%body_indices) > 0) then
                    error_msg = 'direct LIRIC session MVP only supports empty programs'
                    return
                end if
            end if
        class default
            error_msg = 'direct LIRIC session MVP only supports a top-level program unit'
            return
        end select

        call set_empty(error_msg)
    end subroutine validate_empty_program

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module session_program_lowering
