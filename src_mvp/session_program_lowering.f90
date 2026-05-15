module session_program_lowering
    use, intrinsic :: iso_c_binding, only: c_int64_t
    use fortfront, only: ast_arena_t, binary_op_node, literal_node, &
                         program_node, stop_node
    use liric_session_bindings, only: liric_session_t, liric_session_create, &
                                      lr_operand_desc_t, LR_OP_ADD, LR_OP_MUL, &
                                      LR_OP_SDIV, LR_OP_SUB
    implicit none
    private

    public :: lower_program_to_liric_exe

contains

    subroutine lower_program_to_liric_exe(arena, root_index, output_path, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(liric_session_t) :: session
        type(lr_operand_desc_t) :: return_value

        call validate_program(arena, root_index, error_msg)
        if (len_trim(error_msg) > 0) return

        call liric_session_create(session, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. session%begin_i32_main(error_msg)) then
            call session%destroy()
            return
        end if

        call lower_program_return(arena, root_index, session, return_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) then
            call session%destroy()
            return
        end if

        if (.not. session%emit_ret_i32_operand(return_value, error_msg)) then
            call session%destroy()
            return
        end if

        if (.not. session%finish_and_emit_exe(output_path, error_msg)) then
            call session%destroy()
            return
        end if

        call session%destroy()
        call set_empty(error_msg)
    end subroutine lower_program_to_liric_exe

    subroutine validate_program(arena, root_index, error_msg)
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
            continue
        class default
            error_msg = 'direct LIRIC session MVP only supports a top-level program unit'
            return
        end select

        call set_empty(error_msg)
    end subroutine validate_program

    subroutine lower_program_return(arena, root_index, session, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) then
                value = session%i32_immediate(0_c_int64_t)
                call set_empty(error_msg)
            else if (size(program%body_indices) == 0) then
                value = session%i32_immediate(0_c_int64_t)
                call set_empty(error_msg)
            else if (size(program%body_indices) == 1) then
                call lower_return_statement(arena, program%body_indices(1), &
                                            session, value, error_msg)
            else
                error_msg = 'direct LIRIC session MVP only supports one statement'
            end if
        class default
            error_msg = 'direct LIRIC session MVP only supports a program node'
        end select
    end subroutine lower_program_return

    subroutine lower_return_statement(arena, node_index, session, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'program body index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (stop_node)
            if (node%stop_code_index <= 0) then
                value = session%i32_immediate(0_c_int64_t)
                call set_empty(error_msg)
            else
                call lower_i32_expression(arena, node%stop_code_index, session, &
                                          value, error_msg)
            end if
        class default
            error_msg = 'direct LIRIC session MVP only supports STOP statements'
        end select
    end subroutine lower_return_statement

    recursive subroutine lower_i32_expression(arena, node_index, session, &
                                              value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(liric_session_t), intent(inout) :: session
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            call parse_i32_literal(node%value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = session%i32_immediate(literal_value)
        type is (binary_op_node)
            call lower_i32_expression(arena, node%left_index, session, lhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, node%right_index, session, rhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_opcode(node%operator, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. session%emit_i32_binary(opcode, lhs, rhs, value, &
                                              error_msg)) return
        class default
            error_msg = 'direct LIRIC session MVP only supports integer expressions'
        end select
    end subroutine lower_i32_expression

    subroutine parse_i32_literal(text, value, error_msg)
        character(len=*), intent(in) :: text
        integer(c_int64_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_status

        read (text, *, iostat=io_status) value
        if (io_status == 0) then
            call set_empty(error_msg)
        else
            error_msg = 'invalid integer literal for direct LIRIC lowering: '// &
                        trim(text)
        end if
    end subroutine parse_i32_literal

    subroutine integer_opcode(source_op, opcode, error_msg)
        character(len=*), intent(in) :: source_op
        integer, intent(out) :: opcode
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            opcode = LR_OP_ADD
        case ('-')
            opcode = LR_OP_SUB
        case ('*')
            opcode = LR_OP_MUL
        case ('/')
            opcode = LR_OP_SDIV
        case default
            error_msg = 'direct LIRIC session MVP does not support operator: '// &
                        trim(source_op)
            opcode = 0
        end select
    end subroutine integer_opcode

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module session_program_lowering
