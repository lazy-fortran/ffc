module session_program_lowering
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t
    use fortfront, only: assignment_node, ast_arena_t, binary_op_node, &
                         declaration_node, do_loop_node, identifier_node, if_node, &
                         literal_node, print_statement_node, program_node, &
                         stop_node
    use liric_session_bindings, only: liric_session_t, liric_session_create, &
                                      lr_operand_desc_t
    use liric_session_control_bindings, only: create_liric_block, &
                                              emit_liric_br, &
                                              emit_liric_condbr, &
                                              emit_liric_i32_icmp, &
                                              emit_liric_i32_phi, &
                                              set_liric_block
    use liric_session_io_bindings, only: emit_liric_print_i32, &
                                         prepare_liric_i32_print
    use session_lowering_ops, only: integer_compare_predicate, &
                                    integer_opcode, parse_i32_literal
    implicit none
    private

    public :: lower_program_to_liric_exe

    integer, parameter :: MAX_SYMBOLS = 64

    type :: symbol_t
        character(len=64) :: name = ''
        type(lr_operand_desc_t) :: value
    end type symbol_t

    type :: lowering_context_t
        type(liric_session_t) :: session
        type(symbol_t) :: symbols(MAX_SYMBOLS)
        integer :: symbol_count = 0
        integer(c_int32_t) :: current_block_id = 0_c_int32_t
        integer(c_int32_t) :: i32_print_format_id = -1_c_int32_t
        logical :: current_block_terminated = .false.
    end type lowering_context_t

    type :: branch_result_t
        type(symbol_t) :: symbols(MAX_SYMBOLS)
        integer :: symbol_count = 0
        integer(c_int32_t) :: predecessor_block_id = 0_c_int32_t
        logical :: terminated = .false.
    end type branch_result_t

contains

    subroutine lower_program_to_liric_exe(arena, root_index, output_path, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: return_value

        call validate_program(arena, root_index, error_msg)
        if (len_trim(error_msg) > 0) return

        call liric_session_create(context%session, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. prepare_liric_i32_print(context%session, &
                                          context%i32_print_format_id, &
                                          error_msg)) then
            call context%session%destroy()
            return
        end if

        if (.not. context%session%begin_i32_main(error_msg)) then
            call context%session%destroy()
            return
        end if

        call lower_program_return(arena, root_index, context, return_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) then
            call context%session%destroy()
            return
        end if

        if (.not. context%current_block_terminated .and. &
            .not. context%session%emit_ret_i32_operand(return_value, &
                                                       error_msg)) then
            call context%session%destroy()
            return
        end if

        if (.not. context%session%finish_and_emit_exe(output_path, error_msg)) then
            call context%session%destroy()
            return
        end if

        call context%session%destroy()
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

    subroutine lower_program_return(arena, root_index, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        value = context%session%i32_immediate(0_c_int64_t)
        context%current_block_terminated = .false.
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                call lower_statement(arena, program%body_indices(i), context, &
                                     value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%current_block_terminated) return
            end do
        class default
            error_msg = 'direct LIRIC session MVP only supports a program node'
        end select
    end subroutine lower_program_return

    recursive subroutine lower_statement(arena, node_index, context, value, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        context%current_block_terminated = .false.
        value = context%session%i32_immediate(0_c_int64_t)
        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'program body index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (declaration_node)
            call lower_declaration(node, context, error_msg)
        type is (assignment_node)
            call lower_assignment(arena, node, context, error_msg)
        type is (print_statement_node)
            call lower_print(arena, node, context, error_msg)
        type is (stop_node)
            call lower_stop(arena, node, context, value, error_msg)
            if (len_trim(error_msg) == 0) then
                if (.not. context%session%emit_ret_i32_operand(value, &
                                                               error_msg)) return
                context%current_block_terminated = .true.
            end if
        type is (if_node)
            call lower_if(arena, node, context, value, error_msg)
        type is (do_loop_node)
            call lower_do_loop(arena, node, context, value, error_msg)
        class default
            error_msg = 'direct LIRIC session MVP supports declarations, assignments, PRINT, DO, IF, STOP'
        end select
    end subroutine lower_statement

    include 'session_program_lowering_control.inc'
    include 'session_program_lowering_loops.inc'

    subroutine lower_statement_list(arena, node_indices, context, value, &
                                    terminated, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        logical, intent(out) :: terminated
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        terminated = .false.
        value = context%session%i32_immediate(0_c_int64_t)
        call set_empty(error_msg)

        do i = 1, size(node_indices)
            call lower_statement(arena, node_indices(i), context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%current_block_terminated) then
                terminated = .true.
                return
            end if
        end do
    end subroutine lower_statement_list

    subroutine lower_declaration(node, context, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call require_integer_declaration(node, error_msg)
        if (len_trim(error_msg) > 0) return

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_i32_symbol(context, node%var_names(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_i32_symbol(context, node%var_name, error_msg)
        else
            error_msg = 'integer declaration did not expose a variable name'
        end if
    end subroutine lower_declaration

    subroutine require_integer_declaration(node, error_msg)
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        if (trim(node%type_name) /= 'integer') then
            error_msg = 'direct LIRIC session MVP only supports integer declarations'
        end if
    end subroutine require_integer_declaration

    subroutine define_i32_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate integer declaration: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many integer symbols for direct LIRIC session MVP'
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value = context%session%i32_immediate(0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i32_symbol

    subroutine lower_assignment(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: name
        integer :: symbol_index

        call identifier_name(arena, node%target_index, name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'assignment target was not declared: '//trim(name)
            return
        end if

        call lower_i32_expression(arena, node%value_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(symbol_index)%value = value
        call set_empty(error_msg)
    end subroutine lower_assignment

    subroutine lower_stop(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(stop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (node%stop_code_index <= 0) then
            value = context%session%i32_immediate(0_c_int64_t)
            call set_empty(error_msg)
        else
            call lower_i32_expression(arena, node%stop_code_index, context, &
                                      value, error_msg)
        end if
    end subroutine lower_stop

    subroutine lower_print(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i

        if (.not. allocated(node%expression_indices)) then
            call set_empty(error_msg)
            return
        end if

        do i = 1, size(node%expression_indices)
            call lower_i32_expression(arena, node%expression_indices(i), &
                                      context, value, error_msg)
            if (len_trim(error_msg) > 0) return

            if (.not. emit_liric_print_i32(context%session, &
                                           context%i32_print_format_id, value, &
                                           error_msg)) return
        end do

        call set_empty(error_msg)
    end subroutine lower_print

    recursive subroutine lower_i32_expression(arena, node_index, context, &
                                              value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            call parse_i32_literal(node%value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = context%session%i32_immediate(literal_value)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index <= 0) then
                error_msg = 'integer identifier was not declared: '//trim(node%name)
                return
            end if
            value = context%symbols(symbol_index)%value
            call set_empty(error_msg)
        type is (binary_op_node)
            call lower_i32_expression(arena, node%left_index, context, lhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, node%right_index, context, rhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_opcode(node%operator, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. context%session%emit_i32_binary(opcode, lhs, rhs, value, &
                                                      error_msg)) return
        class default
            error_msg = 'direct LIRIC session MVP only supports integer expressions'
        end select
    end subroutine lower_i32_expression

    recursive subroutine lower_i1_condition(arena, node_index, context, &
                                            value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer(c_int) :: pred

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'condition index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (binary_op_node)
            call lower_i32_expression(arena, node%left_index, context, lhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, node%right_index, context, rhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_compare_predicate(node%operator, pred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, pred, lhs, rhs, &
                                          value, error_msg)) return
        class default
            error_msg = 'direct LIRIC session IF requires an integer comparison'
        end select
    end subroutine lower_i1_condition

    subroutine identifier_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'identifier index does not reference an AST node'
            call set_empty(name)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            name = node%name
            call set_empty(error_msg)
        class default
            error_msg = 'expected identifier assignment target'
            call set_empty(name)
        end select
    end subroutine identifier_name

    integer function find_symbol(context, name) result(index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        index = 0
        do i = 1, context%symbol_count
            if (trim(context%symbols(i)%name) == trim(name)) then
                index = i
                return
            end if
        end do
    end function find_symbol

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module session_program_lowering
