module session_program_lowering
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, &
                                           c_int64_t
    use fortfront, only: assignment_node, ast_arena_t, binary_op_node, &
                         call_or_subscript_node, declaration_node, do_loop_node, &
                         function_def_node, identifier_node, if_node, &
                         literal_node, parameter_declaration_node, &
                         print_statement_node, program_node, stop_node
    use liric_session_bindings, only: liric_session_t, liric_session_create, &
                                      lr_operand_desc_t
    use liric_session_control_bindings, only: create_liric_block, &
                                              emit_liric_br, &
                                              emit_liric_condbr, &
                                              emit_liric_i32_icmp, &
                                              emit_liric_i32_phi, &
                                              set_liric_block
    use liric_session_io_bindings, only: emit_liric_f64_binary, &
                                         emit_liric_print_f64, &
                                         emit_liric_print_i32, &
                                         emit_liric_print_string, &
                                         liric_f64_immediate, &
                                         prepare_liric_print_runtime
    use session_lowering_ops, only: integer_compare_predicate, &
                                    integer_opcode, parse_i32_literal
    implicit none
    private

    public :: lower_program_to_liric_exe

    integer, parameter :: MAX_SYMBOLS = 64
    integer, parameter :: VALUE_I32 = 1
    integer, parameter :: VALUE_F64 = 2

    type :: symbol_t
        character(len=64) :: name = ''
        integer :: value_kind = VALUE_I32
        type(lr_operand_desc_t) :: value
    end type symbol_t

    type :: lowering_context_t
        type(liric_session_t) :: session
        type(symbol_t) :: symbols(MAX_SYMBOLS)
        integer :: symbol_count = 0
        integer(c_int32_t) :: current_block_id = 0_c_int32_t
        integer(c_int32_t) :: i32_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: f64_print_format_id = -1_c_int32_t
        integer(c_int32_t) :: str_print_format_id = -1_c_int32_t
        integer :: string_literal_count = 0
        logical :: in_internal_function = .false.
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

        if (.not. prepare_liric_print_runtime(context%session, &
                                              context%i32_print_format_id, &
                                              context%f64_print_format_id, &
                                              context%str_print_format_id, &
                                              error_msg)) then
            call context%session%destroy()
            return
        end if

        call lower_internal_functions(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
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
                if (is_function_def(arena, program%body_indices(i))) cycle
                call lower_statement(arena, program%body_indices(i), context, &
                                     value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%current_block_terminated) return
            end do
        class default
            error_msg = 'direct LIRIC session MVP only supports a program node'
        end select
    end subroutine lower_program_return

    subroutine lower_internal_functions(arena, root_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (function_def_node)
                    call lower_i32_function(arena, node, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end do
        end select
    end subroutine lower_internal_functions

    logical function is_function_def(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_function_def = .false.
        if (.not. arena%has_node_at(node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (function_def_node)
            is_function_def = .true.
        end select
    end function is_function_def

    subroutine lower_i32_function(arena, node, parent_context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        logical :: terminated

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        if (allocated(node%return_type) .and. trim(node%return_type) /= 'integer') then
            error_msg = 'direct LIRIC session MVP only supports integer functions'
            return
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%f64_print_format_id = parent_context%f64_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%in_internal_function = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (.not. context%session%begin_i32_function(node%name, param_count, &
                                                     error_msg)) return

        call define_i32_symbol(context, node%name, error_msg)
        if (len_trim(error_msg) > 0) return
        call define_i32_parameters(arena, node, context, error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        value = context%symbols(find_symbol(context, node%name))%value
        if (.not. context%current_block_terminated) then
            if (.not. context%session%emit_ret_i32_operand(value, error_msg)) return
        end if
        if (.not. context%session%finish_function(error_msg)) return

        parent_context%session = context%session
        call set_empty(error_msg)
    end subroutine lower_i32_function

    subroutine define_i32_parameters(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%param_indices)) return
        do i = 1, size(node%param_indices)
            call parameter_name(arena, node%param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            call define_i32_parameter_symbol(context, name, i - 1, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine define_i32_parameters

    subroutine parameter_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'parameter index does not reference an AST node'
            call set_empty(name)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            name = node%name
            call set_empty(error_msg)
        type is (identifier_node)
            name = node%name
            call set_empty(error_msg)
        class default
            error_msg = 'direct LIRIC session function parameter needs a name'
            call set_empty(name)
        end select
    end subroutine parameter_name

    subroutine define_i32_parameter_symbol(context, name, param_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate function parameter: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many scalar symbols for direct LIRIC session MVP'
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = context%session%i32_param(param_index)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i32_parameter_symbol

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

        integer :: value_kind

        call declaration_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_symbol(context, node%var_names(i), value_kind, &
                                   error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_symbol(context, node%var_name, value_kind, error_msg)
        else
            error_msg = 'scalar declaration did not expose a variable name'
        end if
    end subroutine lower_declaration

    subroutine declaration_value_kind(node, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        select case (trim(node%type_name))
        case ('integer')
            value_kind = VALUE_I32
        case ('real')
            value_kind = VALUE_F64
        case default
            error_msg = 'direct LIRIC session MVP only supports integer and real declarations'
        end select
    end subroutine declaration_value_kind

    subroutine define_symbol(context, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        if (value_kind == VALUE_I32) then
            call define_i32_symbol(context, name, error_msg)
        else if (value_kind == VALUE_F64) then
            call define_f64_symbol(context, name, error_msg)
        else
            error_msg = 'unknown scalar value kind for direct LIRIC session'
        end if
    end subroutine define_symbol

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
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = context%session%i32_immediate(0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i32_symbol

    subroutine define_f64_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate real declaration: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many scalar symbols for direct LIRIC session MVP'
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_F64
        context%symbols(index)%value = liric_f64_immediate(context%session, &
                                                           0.0_c_double)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_f64_symbol

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

        if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
            call lower_f64_expression(arena, node%value_index, context, value, &
                                      error_msg)
        else
            call lower_i32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        end if
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
            call lower_print_expression(arena, node%expression_indices(i), &
                                        context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do

        call set_empty(error_msg)
    end subroutine lower_print

    subroutine lower_print_expression(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        real(c_double) :: real_value
        integer :: symbol_index

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0 .and. &
                context%symbols(symbol_index)%value_kind == VALUE_F64) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               context%f64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
        type is (binary_op_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               context%f64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
        type is (literal_node)
            if (is_character_literal(node)) then
                call strip_literal_quotes(node%value, character_value)
                context%string_literal_count = context%string_literal_count + 1
                write (string_name, '(A,I0)') '.ffc.str.', &
                    context%string_literal_count
                if (.not. emit_liric_print_string(context%session, &
                                                  context%str_print_format_id, &
                                                  trim(string_name), &
                                                  character_value, &
                                                  error_msg)) return
                return
            end if
            if (is_logical_literal(node)) then
                value = context%session%i32_immediate(logical_i32_value(node%value))
                if (.not. emit_liric_print_i32(context%session, &
                                               context%i32_print_format_id, &
                                               value, error_msg)) return
                return
            end if
            if (is_real_literal(node)) then
                call parse_f64_literal(node%value, real_value, error_msg)
                if (len_trim(error_msg) > 0) return
                value = liric_f64_immediate(context%session, real_value)
                if (.not. emit_liric_print_f64(context%session, &
                                               context%f64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
        end select

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32(context%session, &
                                       context%i32_print_format_id, value, &
                                       error_msg)) return
    end subroutine lower_print_expression

    recursive subroutine lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        real(c_double) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'real expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            call parse_f64_literal(node%value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = liric_f64_immediate(context%session, literal_value)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index <= 0) then
                error_msg = 'real identifier was not declared: '//trim(node%name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_F64) then
                error_msg = 'real expression used non-real identifier: '// &
                            trim(node%name)
                return
            end if
            value = context%symbols(symbol_index)%value
            call set_empty(error_msg)
        type is (binary_op_node)
            call lower_f64_expression(arena, node%left_index, context, lhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, node%right_index, context, rhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call real_opcode(node%operator, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_binary(context%session, opcode, lhs, rhs, &
                                            value, error_msg)) return
        class default
            error_msg = 'direct LIRIC session MVP only supports real expressions'
        end select
    end subroutine lower_f64_expression

    recursive logical function is_f64_expression(arena, node_index, context) &
        result(is_f64)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_f64 = .false.
        if (.not. arena%has_node_at(node_index)) return

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            is_f64 = is_real_literal(node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            is_f64 = symbol_index > 0 .and. &
                     context%symbols(symbol_index)%value_kind == VALUE_F64
        type is (binary_op_node)
            is_f64 = is_f64_expression(arena, node%left_index, context) .or. &
                     is_f64_expression(arena, node%right_index, context)
        end select
    end function is_f64_expression

    subroutine real_opcode(source_op, opcode, error_msg)
        character(len=*), intent(in) :: source_op
        integer, intent(out) :: opcode
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            opcode = 18
        case ('-')
            opcode = 19
        case ('*')
            opcode = 20
        case ('/')
            opcode = 21
        case default
            error_msg = 'direct LIRIC session MVP does not support real operator: '// &
                        trim(source_op)
        end select
    end subroutine real_opcode

    logical function is_real_literal(node)
        type(literal_node), intent(in) :: node

        is_real_literal = .false.
        if (allocated(node%literal_type)) then
            is_real_literal = trim(node%literal_type) == 'real'
        end if
        is_real_literal = is_real_literal .or. index(node%value, '.') > 0 .or. &
                          index(node%value, 'e') > 0 .or. &
                          index(node%value, 'E') > 0
    end function is_real_literal

    logical function is_character_literal(node)
        type(literal_node), intent(in) :: node

        is_character_literal = .false.
        if (allocated(node%literal_type)) then
            is_character_literal = trim(node%literal_type) == 'character'
        end if
        is_character_literal = is_character_literal .or. &
                               starts_with_quote(node%value)
    end function is_character_literal

    logical function is_logical_literal(node)
        type(literal_node), intent(in) :: node
        character(len=:), allocatable :: value

        value = trim(node%value)
        is_logical_literal = value == '.true.' .or. value == '.false.'
        if (allocated(node%literal_type)) then
            is_logical_literal = is_logical_literal .or. &
                                 trim(node%literal_type) == 'logical'
        end if
    end function is_logical_literal

    logical function starts_with_quote(text)
        character(len=*), intent(in) :: text

        starts_with_quote = len_trim(text) >= 2 .and. &
                            (text(1:1) == '"' .or. text(1:1) == "'")
    end function starts_with_quote

    subroutine strip_literal_quotes(text, value)
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: value
        integer :: text_len

        text_len = len_trim(text)
        if (text_len >= 2 .and. &
            ((text(1:1) == '"' .and. text(text_len:text_len) == '"') .or. &
             (text(1:1) == "'" .and. text(text_len:text_len) == "'"))) then
            value = text(2:text_len - 1)
        else
            value = trim(text)
        end if
    end subroutine strip_literal_quotes

    integer(c_int64_t) function logical_i32_value(text) result(value)
        character(len=*), intent(in) :: text

        if (trim(text) == '.true.') then
            value = 1_c_int64_t
        else
            value = 0_c_int64_t
        end if
    end function logical_i32_value

    subroutine parse_f64_literal(text, value, error_msg)
        character(len=*), intent(in) :: text
        real(c_double), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_stat

        read (text, *, iostat=io_stat) value
        if (io_stat == 0) then
            call set_empty(error_msg)
        else
            error_msg = 'invalid real literal for direct LIRIC session: '// &
                        trim(text)
        end if
    end subroutine parse_f64_literal

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
        type is (call_or_subscript_node)
            call lower_i32_call(arena, node, context, value, error_msg)
        class default
            error_msg = 'direct LIRIC session MVP only supports integer expressions'
        end select
    end subroutine lower_i32_expression

    subroutine lower_i32_call(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer :: i

        if (node%is_array_access) then
            error_msg = 'direct LIRIC session MVP does not support array access'
            return
        end if
        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function call requires a name'
            return
        end if

        if (allocated(node%arg_indices)) then
            allocate (args(size(node%arg_indices)))
            do i = 1, size(node%arg_indices)
                call lower_i32_expression(arena, node%arg_indices(i), context, &
                                          args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else
            allocate (args(0))
        end if

        if (.not. context%session%emit_i32_call(node%name, args, value, &
                                                error_msg)) return
    end subroutine lower_i32_call

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
        type is (call_or_subscript_node)
            if (node%is_array_access .or. allocated(node%arg_indices)) then
                error_msg = 'expected scalar assignment target'
                call set_empty(name)
                return
            end if
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
