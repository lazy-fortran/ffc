module session_program_lowering
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, &
                                                                              c_int64_t
    use fortfront, only: assignment_node, ast_arena_t, binary_op_node, &
                         call_or_subscript_node, declaration_node, do_loop_node, &
                         cycle_node, exit_node, function_def_node, &
                         identifier_node, if_node, literal_node, &
                         parameter_declaration_node, &
                         print_statement_node, program_node, module_node, stop_node, &
                         subroutine_def_node, get_subroutine_call_arg_indices, &
                         get_subroutine_call_name, is_subroutine_call_statement
    use liric_session_bindings, only: liric_session_t, liric_session_create, &
                                      lr_operand_desc_t, LR_OP_ADD, LR_OP_SUB
    use liric_session_control_bindings, only: create_liric_block, &
                                              emit_liric_br, &
                                              emit_liric_condbr, &
                                              emit_liric_f64_fcmp, &
                                              emit_liric_i32_icmp, &
                                              emit_liric_i32_phi, &
                                              emit_liric_phi, &
                                              LR_FCMP_OGE, &
                                              LR_FCMP_OLE, &
                                              LR_CMP_SGE, &
                                              LR_CMP_SLE, &
                                              LR_CMP_NE, &
                                              set_liric_block
    use liric_session_io_bindings, only: emit_liric_f64_binary, &
                                         emit_liric_i32_to_f64, &
                                         emit_liric_print_f64, &
                                         emit_liric_print_i32, &
                                         emit_liric_print_string_operand, &
                                         emit_liric_print_string, &
                                         liric_f64_immediate, &
                                         LR_OP_FSUB, &
                                         materialize_liric_string, &
                                         prepare_liric_print_runtime
    use liric_session_procedure_bindings, only: begin_liric_f64_function, &
                                                emit_liric_f64_alloca, &
                                                emit_liric_f64_call, &
                                                emit_liric_f64_load, &
                                                emit_liric_f64_store
    use session_lowering_ops, only: integer_compare_predicate, &
                                    integer_opcode, parse_i32_literal
    implicit none
    private

    public :: lower_program_to_liric_exe
    public :: lower_program_to_liric_object

    integer, parameter :: MAX_SYMBOLS = 64
    integer, parameter :: MAX_PROCEDURES = 32
    integer, parameter :: VALUE_I32 = 1
    integer, parameter :: VALUE_F64 = 2
    integer, parameter :: VALUE_LOGICAL = 3
    integer, parameter :: VALUE_CHARACTER = 4
    integer, parameter :: I32_INTRINSIC_NONE = 0
    integer, parameter :: I32_INTRINSIC_ABS = 1
    integer, parameter :: I32_INTRINSIC_MIN = 2
    integer, parameter :: I32_INTRINSIC_MAX = 3
    integer, parameter :: F64_INTRINSIC_NONE = 0
    integer, parameter :: F64_INTRINSIC_ABS = 1
    integer, parameter :: F64_INTRINSIC_MIN = 2
    integer, parameter :: F64_INTRINSIC_MAX = 3
    integer, parameter :: F64_INTRINSIC_REAL = 4
    character(len=8), parameter :: I32_INTRINSIC_NAMES(3) = &
                                   [character(len=8) :: 'abs', 'min', 'max']
    integer, parameter :: I32_INTRINSIC_IDS(3) = &
                          [I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, I32_INTRINSIC_MAX]
    character(len=8), parameter :: F64_INTRINSIC_NAMES(4) = &
                                   [character(len=8) :: 'abs', 'min', 'max', 'real']
    integer, parameter :: F64_INTRINSIC_IDS(4) = &
                          [F64_INTRINSIC_ABS, F64_INTRINSIC_MIN, F64_INTRINSIC_MAX, &
                           F64_INTRINSIC_REAL]

    type :: symbol_t
        character(len=64) :: name = ''
        integer :: value_kind = VALUE_I32
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: address
        logical :: is_parameter = .false.
        logical :: is_reference = .false.
        logical :: has_address = .false.
        integer :: character_length = 0
        logical :: has_character_value = .false.
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
        character(len=64) :: function_names(MAX_PROCEDURES)
        integer :: function_value_kinds(MAX_PROCEDURES) = VALUE_I32
        integer :: function_count = 0
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

        call lower_program_to_liric_path(arena, root_index, output_path, &
                                         .true., error_msg)
    end subroutine lower_program_to_liric_exe

    subroutine lower_program_to_liric_object(arena, root_index, output_path, &
                                             error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        character(len=:), allocatable, intent(out) :: error_msg

        call lower_program_to_liric_path(arena, root_index, output_path, &
                                         .false., error_msg)
    end subroutine lower_program_to_liric_object

    subroutine lower_program_to_liric_path(arena, root_index, output_path, &
                                           emit_executable, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        character(len=*), intent(in) :: output_path
        logical, intent(in) :: emit_executable
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

        call collect_internal_function_names(arena, root_index, context, &
                                             error_msg)
        if (len_trim(error_msg) > 0) then
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

        if (.not. context%current_block_terminated) then
            if (.not. context%session%emit_ret_i32_operand(return_value, &
                                                           error_msg)) then
                call context%session%destroy()
                return
            end if
        end if

        if (emit_executable) then
            if (.not. context%session%finish_and_emit_exe(output_path, &
                                                          error_msg)) then
                call context%session%destroy()
                return
            end if
        else
            if (.not. context%session%finish_and_emit_object(output_path, &
                                                             error_msg)) then
                call context%session%destroy()
                return
            end if
        end if

        call context%session%destroy()
        call set_empty(error_msg)
    end subroutine lower_program_to_liric_path

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
        type is (module_node)
            call unsupported_feature_error('module program unit', &
                                           program%line, program%column, &
                                           'direct LIRIC session only lowers '// &
                                           'top-level programs', error_msg)
            return
        class default
            call unsupported_feature_error('program unit', &
                                           arena%get_node_line(root_index), &
                                           arena%get_node_column(root_index), &
                                           'direct LIRIC session only lowers '// &
                                           'top-level programs', error_msg)
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
                if (is_internal_procedure_entry(arena, program%body_indices(i))) cycle
                call lower_statement(arena, program%body_indices(i), context, &
                                     value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%current_block_terminated) return
            end do
        class default
            error_msg = 'direct LIRIC session MVP only supports a program node'
        end select
    end subroutine lower_program_return

    include 'session_program_lowering_functions.inc'

    recursive subroutine lower_statement(arena, node_index, context, value, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: node_type

        context%current_block_terminated = .false.
        value = context%session%i32_immediate(0_c_int64_t)
        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'program body index does not reference an AST node'
            return
        end if

        if (is_subroutine_call_statement(arena, node_index)) then
            call lower_subroutine_call(arena, node_index, context, error_msg)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            call lower_parameter_declaration(node, context, error_msg)
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
        type is (exit_node)
            call unsupported_feature_error('exit statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session does not '// &
                                           'support early loop exits', error_msg)
        type is (cycle_node)
            call unsupported_feature_error('cycle statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session does not '// &
                                           'support early loop backedges', &
                                           error_msg)
        type is (if_node)
            call lower_if(arena, node, context, value, error_msg)
        type is (do_loop_node)
            call lower_do_loop(arena, node, context, value, error_msg)
        class default
            node_type = 'unknown'
            if (allocated(arena%entries(node_index)%node_type)) then
                node_type = arena%entries(node_index)%node_type
            end if
            error_msg = 'direct LIRIC session MVP does not support statement node: '// &
                        trim(node_type)
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

        if (node%is_array) then
            call unsupported_feature_error('array declaration', &
                                           node%line, node%column, &
                                           'fixed-size arrays are not supported '// &
                                           'by direct LIRIC session', error_msg)
            return
        end if

        call declaration_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_declared_symbol(context, node, node%var_names(i), &
                                            value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_declared_symbol(context, node, node%var_name, &
                                        value_kind, error_msg)
        else
            error_msg = 'scalar declaration did not expose a variable name'
        end if
    end subroutine lower_declaration

    subroutine define_declared_symbol(context, node, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: existing_index

        existing_index = find_symbol(context, name)
        if (existing_index > 0) then
            if (context%symbols(existing_index)%is_parameter .and. &
                value_kind == VALUE_CHARACTER) then
                call unsupported_feature_error( &
                    'character parameter declaration', node%line, node%column, &
                    'scalar character parameters are not supported '// &
                    'by direct LIRIC session', error_msg)
                return
            end if
        end if

        if (value_kind == VALUE_CHARACTER) then
            call define_declared_character_symbol(context, node, name, error_msg)
        else
            call define_symbol(context, name, value_kind, error_msg)
        end if
    end subroutine define_declared_symbol

    subroutine lower_parameter_declaration(node, context, error_msg)
        type(parameter_declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        integer :: value_kind

        if (.not. allocated(node%name)) then
            error_msg = 'parameter declaration did not expose a name'
            return
        end if
        if (node%is_array) then
            call unsupported_feature_error('array parameter declaration', &
                                           node%line, node%column, &
                                           'array parameters are not supported '// &
                                           'by direct LIRIC session', error_msg)
            return
        end if
        if (allocated(node%type_name)) then
            call type_name_value_kind(node%type_name, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            if (value_kind == VALUE_CHARACTER) then
                call unsupported_feature_error( &
                    'character parameter declaration', &
                    node%line, node%column, &
                    'scalar character parameters are not supported '// &
                    'by direct LIRIC session', error_msg)
                return
            end if
        else
            value_kind = VALUE_I32
        end if

        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'parameter declaration did not match a dummy argument: '// &
                        trim(node%name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_parameter) then
            error_msg = 'parameter declaration did not match a dummy argument: '// &
                        trim(node%name)
            return
        end if

        call update_parameter_symbol(context, symbol_index, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine lower_parameter_declaration

    subroutine type_name_value_kind(type_name, value_kind, error_msg)
        character(len=*), intent(in) :: type_name
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (is_character_type_name(type_name)) then
            value_kind = VALUE_CHARACTER
            return
        end if
        select case (trim(type_name))
        case ('integer')
            value_kind = VALUE_I32
        case ('real')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case default
            error_msg = 'direct LIRIC session MVP only supports integer, real, '// &
                        'and logical scalar types'
        end select
    end subroutine type_name_value_kind

    subroutine declaration_value_kind(node, value_kind, error_msg)
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) return
        call type_name_value_kind(node%type_name, value_kind, error_msg)
    end subroutine declaration_value_kind

    subroutine define_symbol(context, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: existing_index

        existing_index = find_symbol(context, name)
        if (existing_index > 0) then
            if (context%symbols(existing_index)%is_parameter) then
                call update_parameter_symbol(context, existing_index, value_kind, &
                                             error_msg)
                return
            end if
        end if

        if (value_kind == VALUE_I32) then
            call define_i32_symbol(context, name, error_msg)
        else if (value_kind == VALUE_F64) then
            call define_f64_symbol(context, name, error_msg)
        else if (value_kind == VALUE_LOGICAL) then
            call define_logical_symbol(context, name, error_msg)
        else if (value_kind == VALUE_CHARACTER) then
            call define_character_symbol(context, name, 1, error_msg)
        else
            error_msg = 'unknown scalar value kind for direct LIRIC session'
        end if
    end subroutine define_symbol

    subroutine update_parameter_symbol(context, index, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: index
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        if (index <= 0 .or. index > context%symbol_count) then
            error_msg = 'parameter index is outside the symbol table'
            return
        end if
        if (.not. context%symbols(index)%is_parameter) then
            error_msg = 'symbol is not a parameter: '//trim(context%symbols(index)%name)
            return
        end if

        context%symbols(index)%value_kind = value_kind
        if (value_kind == VALUE_F64) then
            context%symbols(index)%value = liric_f64_immediate(context%session, &
                                                               0.0_c_double)
        else if (value_kind == VALUE_LOGICAL .or. value_kind == VALUE_I32) then
            context%symbols(index)%value = context%session%i32_immediate(0_c_int64_t)
        else
            error_msg = 'unsupported parameter declaration value kind'
            return
        end if
        call set_empty(error_msg)
    end subroutine update_parameter_symbol

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

    subroutine define_logical_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate logical declaration: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many scalar symbols for direct LIRIC session MVP'
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_LOGICAL
        context%symbols(index)%value = context%session%i32_immediate(0_c_int64_t)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_logical_symbol

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
        else if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
            call lower_logical_expression(arena, node%value_index, context, &
                                          value, error_msg)
        else if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
            call lower_character_assignment(arena, node, context, symbol_index, &
                                            error_msg)
            return
        else
            call lower_i32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        context%symbols(symbol_index)%value = value
        if (context%symbols(symbol_index)%has_address .and. &
            context%symbols(symbol_index)%is_reference) then
            call store_reference_value(context, symbol_index, value, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call set_empty(error_msg)
    end subroutine lower_assignment

    include 'session_program_lowering_arguments.inc'

    include 'session_program_lowering_character.inc'

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

    include 'session_program_lowering_values.inc'

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
            if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
                error_msg = 'integer expression used non-integer identifier: '// &
                            trim(node%name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. context%session%emit_i32_load( &
                    context%symbols(symbol_index)%address, value, &
                    error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
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
        integer, allocatable :: copyback_indices(:)
        integer :: intrinsic_id

        if (node%is_array_access) then
            error_msg = 'direct LIRIC session MVP does not support array access'
            return
        end if
        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function call requires a name'
            return
        end if

        intrinsic_id = i32_intrinsic_id(node%name)
        if (.not. is_contained_i32_function(context, node%name)) then
            if (intrinsic_id /= I32_INTRINSIC_NONE) then
                call lower_i32_intrinsic_call(arena, node, intrinsic_id, &
                                              context, value, error_msg)
                return
            end if
            if (node%is_intrinsic) then
                call unsupported_intrinsic_error(node, error_msg)
                return
            end if
        end if

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_I32, args, copyback_indices, &
                                        error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. context%session%emit_i32_call(node%name, args, value, &
                                                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_i32_call

    include 'session_program_lowering_intrinsics.inc'

    subroutine lower_subroutine_call(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer, allocatable :: arg_indices(:)
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        call get_subroutine_call_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return

        call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
                                    args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. context%session%emit_void_call(name, args, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_subroutine_call

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
        type is (literal_node)
            call lower_logical_expression(arena, node_index, context, lhs, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = context%session%i32_immediate(0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                                          rhs, value, error_msg)) return
        type is (identifier_node)
            call lower_logical_expression(arena, node_index, context, lhs, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = context%session%i32_immediate(0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                                          rhs, value, error_msg)) return
        class default
            error_msg = 'direct LIRIC session IF requires an integer '// &
                        'comparison or logical expression'
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

    include 'session_program_lowering_text.inc'

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

    subroutine unsupported_feature_error(feature, line, column, limitation, &
                                         error_msg)
        character(len=*), intent(in) :: feature
        integer, intent(in) :: line
        integer, intent(in) :: column
        character(len=*), intent(in) :: limitation
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: location

        if (line > 0 .and. column > 0) then
            write (location, '(" at line ",I0,", column ",I0)') line, column
            error_msg = 'unsupported '//trim(feature)//trim(location)//': '// &
                        trim(limitation)
        else
            error_msg = 'unsupported '//trim(feature)//': '//trim(limitation)
        end if
    end subroutine unsupported_feature_error

end module session_program_lowering
