module session_program_lowering
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, &
                                                                              c_int64_t
    use ast_base, only: LITERAL_INTEGER
    use ast_nodes_bounds, only: array_slice_node, range_expression_node
    use ast_nodes_core, only: component_access_node
    use ast_nodes_data, only: derived_type_node
    use ast_nodes_misc, only: use_statement_node
    use fortfront, only: assignment_node, ast_arena_t, binary_op_node, &
                         call_or_subscript_node, case_block_node, &
                         case_default_node, declaration_node, do_loop_node, &
                         do_while_node, cycle_node, exit_node, function_def_node, &
                         identifier_node, if_node, literal_node, &
                         parameter_declaration_node, &
                         print_statement_node, program_node, read_statement_node, &
                         module_node, &
                         return_node, select_case_node, stop_node, &
                         subroutine_def_node, write_statement_node, &
                         allocate_statement_node, deallocate_statement_node, &
                         get_subroutine_call_arg_indices, &
                         get_subroutine_call_name, is_subroutine_call_statement, &
                         is_declaration_node, is_module_node, is_program_node
use liric_session_bindings, only: destroy, begin_i32_main, &
                                       begin_i32_function, begin_void_subroutine, &
                                       begin_ptr_function, &
                                      emit_ret_i32_operand, emit_ret_void, &
                                      finish_function, finish_and_emit_exe, &
                                      finish_and_emit_object, emit_void_call, &
                                      emit_i32_call, liric_session_create, &
                                      i32_immediate, i32_vreg, lr_operand_desc_t, &
                                      LR_OP_ADD, LR_OP_SREM, LR_OP_SUB
    use liric_session_memory_bindings, only: reserve_i32_vreg, i64_immediate, &
                                              emit_i32_binary, emit_i32_binary_into, &
                                              emit_i32_copy_to, emit_i32_alloca, &
                                              emit_i32_load, emit_i32_store, &
                                              emit_i64_load, emit_ptr_load, &
                                              emit_i64_store, &
                                              emit_i64_binary, emit_i64_alloca, &
                                              emit_alloca_bytes, emit_malloc, &
                                              emit_ptr_store, &
                                               emit_memcpy, emit_i64_load_at, &
                                               emit_i64_store_at, &
                                               emit_i32_array_alloca, &
                                              emit_i32_array_element_addr, &
                                              emit_ptr_offset, ptr_param
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
                                              LR_CMP_EQ, &
                                              set_liric_block
use liric_session_format_bindings, only: LR_OP_FSUB, &
                                            prepare_liric_print_runtime
    use liric_session_io_bindings, only: emit_liric_f64_binary, &
                                          emit_liric_i32_to_f64, &
                                          emit_liric_print_f64, &
                                          emit_liric_print_f64_value, &
                                          emit_liric_print_i32, &
                                          emit_liric_print_i32_value, &
                                          emit_liric_print_newline, &
                                          emit_liric_print_space, &
                                          emit_liric_print_string_operand, &
                                          emit_liric_print_string_operand_value, &
                                          emit_liric_print_string, &
                                          emit_liric_print_string_value, &
                                          liric_f64_immediate, &
                                          materialize_liric_string
    use liric_session_procedure_bindings, only: begin_liric_f64_function, &
                                                emit_liric_f64_alloca, &
                                                emit_liric_f64_call, &
                                                emit_liric_f64_load, &
                                                emit_liric_f64_store
    use session_lowering_ops, only: integer_compare_predicate, &
                                    integer_opcode, parse_i32_literal
  use ffc_strings, only: set_empty
    use ffc_fortfront_queries, only: node_exists, get_node_line, &
                                      get_node_column, get_node_type_at
   use session_program_lowering_types, only: lowering_context_t, &
                                                branch_result_t, symbol_t, &
                                                derived_type_info_t, &
                                                module_exports_t, &
                                                VALUE_I32, VALUE_F64, &
                                                 VALUE_LOGICAL, VALUE_CHARACTER, &
                                                 VALUE_DERIVED, &
                                                 VALUE_DEFERRED_CHARACTER_RESULT, &
                                                 I32_INTRINSIC_NONE, &
                                                I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, &
                                                I32_INTRINSIC_MAX, I32_INTRINSIC_MOD, &
                                                F64_INTRINSIC_NONE, F64_INTRINSIC_ABS, &
                                                F64_INTRINSIC_MIN, F64_INTRINSIC_MAX, &
                                                F64_INTRINSIC_REAL, I32_INTRINSIC_NAMES, &
                                                I32_INTRINSIC_IDS, F64_INTRINSIC_NAMES, &
                                                F64_INTRINSIC_IDS
    implicit none
    private
    public :: lower_program_to_liric_exe
    public :: lower_program_to_liric_object
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
        allocate(context%symbols(16))
        allocate(context%derived_types(8))
        allocate(context%module_exports(4))
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        context%function_value_kinds = VALUE_I32
        context%arena = arena
        if (.not. prepare_liric_print_runtime(context%session, &
                                              context%i32_print_format_id, &
                                              context%f64_print_format_id, &
                                              context%str_print_format_id, &
                                              error_msg)) then
            call destroy(context%session)
            return
        end if
        call collect_derived_type_definitions(arena, root_index, context, &
                                              error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call collect_module_exports(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call collect_internal_function_names(arena, root_index, context, &
                                             error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        call lower_internal_functions(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        if (.not. begin_i32_main(context%session, error_msg)) then
            call destroy(context%session)
            return
        end if
        call lower_program_return(arena, root_index, context, return_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) then
            call destroy(context%session)
            return
        end if
        if (.not. context%current_block_terminated) then
            if (.not. emit_ret_i32_operand(context%session, return_value, &
                                                           error_msg)) then
                call destroy(context%session)
                return
            end if
        end if
        if (emit_executable) then
            if (.not. finish_and_emit_exe(context%session, output_path, &
                                                          error_msg)) then
                call destroy(context%session)
                return
            end if
        else
            if (.not. finish_and_emit_object(context%session, output_path, &
                                                             error_msg)) then
                call destroy(context%session)
                return
            end if
        end if
        call destroy(context%session)
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
        if (.not. node_exists(arena, root_index)) then
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
                                           get_node_line(arena, root_index), &
                                           get_node_column(arena, root_index), &
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
        integer :: j
        logical :: has_executable_statements
        logical :: has_nested_program
        value = i32_immediate(context%session, 0_c_int64_t)
        context%current_block_terminated = .false.
        has_executable_statements = .false.
        has_nested_program = .false.
        call set_empty(error_msg)
        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            ! Check if there's a program_node in the body (multi-unit source)
            has_nested_program = .false.
            do i = 1, size(program%body_indices)
                if (is_program_node(arena, program%body_indices(i))) then
                    has_nested_program = .true.
                    exit
                end if
            end do
            ! Process body elements
            do i = 1, size(program%body_indices)
                if (is_internal_procedure_entry(arena, program%body_indices(i))) cycle
                if (is_module_node(arena, program%body_indices(i))) then
                    ! Skip module nodes only if there's a nested program
                    if (.not. has_nested_program) then
                        call unsupported_feature_error('module definition', &
                            arena%entries(program%body_indices(i))%node%line, &
                            arena%entries(program%body_indices(i))%node%column, &
                            'direct LIRIC session only supports modules as USE '// &
                            'targets, not as top-level program units', error_msg)
                        return
                    end if
                    cycle
                end if
                call lower_statement(arena, program%body_indices(i), context, &
                                     value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%current_block_terminated) return
                has_executable_statements = .true.
            end do
        class default
            error_msg = 'ffc direct-session lowering only supports a program node'
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
        value = i32_immediate(context%session, 0_c_int64_t)
        if (.not. node_exists(arena, node_index)) then
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
        type is (read_statement_node)
            call unsupported_feature_error('read statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session does not '// &
                                           'support input runtime calls', &
                                           error_msg)
        type is (write_statement_node)
            call unsupported_feature_error('write statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session only supports '// &
                                           'minimal print output', error_msg)
        type is (allocate_statement_node)
            call unsupported_feature_error('allocate statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session does not '// &
                                           'support dynamic allocation', &
                                           error_msg)
        type is (deallocate_statement_node)
            call unsupported_feature_error('deallocate statement', node%line, &
                                           node%column, &
                                           'direct LIRIC session does not '// &
                                           'support dynamic allocation', &
                                           error_msg)
        type is (return_node)
            if (context%in_internal_subroutine) then
                if (.not. emit_ret_void(context%session, error_msg)) return
                context%current_block_terminated = .true.
            else if (context%in_internal_function) then
                call lower_function_return(node, context, error_msg)
            else
                call unsupported_feature_error('return statement', node%line, &
                                               node%column, &
                                               'direct LIRIC session does not '// &
                                               'support return at top level', &
                                               error_msg)
            end if
        type is (stop_node)
            call lower_stop(arena, node, context, value, error_msg)
            if (len_trim(error_msg) == 0) then
                if (.not. emit_ret_i32_operand(context%session, value, &
                                                               error_msg)) return
                context%current_block_terminated = .true.
            end if
        type is (exit_node)
            if (.not. context%in_counted_do) then
                call unsupported_feature_error('exit statement', node%line, &
                                               node%column, &
                                               'direct LIRIC session only '// &
                                               'supports exit inside a counted '// &
                                               'do', error_msg)
                return
            end if
            if (.not. emit_liric_br(context%session, &
                                    context%current_loop_exit_block, &
                                    error_msg)) return
            context%current_block_terminated = .true.
            context%current_block_exited_loop = .true.
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
        type is (do_while_node)
            call unsupported_feature_error('do while statement', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'literal-bound counted do loops', &
                                           error_msg)
        type is (select_case_node)
            call lower_select_case(arena, node, context, error_msg)
        type is (derived_type_node)
            call lower_derived_type_definition(node, context, error_msg)
        type is (module_node)
            ! Module nodes are skipped silently (they're USE targets, not executable)
            call set_empty(error_msg)
        type is (program_node)
            call lower_program_body(arena, node, context, value, error_msg)
        
        type is (use_statement_node)
            call import_module_derived_types(arena, node, context, error_msg)
        class default
            node_type = get_node_type_at(arena, node_index)
            if (len_trim(node_type) == 0) node_type = 'unknown'
            call unsupported_statement_node(arena, node_index, node_type, error_msg)
        end select
    end subroutine lower_statement
    subroutine unsupported_statement_node(arena, node_index, node_type, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: node_type
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: column
        integer :: line
        line = get_node_line(arena, node_index)
        column = get_node_column(arena, node_index)
        select case (trim(node_type))
        case ('implicit_statement')
            call set_empty(error_msg)
        case ('use_statement')
            call unsupported_feature_error('use statement', line, column, &
                                           'direct LIRIC session does not '// &
                                           'support module imports', error_msg)
        case ('interface_block')
            call unsupported_feature_error('interface block', line, column, &
                                           'direct LIRIC session does not '// &
                                           'support generic or explicit '// &
                                           'interfaces', error_msg)
        case default
            call unsupported_feature_error('statement node: '//trim(node_type), &
                                           line, column, &
                                           'direct LIRIC session does not '// &
                                           'support this AST node', error_msg)
        end select
    end subroutine unsupported_statement_node
    include 'session_program_lowering_control.inc'
    include 'session_program_lowering_loops.inc'
    include 'session_program_lowering_derived_types.inc'
    include 'session_program_lowering_derived_type_ops.inc'
    include 'session_program_lowering_derived_module_ops.inc'
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
        value = i32_immediate(context%session, 0_c_int64_t)
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
        integer :: array_lower_bound
        integer :: array_size
        integer :: derived_type_index
        integer :: i
        integer :: value_kind
        derived_type_index = declaration_derived_type_index(context, node)
        if (derived_type_index > 0) then
            call lower_derived_type_declaration(node, context, derived_type_index, &
                                                error_msg)
            return
        end if
        if (node%is_array) then
            call declaration_value_kind(node, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            if (value_kind /= VALUE_I32) then
                call unsupported_feature_error('array declaration', node%line, &
                                               node%column, &
                                               'ffc direct-session lowering only '// &
                                               'supports integer arrays', error_msg)
                return
            end if
            call get_array_bounds(node, context, array_lower_bound, array_size, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            if (node%is_multi_declaration .and. allocated(node%var_names)) then
                do i = 1, size(node%var_names)
                    call define_declared_array_symbol( &
                        context, node, node%var_names(i), array_lower_bound, &
                        array_size, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(node%var_name)) then
                call define_declared_array_symbol(context, node, node%var_name, &
                                                  array_lower_bound, array_size, &
                                                  error_msg)
            else
                error_msg = 'array declaration did not expose a variable name'
            end if
            return
        end if
        call declaration_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%is_parameter) then
            call lower_constant_declaration(node, context, value_kind, error_msg)
            return
        end if
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
    include 'session_program_lowering_declarations.inc'
    subroutine define_declared_symbol(context, node, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
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
        integer :: symbol_index, value_kind, character_length
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
            call type_name_value_kind(node%type_name, node%line, node%column, &
                                      value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
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
        if (value_kind == VALUE_CHARACTER) then
            call parse_character_length(node%type_name, character_length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (character_length > 0) then
                call unsupported_feature_error( &
                    'character parameter declaration', node%line, node%column, &
                    'only assumed-length character parameters are supported '// &
                    'by direct LIRIC session', error_msg)
                return
            end if
            call bind_character_parameter_symbol(context, symbol_index, error_msg)
        else
            call update_parameter_symbol(context, symbol_index, value_kind, &
                                         error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine lower_parameter_declaration
    include 'session_program_lowering_arrays.inc'
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
            context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
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
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
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
        call grow_symbols(context)
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
       call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_LOGICAL
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
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
        select type (target => arena%entries(node%target_index)%node)
        type is (call_or_subscript_node)
            if (target%is_array_access) then
                call lower_array_element_assignment(arena, node, target, context, &
                                                    value, error_msg)
                return
            end if
        type is (component_access_node)
            call lower_derived_component_assignment(arena, node, target, context, &
                                                    value, error_msg)
            return
        end select
        call identifier_name(arena, node%target_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'assignment target was not declared: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%is_array) then
            select type (target => arena%entries(node%target_index)%node)
            type is (identifier_node)
                call unsupported_feature_error('array assignment target', &
                                               target%line, target%column, &
                                               'whole-array assignment is not '// &
                                               'supported', error_msg)
            class default
                call unsupported_feature_error('array assignment target', &
                                               node%line, node%column, &
                                               'whole-array assignment is not '// &
                                               'supported', error_msg)
            end select
            return
        end if
        if (context%symbols(symbol_index)%is_derived) then
            call lower_derived_whole_assignment_diagnostic(arena, node, &
                                                           context, symbol_index, &
                                                           error_msg)
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
    include 'session_program_lowering_deferred_char.inc'
    subroutine lower_function_return(node, context, error_msg)
        type(return_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: idx
        type(lr_operand_desc_t) :: result_value
        idx = context%current_function_result_index
        if (idx <= 0) then
            error_msg = 'function return without a tracked result symbol'
            return
        end if
        select case (context%symbols(idx)%value_kind)
        case (VALUE_I32, VALUE_LOGICAL, VALUE_F64)
            result_value = context%symbols(idx)%value
            if (.not. emit_ret_i32_operand(context%session, result_value, &
                                                           error_msg)) return
            context%current_block_terminated = .true.
        case default
            call unsupported_feature_error('return from non-scalar function', &
                                           node%line, node%column, &
                                           'only integer, logical and real '// &
                                           'function returns are supported', &
                                           error_msg)
        end select
    end subroutine lower_function_return
    subroutine lower_stop(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(stop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        if (node%stop_code_index <= 0) then
            value = i32_immediate(context%session, 0_c_int64_t)
            call set_empty(error_msg)
        else
            call lower_i32_expression(arena, node%stop_code_index, context, &
                                      value, error_msg)
        end if
    end subroutine lower_stop
    include 'session_program_lowering_print_ops.inc'
    include 'session_program_lowering_print_expr.inc'
    include 'session_program_lowering_expr_lowering.inc'
    include 'session_program_lowering_literal_utils.inc'
    include 'session_program_lowering_integer.inc'
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
        if (.not. emit_void_call(context%session, name, args, error_msg)) return
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
        if (.not. node_exists(arena, node_index)) then
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
            rhs = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                                          rhs, value, error_msg)) return
        type is (identifier_node)
            call lower_logical_expression(arena, node_index, context, lhs, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = i32_immediate(context%session, 0_c_int64_t)
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
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'identifier index does not reference an AST node'
            call set_empty(name)
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            name = node%name
            call set_empty(error_msg)
        type is (call_or_subscript_node)
            if (allocated(node%arg_indices)) then
                error_msg = 'expected scalar assignment target'
                call set_empty(name)
                return
            end if
            name = node%name
            call set_empty(error_msg)
        type is (component_access_node)
            call unsupported_feature_error('derived type component assignment target', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support assigning to components', &
                                           error_msg)
            call set_empty(name)
        class default
            error_msg = 'expected identifier assignment target'
            call set_empty(name)
        end select
    end subroutine identifier_name
    subroutine grow_symbols(context)
        type(lowering_context_t), intent(inout) :: context
        type(symbol_t), allocatable :: old_symbols(:)
        integer :: new_size

        if (context%symbol_count < size(context%symbols)) return
        old_symbols = context%symbols
        new_size = 2 * size(context%symbols)
        deallocate(context%symbols)
        allocate(context%symbols(new_size))
        context%symbols(1:size(old_symbols)) = old_symbols
        deallocate(old_symbols)
    end subroutine grow_symbols

    subroutine grow_derived_types(context)
        type(lowering_context_t), intent(inout) :: context
        type(derived_type_info_t), allocatable :: old_types(:)
        integer :: new_size

        if (context%derived_type_count < size(context%derived_types)) return
        old_types = context%derived_types
        new_size = 2 * size(context%derived_types)
        deallocate(context%derived_types)
        allocate(context%derived_types(new_size))
        context%derived_types(1:size(old_types)) = old_types
        deallocate(old_types)
    end subroutine grow_derived_types

    subroutine grow_module_exports(context)
        type(lowering_context_t), intent(inout) :: context
        type(module_exports_t), allocatable :: old_exports(:)
        integer :: new_size

        if (context%module_export_count < size(context%module_exports)) return
        old_exports = context%module_exports
        new_size = 2 * size(context%module_exports)
        deallocate(context%module_exports)
        allocate(context%module_exports(new_size))
        context%module_exports(1:size(old_exports)) = old_exports
        deallocate(old_exports)
    end subroutine grow_module_exports

    subroutine grow_function_names(context)
        type(lowering_context_t), intent(inout) :: context
        character(len=64), allocatable :: old_names(:)
        integer, allocatable :: old_kinds(:)
        integer :: new_size

        if (context%function_count < size(context%function_names)) return
        old_names = context%function_names
        old_kinds = context%function_value_kinds
        new_size = 2 * size(context%function_names)
        deallocate(context%function_names)
        deallocate(context%function_value_kinds)
        allocate(context%function_names(new_size))
        allocate(context%function_value_kinds(new_size))
        context%function_names(1:size(old_names)) = old_names
        context%function_value_kinds(1:size(old_kinds)) = old_kinds
        deallocate(old_names)
        deallocate(old_kinds)
    end subroutine grow_function_names

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
    include 'session_program_lowering_select.inc'
    include 'session_program_lowering_diagnostics.inc'
end module session_program_lowering
