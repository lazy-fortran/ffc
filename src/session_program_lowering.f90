module session_program_lowering
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, &
                                                                              c_int64_t
    use ast_base, only: LITERAL_INTEGER
    use ast_nodes_bounds, only: array_slice_node, array_bounds_node, &
                                range_expression_node
    use ast_nodes_core, only: component_access_node, array_literal_node, &
                              pointer_assignment_node
    use ast_nodes_data, only: derived_type_node, type_binding_node
    use ast_nodes_misc, only: use_statement_node, interface_block_node, &
                              visibility_statement_node
    use ast_nodes_conditional, only: select_type_node, type_guard_block_node
    use fortfront, only: assignment_node, ast_arena_t, &
                         call_or_subscript_node, case_block_node, &
                         case_range_node, &
                         case_default_node, declaration_node, do_loop_node, &
                         do_while_node, cycle_node, exit_node, function_def_node, &
                         if_node, &
                         parameter_declaration_node, &
                         print_statement_node, program_node, read_statement_node, &
                         module_node, &
                         return_node, select_case_node, stop_node, &
                         subroutine_def_node, write_statement_node, &
                         allocate_statement_node, deallocate_statement_node, &
                         get_subroutine_call_arg_indices, &
                         get_subroutine_call_name, is_subroutine_call_statement, &
                         is_binary_op, get_binary_op_info, &
                         is_literal, get_literal_info, &
                         is_identifier, get_identifier_name, &
                         is_declaration_node, is_module_node, is_program_node
use liric_session_bindings, only: destroy, begin_i32_main, &
                                       begin_i32_function, begin_void_subroutine, &
                                       begin_ptr_function, &
                                      emit_ret_i32_operand, emit_ret_void, &
                                      finish_function, finish_and_emit_exe, &
                                      finish_and_emit_object, emit_void_call, &
                                      emit_i32_call, liric_session_create, &
                                      i32_immediate, i32_vreg, lr_operand_desc_t, &
                                      lr_type_ptr_s, &
                                      LR_OP_ADD, LR_OP_SREM, LR_OP_SUB, &
                                      LR_OP_MUL, &
                                      LR_OP_AND, LR_OP_OR, LR_OP_XOR, &
                                      LR_OP_SHL, LR_OP_LSHR, LR_OP_KIND_IMM_I64
    use liric_session_memory_bindings, only: reserve_i32_vreg, i64_immediate, &
                                              ptr_vreg, &
                                              emit_i32_binary, emit_i32_binary_into, &
                                              emit_i32_copy_to, emit_i32_alloca, &
                                              emit_i32_load, emit_i32_store, &
                                              emit_i64_load, emit_ptr_load, &
                                              emit_i64_store, &
                                              emit_i64_binary, emit_i64_alloca, &
                                              emit_alloca_bytes, emit_malloc, &
                                              emit_free, emit_ptr_store, &
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
                                              emit_liric_phi_n, &
                                              LR_FCMP_OGE, &
                                              LR_FCMP_OLE, &
                                              LR_CMP_SGE, &
                                              LR_CMP_SGT, &
                                              LR_CMP_SLE, &
                                              LR_CMP_SLT, &
                                              LR_CMP_NE, &
                                              LR_CMP_EQ, &
                                              set_liric_block
use liric_session_format_bindings, only: LR_OP_FSUB, &
                                            prepare_liric_print_runtime, &
                                            create_printf_format_global, &
                                            printf_format_ptr, &
                                            create_type_info_global
    use liric_session_real_print_bindings, only: synthesize_real8_printer, &
                                                 synthesize_get_arg_helper, &
                                                 emit_get_arg_call, emit_snprintf, &
                                                 emit_sscanf
    use liric_session_io_bindings, only: emit_liric_f64_binary, &
                                          emit_liric_i32_to_f64, &
                                          emit_liric_f64_to_i32, &
                                          emit_liric_char_byte_zext, &
                                          emit_liric_i32_to_i64, &
                                          emit_liric_store_char_byte, &
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
    use ffc_fortfront_queries, only: node_exists, get_node_type_at, &
                                     get_program_body_info, get_module_body_info, &
                                     get_function_body_info, get_subroutine_body_info, &
                                     get_select_case_info, get_case_block_info, &
                                     get_case_default_body, get_case_range_info, &
                                     get_select_type_info, get_type_guard_info, &
                                     is_derived_type_node, is_declaration_node, &
                                     get_derived_type_name, get_derived_type_components, &
                                     get_declaration_var_name, get_declaration_type_name, &
                                     get_declaration_has_initializer, &
                                     get_declaration_initializer_index
    use fortfront_utils, only: get_node_as_function_def, &
                               get_node_as_program, &
                               get_node_as_subroutine_def
    use fortfront, only: get_node_line, get_node_column
    use session_program_lowering_types, only: lowering_context_t, &
                                                branch_result_t, symbol_t, &
                                                array_section_info_t, &
                                                derived_type_info_t, &
                                                module_exports_t, &
                                                external_procedure_t, &
                                                MAX_PROC_ARGS, &
                                                VALUE_I32, VALUE_F64, &
                                                 VALUE_LOGICAL, VALUE_CHARACTER, &
                                                 VALUE_DERIVED, &
                                                 VALUE_DEFERRED_CHARACTER_RESULT, &
                                                 VALUE_SUBROUTINE, VALUE_C_PTR, &
                                                 VALUE_CLASS_STAR, &
                                                 TYPE_ID_INTEGER, TYPE_ID_REAL, &
                                                 TYPE_ID_LOGICAL, &
                                                 I32_INTRINSIC_NONE, &
                                                I32_INTRINSIC_ABS, I32_INTRINSIC_MIN, &
                                                I32_INTRINSIC_MAX, I32_INTRINSIC_MOD, &
                                                I32_INTRINSIC_IAND, I32_INTRINSIC_IOR, &
                                                I32_INTRINSIC_IEOR, I32_INTRINSIC_NOT, &
                                                I32_INTRINSIC_ISHFT, &
                                                I32_INTRINSIC_ISHFTC, &
                                                I32_INTRINSIC_SIGN, &
                                                I32_INTRINSIC_INT, I32_INTRINSIC_NINT, &
                                                I32_INTRINSIC_FLOOR, &
                                                I32_INTRINSIC_CEILING, &
                                                 I32_INTRINSIC_MATMUL, &
                                                 I32_INTRINSIC_TRANSPOSE, &
                                                 I32_INTRINSIC_DOT_PRODUCT, &
                                                 I32_INTRINSIC_RESHAPE, &
                                                 F64_INTRINSIC_SIGN, &
                                                F64_INTRINSIC_SQRT, F64_INTRINSIC_EXP, &
                                                F64_INTRINSIC_LOG, F64_INTRINSIC_SIN, &
                                                F64_INTRINSIC_COS, F64_INTRINSIC_TAN, &
                                                F64_INTRINSIC_ATAN, F64_INTRINSIC_ATAN2, &
                                                F64_INTRINSIC_NONE, F64_INTRINSIC_ABS, &
                                                F64_INTRINSIC_MIN, F64_INTRINSIC_MAX, &
                                                F64_INTRINSIC_REAL, I32_INTRINSIC_NAMES, &
                                                I32_INTRINSIC_IDS, F64_INTRINSIC_NAMES, &
                                                F64_INTRINSIC_IDS
    use ffc_module_artefact, only: module_info_t, fmod_parameter_t, &
                                   fmod_component_t, fmod_derived_type_t, &
                                   write_fmod, read_fmod
    implicit none
    private
    public :: lower_program_to_liric_exe
    public :: lower_program_to_liric_object
contains
    include 'session_program_lowering_top.inc'
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
            if (node%is_allocatable) then
                call lower_allocatable_declaration(node, context, error_msg)
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
        else if (value_kind == VALUE_CLASS_STAR) then
            call define_declared_class_star_symbol(context, name, error_msg)
        else
            call define_symbol(context, name, value_kind, error_msg)
        end if
    end subroutine define_declared_symbol

    subroutine define_declared_class_star_symbol(context, name, error_msg)
        ! A class(*) dummy: the parameter pointer addresses a 16-byte
        ! {void* data; i64 type_id} descriptor (#141). data is at offset 0 and
        ! the runtime type id at offset 8.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        index = find_symbol(context, name)
        if (index <= 0) then
            error_msg = 'class(*) is only supported as a dummy argument: '// &
                        trim(name)
            return
        end if
        if (.not. context%symbols(index)%is_parameter) then
            error_msg = 'class(*) is only supported as a dummy argument: '// &
                        trim(name)
            return
        end if
        context%symbols(index)%value_kind = VALUE_CLASS_STAR
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 0_c_int64_t, &
            context%symbols(index)%deferred_data, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(index)%address, 8_c_int64_t, &
            context%symbols(index)%deferred_length, error_msg)) return
        call set_empty(error_msg)
    end subroutine define_declared_class_star_symbol
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
    include 'session_program_lowering_array_elements.inc'
    include 'session_program_lowering_allocatable.inc'
    include 'session_program_lowering_internal_write.inc'
    include 'session_program_lowering_internal_read.inc'
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
            ! A re-declaration of an existing same-kind symbol is benign (e.g. a
            ! function result variable pre-defined from the result clause and
            ! then declared in the body). True conflicts are caught by FortFront.
            if (context%symbols(existing_index)%value_kind == value_kind) then
                call set_empty(error_msg)
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
        else if (value_kind == VALUE_C_PTR) then
            call define_c_ptr_symbol(context, name, error_msg)
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
        else if (value_kind == VALUE_C_PTR) then
            context%symbols(index)%value = null_ptr_operand(context)
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
            if (target%base_expr_index > 0) then
                call lower_derived_component_element_assignment(arena, node, &
                    target, context, error_msg)
                return
            end if
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
        if (context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('allocatable array assignment', &
                node%line, node%column, 'assignment to an allocatable array '// &
                'is not supported; element access lands in a later issue', &
                error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_array) then
            if (node_exists(arena, node%value_index)) then
                select type (val => arena%entries(node%value_index)%node)
                type is (array_literal_node)
                    call lower_array_constructor_assignment(arena, val, &
                        symbol_index, context, error_msg)
                    return
                class default
                    call lower_array_whole_assignment(arena, node, symbol_index, &
                                                      context, error_msg)
                    return
                end select
            end if
            if (is_identifier(arena, node%target_index)) then
                call unsupported_feature_error('array assignment target', &
                    get_node_line(arena, node%target_index), &
                    get_node_column(arena, node%target_index), &
                    'whole-array assignment is not supported', error_msg)
            else
                call unsupported_feature_error('array assignment target', &
                                               node%line, node%column, &
                                               'whole-array assignment is not '// &
                                               'supported', error_msg)
            end if
            return
        end if
        if (context%symbols(symbol_index)%is_derived) then
            if (is_derived_result_call(arena, node%value_index, context)) then
                call lower_derived_result_call(arena, node%value_index, &
                                               symbol_index, context, error_msg)
                return
            end if
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
        else if (context%symbols(symbol_index)%value_kind == VALUE_C_PTR) then
            call lower_c_ptr_expression(arena, node%value_index, context, value, &
                                        error_msg)
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
    include 'session_program_lowering_c_ptr.inc'
    include 'session_program_lowering_fmod.inc'
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
        if (same_name(name, 'get_command_argument')) then
            call lower_get_command_argument(arena, arg_indices, context, error_msg)
            return
        end if
        if (same_name(name, 'c_f_pointer')) then
            call lower_c_f_pointer(arena, arg_indices, context, error_msg)
            return
        end if
        if (is_method_subroutine_call(name)) then
            call lower_method_subroutine_call(arena, name, arg_indices, context, &
                                              error_msg)
            return
        end if
        if (external_procedure_index(context, name) > 0) then
            call lower_external_void_call(arena, node_index, &
                external_procedure_index(context, name), context, error_msg)
            return
        end if
        call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
                                    name, args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_call_with_optional_padding(context, &
                call_emit_name(arena, name), args, error_msg)) &
            return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_subroutine_call

    logical function emit_call_with_optional_padding(context, name, args, &
                                                     error_msg) result(ok)
        ! Emit a void call to a contained subroutine, padding omitted trailing
        ! optional dummies with null pointers up to the callee's declared
        ! parameter count.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        type(lr_operand_desc_t), intent(in) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: padded(:)
        type(lr_operand_desc_t) :: nullptr
        integer :: pcount, j

        ok = .false.
        pcount = proc_param_count(context, name)
        if (pcount <= size(args)) then
            ok = emit_void_call(context%session, name, args, error_msg)
            return
        end if

        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        nullptr%typ = lr_type_ptr_s(context%session%handle)
        nullptr%global_offset = 0_c_int64_t
        allocate (padded(pcount))
        if (size(args) > 0) padded(1:size(args)) = args
        do j = size(args) + 1, pcount
            padded(j) = nullptr
        end do
        ok = emit_void_call(context%session, name, padded, error_msg)
    end function emit_call_with_optional_padding
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
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'condition index does not reference an AST node'
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(adjustl(lowercase_text(bin_op))) == '.not.') then
                ! Unary .not. is parsed as a binary op with a virtual operand;
                ! the real condition is the right operand. Invert it.
                call lower_i1_condition(arena, bin_right, context, lhs, &
                                        error_msg)
                if (len_trim(error_msg) > 0) return
                rhs = lhs
                rhs%kind = LR_OP_KIND_IMM_I64
                rhs%payload = 0_c_int64_t
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, lhs, &
                                              rhs, value, error_msg)) return
                call set_empty(error_msg)
                return
            end if
            call lower_i32_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_compare_predicate(bin_op, pred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, pred, lhs, rhs, &
                                          value, error_msg)) return
            return
        end if
        if (is_literal(arena, node_index)) then
            call lower_logical_expression(arena, node_index, context, lhs, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                                          rhs, value, error_msg)) return
            return
        end if
        if (is_identifier(arena, node_index)) then
            call lower_logical_expression(arena, node_index, context, lhs, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            rhs = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, lhs, &
                                          rhs, value, error_msg)) return
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (is_present_call(arena, node_index)) then
                call lower_present_condition(arena, node_index, context, value, &
                                             error_msg)
            else if (allocated(node%name)) then
                if (same_name(node%name, 'c_associated')) then
                    call lower_c_associated(arena, node%arg_indices, context, &
                                            value, error_msg)
                else
                    error_msg = 'direct LIRIC session IF condition supports '// &
                                'comparisons, logicals, and present()'
                end if
            else
                error_msg = 'direct LIRIC session IF condition supports '// &
                            'comparisons, logicals, and present()'
            end if
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
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            return
        end if
        select type (node => arena%entries(node_index)%node)
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
        type(symbol_t), allocatable :: tmp(:)
        integer :: old_size

        if (context%symbol_count < size(context%symbols)) return
        old_size = size(context%symbols)
        call move_alloc(context%symbols, tmp)
        allocate(context%symbols(2 * old_size))
        context%symbols(1:old_size) = tmp
    end subroutine grow_symbols

    subroutine grow_derived_types(context)
        type(lowering_context_t), intent(inout) :: context
        type(derived_type_info_t), allocatable :: tmp(:)
        integer :: old_size

        if (context%derived_type_count < size(context%derived_types)) return
        old_size = size(context%derived_types)
        call move_alloc(context%derived_types, tmp)
        allocate(context%derived_types(2 * old_size))
        context%derived_types(1:old_size) = tmp
    end subroutine grow_derived_types

    subroutine grow_module_exports(context)
        type(lowering_context_t), intent(inout) :: context
        type(module_exports_t), allocatable :: tmp(:)
        integer :: old_size

        if (context%module_export_count < size(context%module_exports)) return
        old_size = size(context%module_exports)
        call move_alloc(context%module_exports, tmp)
        allocate(context%module_exports(2 * old_size))
        context%module_exports(1:old_size) = tmp
    end subroutine grow_module_exports

    subroutine grow_function_names(context)
        type(lowering_context_t), intent(inout) :: context
        character(len=64), allocatable :: tmp_names(:)
        integer, allocatable :: tmp_kinds(:)
        integer, allocatable :: tmp_counts(:)
        integer, allocatable :: tmp_indices(:)
        integer :: old_size, new_size

        if (context%function_count < size(context%function_names)) return
        old_size = size(context%function_names)
        new_size = 2 * old_size
        call move_alloc(context%function_names, tmp_names)
        call move_alloc(context%function_value_kinds, tmp_kinds)
        call move_alloc(context%function_param_counts, tmp_counts)
        if (allocated(context%function_node_indices)) &
            call move_alloc(context%function_node_indices, tmp_indices)
        allocate(context%function_names(new_size))
        allocate(context%function_value_kinds(new_size))
        allocate(context%function_param_counts(new_size))
        allocate(context%function_node_indices(new_size))
        context%function_param_counts = 0
        context%function_node_indices = 0
        context%function_names(1:old_size) = tmp_names
        context%function_value_kinds(1:old_size) = tmp_kinds
        context%function_param_counts(1:old_size) = tmp_counts
        if (allocated(tmp_indices)) &
            context%function_node_indices(1:size(tmp_indices)) = tmp_indices
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

    logical function same_name(lhs, rhs)
        character(len=*), intent(in) :: lhs
        character(len=*), intent(in) :: rhs

        same_name = lowercase_text(lhs) == lowercase_text(rhs)
    end function same_name

    function lowercase_text(text) result(lowered)
        character(len=*), intent(in) :: text
        character(len=len_trim(text)) :: lowered
        integer :: code
        integer :: i

        do i = 1, len(lowered)
            code = iachar(text(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                lowered(i:i) = achar(code + 32)
            else
                lowered(i:i) = text(i:i)
            end if
        end do
    end function lowercase_text

    include 'session_program_lowering_select.inc'
    include 'session_program_lowering_diagnostics.inc'
end module session_program_lowering
