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

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
                    call lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f64(context%session, &
                                                   context%f64_print_format_id, &
                                                   value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i32(context%session, &
                                                   context%i32_print_format_id, &
                                                   value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    if (.not. context%symbols(symbol_index)%has_character_value) then
                        call unsupported_feature_error( &
                            'character variable print', node%line, node%column, &
                            'direct LIRIC session requires character variables '// &
                            'to be assigned from a literal before printing', &
                            error_msg)
                        return
                    end if
                    if (.not. emit_liric_print_string_operand( &
                        context%session, context%str_print_format_id, &
                        context%symbols(symbol_index)%value, error_msg)) return
                    return
                end if
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
        type is (call_or_subscript_node)
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
                value = i32_immediate(context%session, logical_i32_value(node%value))
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

    subroutine lower_print_expression_value(arena, node_index, context, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        real(c_double) :: real_value
        integer :: symbol_index

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
                    call lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f64_value(context%session, &
                                                         context%f64_print_format_id, &
                                                         value, error_msg)) &
                        return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i32_value(context%session, &
                                                         context%i32_print_format_id, &
                                                         value, error_msg)) &
                        return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    if (.not. context%symbols(symbol_index)%has_character_value) then
                        call unsupported_feature_error( &
                            'character variable print', node%line, node%column, &
                            'direct LIRIC session requires character variables '// &
                            'to be assigned from a literal before printing', &
                            error_msg)
                        return
                    end if
                    if (.not. emit_liric_print_string_operand_value( &
                        context%session, context%str_print_format_id, &
                        context%symbols(symbol_index)%value, error_msg)) return
                    return
                end if
            end if
        type is (binary_op_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     context%f64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
        type is (call_or_subscript_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
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
                if (.not. emit_liric_print_string_value(context%session, &
                                                        context%str_print_format_id, &
                                                        trim(string_name), &
                                                        character_value, &
                                                        error_msg)) return
                return
            end if
            if (is_logical_literal(node)) then
                value = i32_immediate(context%session, logical_i32_value(node%value))
                if (.not. emit_liric_print_i32_value(context%session, &
                                                     context%i32_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
            if (is_real_literal(node)) then
                call parse_f64_literal(node%value, real_value, error_msg)
                if (len_trim(error_msg) > 0) return
                value = liric_f64_immediate(context%session, real_value)
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     context%f64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
        end select

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32_value(context%session, &
                                             context%i32_print_format_id, value, &
                                             error_msg)) return
    end subroutine lower_print_expression_value
