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
        character(len=:), allocatable :: lit_value, lit_type

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            if (is_character_literal(arena, node_index)) then
                call strip_literal_quotes(lit_value, character_value)
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
            if (is_logical_literal(arena, node_index)) then
                value = i32_immediate(context%session, logical_i32_value(lit_value))
                if (.not. emit_liric_print_i32(context%session, &
                                               context%i32_print_format_id, &
                                               value, error_msg)) return
                return
            end if
            if (is_real_literal(arena, node_index)) then
                call parse_f64_literal(lit_value, real_value, error_msg)
                if (len_trim(error_msg) > 0) return
                value = liric_f64_immediate(context%session, real_value)
                if (.not. emit_liric_print_f64(context%session, &
                                               value, error_msg)) return
                return
            end if
            ! An integer literal falls through to the integer print below.
        end if

        if (is_binary_op(arena, node_index)) then
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               value, error_msg)) return
                return
            end if
            ! A non-real binary op falls through to the integer print below.
        else

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
                    call lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f64(context%session, &
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
        type is (call_or_subscript_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               value, error_msg)) return
                return
            end if
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_deferred_char_function(context, node%name)) then
                    call lower_print_deferred_char_call(arena, node, context, &
                        error_msg)
                    return
                end if
            end if
        end select
        end if

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
        character(len=:), allocatable :: lit_value, lit_type

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            if (is_character_literal(arena, node_index)) then
                call strip_literal_quotes(lit_value, character_value)
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
            if (is_logical_literal(arena, node_index)) then
                value = i32_immediate(context%session, logical_i32_value(lit_value))
                call lower_print_logical_value(context, value, error_msg)
                return
            end if
            if (is_real_literal(arena, node_index)) then
                call parse_f64_literal(lit_value, real_value, error_msg)
                if (len_trim(error_msg) > 0) return
                value = liric_f64_immediate(context%session, real_value)
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            ! An integer literal falls through to the integer print below.
        end if

        if (is_binary_op(arena, node_index)) then
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            ! A non-real binary op falls through to the integer print below.
        else

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
                    call lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f64_value(context%session, &
                                                         value, error_msg)) &
                        return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call lower_print_logical_value(context, value, error_msg)
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
        type is (call_or_subscript_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_deferred_char_function(context, node%name)) then
                    call lower_print_deferred_char_call(arena, node, context, &
                        error_msg)
                    return
                end if
            end if
            if (is_char_expr_call(arena, node_index)) then
                call lower_print_char_expr(arena, node_index, context, error_msg)
                return
            end if
        end select
        end if

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32_value(context%session, &
                                             context%i32_print_format_id, value, &
                                             error_msg)) return
    end subroutine lower_print_expression_value

    subroutine lower_print_char_expr(arena, node_index, context, error_msg)
        ! Print a character expression (e.g. trim(s)). The list-directed
        ! leading blank is emitted by the print statement loop.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length

        call char_expr_operands(arena, node_index, context, data_ptr, length, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_string_operand_value(context%session, &
            context%str_print_format_id, data_ptr, error_msg)) return
    end subroutine lower_print_char_expr

    subroutine lower_print_logical_value(context, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond
        type(lr_operand_desc_t) :: zero
        integer(c_int32_t) :: true_block
        integer(c_int32_t) :: false_block
        integer(c_int32_t) :: merge_block
        character(len=64) :: true_name
        character(len=64) :: false_name

        ! gfortran list-directed logical output is " T" / " F"; the leading
        ! blank is the list separator, so the field itself is "T" / "F".
        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, value, zero, &
                                      cond, error_msg)) return

        true_block = create_liric_block(context%session)
        false_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, cond, true_block, &
                                    false_block, error_msg)) return

        context%string_literal_count = context%string_literal_count + 1
        write (true_name, '(A,I0)') '.ffc.logical.', context%string_literal_count
        if (.not. set_liric_block(context%session, true_block, error_msg)) return
        context%current_block_id = true_block
        if (.not. emit_liric_print_string_value(context%session, &
                                                context%str_print_format_id, &
                                                trim(true_name), 'T', &
                                                error_msg)) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        context%string_literal_count = context%string_literal_count + 1
        write (false_name, '(A,I0)') '.ffc.logical.', context%string_literal_count
        if (.not. set_liric_block(context%session, false_block, error_msg)) return
        context%current_block_id = false_block
        if (.not. emit_liric_print_string_value(context%session, &
                                                context%str_print_format_id, &
                                                trim(false_name), 'F', &
                                                error_msg)) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine lower_print_logical_value
