    subroutine lower_print_expression(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: re_val, im_val
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        real(c_double) :: real_value
        real(c_float) :: real_value_f32
        integer :: symbol_index
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

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
                string_name = ffc_unit_global_name(context, 'str.', &
                                                   context%string_literal_count)
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
                if (literal_is_f64(lit_value, context)) then
                    call parse_f64_literal(lit_value, context, real_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = liric_f64_immediate(context%session, real_value)
                    if (.not. emit_liric_print_f64(context%session, &
                                                   value, error_msg)) return
                else
                    call parse_f64_literal(lit_value, context, real_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    real_value_f32 = real(real_value, c_float)
                    value = liric_f32_immediate(context%session, real_value_f32)
                    if (.not. emit_liric_print_f32(context%session, &
                                                   value, error_msg)) return
                end if
                return
            end if
            ! An integer literal falls through to the integer print below.
        end if

        if (is_binary_op(arena, node_index)) then
            ! Character // concatenation prints as a runtime string buffer; only
            ! the // operands case is handled here (intrinsic agent, bug 2).
            if (is_character_concat(arena, node_index, context)) then
                call lower_print_char_expr(arena, node_index, context, error_msg)
                return
            end if
            if (is_complex_valued(arena, node_index, context, VALUE_C8)) then
                call eval_c8_rhs(arena, node_index, context, re_val, im_val, &
                                 error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_complex8_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
                return
            end if
            if (is_complex_valued(arena, node_index, context, VALUE_C4)) then
                call eval_c4_rhs(arena, node_index, context, re_val, im_val, &
                                 error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_complex4_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
                return
            end if
            if (is_f32_expression(arena, node_index, context)) then
                call lower_f32_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f32(context%session, &
                                               value, error_msg)) return
                return
            end if
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               value, error_msg)) return
                return
            end if
            if (is_i64_binary_op(arena, node_index, context)) then
                call lower_i64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i64(context%session, &
                                               context%i64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
            if (is_i8_binary_op(arena, node_index, context)) then
                call lower_i8_expression(arena, node_index, context, value, &
                                         error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i8(context%session, &
                                              context%i8_print_format_id, &
                                              value, error_msg)) return
                return
            end if
            if (is_i16_binary_op(arena, node_index, context)) then
                call lower_i16_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i16(context%session, &
                                               context%i16_print_format_id, &
                                               value, error_msg)) return
                return
            end if
            ! A non-i64 binary op falls through to the integer(4) print below.
        else if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F32) then
                    call lower_f32_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f32(context%session, &
                                                   value, error_msg)) return
                    return
                end if
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
                if (context%symbols(symbol_index)%value_kind == VALUE_I64) then
                    call lower_i64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i64(context%session, &
                                                   context%i64_print_format_id, &
                                                   value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_I8) then
                    call lower_i8_expression(arena, node_index, context, &
                                             value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i8(context%session, &
                                                  context%i8_print_format_id, &
                                                  value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_I16) then
                    call lower_i16_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i16(context%session, &
                                                   context%i16_print_format_id, &
                                                   value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_C4 .and. &
                    .not. context%symbols(symbol_index)%is_array) then
                    call lower_print_complex4(symbol_index, context, error_msg)
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_C8 .and. &
                    .not. context%symbols(symbol_index)%is_array) then
                    call lower_print_complex8(symbol_index, context, error_msg)
                    return
                end if
                if ((context%symbols(symbol_index)%value_kind == VALUE_C4 .or. &
                     context%symbols(symbol_index)%value_kind == VALUE_C8) &
                    .and. context%symbols(symbol_index)%is_array) then
                    call unsupported_feature_error('complex array print', &
                        get_node_line(arena, node_index), &
                        get_node_column(arena, node_index), &
                        'direct LIRIC session supports printing individual '// &
                        'complex array elements, not a whole complex array', &
                        error_msg)
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    if (.not. context%symbols(symbol_index)%has_character_value) then
                        call unsupported_feature_error( &
                            'character variable print', &
                            get_node_line(arena, node_index), &
                            get_node_column(arena, node_index), &
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
        else

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%is_array_access .and. &
                array_access_value_kind(node, context) == VALUE_CHARACTER) then
                call lower_character_array_element_print(arena, node, context, &
                                                         .false., error_msg)
                return
            end if
            if ((node%is_array_access .or. &
                 is_allocatable_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_LOGICAL) then
                call lower_logical_expression(arena, node_index, context, value, &
                                              error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_print_logical_value(context, value, error_msg)
                return
            end if
            if (is_contained_complex_function(context, node%name)) then
                call lower_print_complex_result_call(arena, node_index, context, &
                                                     error_msg)
                return
            end if
            if (.not. node%is_array_access .and. &
                is_contained_array_function(context, node%name)) then
                call lower_print_array_result_call(arena, node_index, context, &
                                                   error_msg)
                return
            end if
            ! any()/all() return LOGICAL and print as T/F (intrinsic agent).
            if (is_logical_reduction_call(arena, node_index, context)) then
                call lower_i32_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_print_logical_value(context, value, error_msg)
                return
            end if
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_logical_function(context, node%name)) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call lower_print_logical_value(context, value, error_msg)
                    return
                end if
            end if
            if (is_f32_expression(arena, node_index, context)) then
                call lower_f32_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f32(context%session, &
                                               value, error_msg)) return
                return
            end if
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
        type(lr_operand_desc_t) :: re_val, im_val
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        real(c_double) :: real_value
        real(c_float) :: real_value_f32
        integer :: symbol_index
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

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
                string_name = ffc_unit_global_name(context, 'str.', &
                                                   context%string_literal_count)
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
                if (literal_is_f64(lit_value, context)) then
                    call parse_f64_literal(lit_value, context, real_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = liric_f64_immediate(context%session, real_value)
                    if (.not. emit_liric_print_f64_value(context%session, &
                                                         value, error_msg)) return
                else
                    call parse_f64_literal(lit_value, context, real_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    real_value_f32 = real(real_value, c_float)
                    value = liric_f32_immediate(context%session, real_value_f32)
                    if (.not. emit_liric_print_f32_value(context%session, &
                                                         value, error_msg)) return
                end if
                return
            end if
            ! An integer literal falls through to the integer print below.
        end if

        if (is_binary_op(arena, node_index)) then
            ! Character // concatenation prints as a runtime string buffer; only
            ! the // operands case is handled here (intrinsic agent, bug 2).
            if (is_character_concat(arena, node_index, context)) then
                call lower_print_char_expr(arena, node_index, context, error_msg)
                return
            end if
            if (is_complex_valued(arena, node_index, context, VALUE_C8)) then
                call eval_c8_rhs(arena, node_index, context, re_val, im_val, &
                                 error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_complex8_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
                return
            end if
            if (is_complex_valued(arena, node_index, context, VALUE_C4)) then
                call eval_c4_rhs(arena, node_index, context, re_val, im_val, &
                                 error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_complex4_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
                return
            end if
            if (is_f32_expression(arena, node_index, context)) then
                call lower_f32_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f32_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            if (is_i64_binary_op(arena, node_index, context)) then
                call lower_i64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i64_value(context%session, &
                                                     context%i64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
            if (is_i8_binary_op(arena, node_index, context)) then
                call lower_i8_expression(arena, node_index, context, value, &
                                         error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i8_value(context%session, &
                                                    context%i8_print_format_id, &
                                                    value, error_msg)) return
                return
            end if
            if (is_i16_binary_op(arena, node_index, context)) then
                call lower_i16_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i16_value(context%session, &
                                                     context%i16_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
            ! A non-i64 binary op falls through to the integer(4) print below.
        else if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F32) then
                    call lower_f32_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f32_value(context%session, &
                                                         value, error_msg)) &
                        return
                    return
                end if
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
                if (context%symbols(symbol_index)%value_kind == VALUE_I64) then
                    call lower_i64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i64_value(context%session, &
                                                         context%i64_print_format_id, &
                                                         value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_I8) then
                    call lower_i8_expression(arena, node_index, context, &
                                             value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i8_value(context%session, &
                                                        context%i8_print_format_id, &
                                                        value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_I16) then
                    call lower_i16_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i16_value(context%session, &
                                                         context%i16_print_format_id, &
                                                         value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_C4 .and. &
                    .not. context%symbols(symbol_index)%is_array) then
                    call lower_print_complex4(symbol_index, context, error_msg)
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_C8 .and. &
                    .not. context%symbols(symbol_index)%is_array) then
                    call lower_print_complex8(symbol_index, context, error_msg)
                    return
                end if
                if ((context%symbols(symbol_index)%value_kind == VALUE_C4 .or. &
                     context%symbols(symbol_index)%value_kind == VALUE_C8) &
                    .and. context%symbols(symbol_index)%is_array) then
                    call unsupported_feature_error('complex array print', &
                        get_node_line(arena, node_index), &
                        get_node_column(arena, node_index), &
                        'direct LIRIC session supports printing individual '// &
                        'complex array elements, not a whole complex array', &
                        error_msg)
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    if (.not. context%symbols(symbol_index)%has_character_value) then
                        call unsupported_feature_error( &
                            'character variable print', &
                            get_node_line(arena, node_index), &
                            get_node_column(arena, node_index), &
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
        else

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%is_array_access .and. &
                array_access_value_kind(node, context) == VALUE_CHARACTER) then
                call lower_character_array_element_print(arena, node, context, &
                                                         .true., error_msg)
                return
            end if
            if (is_contained_complex_function(context, node%name)) then
                call lower_print_complex_result_call(arena, node_index, context, &
                                                     error_msg)
                return
            end if
            if (.not. node%is_array_access .and. &
                is_contained_array_function(context, node%name)) then
                call lower_print_array_result_call(arena, node_index, context, &
                                                   error_msg)
                return
            end if
            ! any()/all() return LOGICAL and print as T/F (intrinsic agent).
            if (is_logical_reduction_call(arena, node_index, context)) then
                call lower_i32_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_print_logical_value(context, value, error_msg)
                return
            end if
            ! obj%comp(i): array-component element, formatted by element kind.
            if (node%base_expr_index > 0) then
                call lower_print_component_element_value(arena, node, node_index, &
                                                         context, error_msg)
                return
            end if
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_logical_function(context, node%name)) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call lower_print_logical_value(context, value, error_msg)
                    return
                end if
            end if
            if (is_f32_expression(arena, node_index, context)) then
                call lower_f32_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f32_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
                return
            end if
            if ((node%is_array_access .or. &
                 is_declared_array_element_ref(node, context) .or. &
                 is_allocatable_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_LOGICAL) then
                ! Logical array element: load as i32 then format as T/F.
                call lower_i32_array_element(arena, node, context, value, &
                                             error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_print_logical_value(context, value, error_msg)
                return
            end if
            if ((node%is_array_access .or. &
                 is_declared_array_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_I64) then
                call lower_i64_array_element(arena, node, context, value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i64_value(context%session, &
                                                     context%i64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
            if ((node%is_array_access .or. &
                 is_declared_array_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_I8) then
                call lower_i8_array_element(arena, node, context, value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i8_value(context%session, &
                                                    context%i8_print_format_id, &
                                                    value, error_msg)) return
                return
            end if
            if ((node%is_array_access .or. &
                 is_declared_array_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_I16) then
                call lower_i16_array_element(arena, node, context, value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_i16_value(context%session, &
                                                     context%i16_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
            if ((node%is_array_access .or. &
                 is_declared_array_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_C4) then
                call load_complex_array_element(arena, node, context, VALUE_C4, &
                    re_val, im_val, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_complex4_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
                return
            end if
            if ((node%is_array_access .or. &
                 is_declared_array_element_ref(node, context)) .and. &
                array_access_value_kind(node, context) == VALUE_C8) then
                call load_complex_array_element(arena, node, context, VALUE_C8, &
                    re_val, im_val, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_complex8_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
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
        type is (component_access_node)
            call lower_print_component_access_value(arena, node, node_index, &
                                                    context, error_msg)
            if (len_trim(error_msg) > 0) return
            return
        end select
        end if

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32_value(context%session, &
                                             context%i32_print_format_id, value, &
                                             error_msg)) return
    end subroutine lower_print_expression_value

    integer function array_access_value_kind(node, context) result(value_kind)
        ! Element value kind of an array subscript reference, or VALUE_I32 when
        ! the name is not a declared array (callers fall back to integer print).
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        value_kind = VALUE_I32
        if (.not. allocated(node%name)) return
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_array .and. &
            .not. context%symbols(symbol_index)%is_allocatable) return
        value_kind = context%symbols(symbol_index)%value_kind
    end function array_access_value_kind

    subroutine lower_print_component_access_value(arena, node, node_index, &
                                                  context, error_msg)
        ! List-directed print of a scalar derived-type component, formatted by
        ! the component's value kind (the statement loop emits the lead blank).
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: comp_kind

        comp_kind = derived_component_access_kind(arena, node, context)
        select case (comp_kind)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_f64_value(context%session, value, &
                                                 error_msg)) return
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_f32_value(context%session, value, &
                                                 error_msg)) return
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_print_logical_value(context, value, error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_i32_value(context%session, &
                                                 context%i32_print_format_id, &
                                                 value, error_msg)) return
        end select
    end subroutine lower_print_component_access_value

    subroutine lower_print_component_element_value(arena, node, node_index, &
                                                   context, error_msg)
        ! List-directed print of obj%comp(i), an element of an array component,
        ! formatted by the component's element value kind.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: comp_kind

        comp_kind = component_element_access_kind(arena, node, context)
        select case (comp_kind)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_f64_value(context%session, value, &
                                                 error_msg)) return
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_f32_value(context%session, value, &
                                                 error_msg)) return
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_print_logical_value(context, value, error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_i32_value(context%session, &
                                                 context%i32_print_format_id, &
                                                 value, error_msg)) return
        end select
    end subroutine lower_print_component_element_value

    integer function component_element_access_kind(arena, node, context) &
            result(kind)
        ! Element value kind of an obj%comp(i) array-component access. The
        ! node's base_expr_index is the component_access (obj%comp), whose
        ! component kind is the element kind. Falls back to VALUE_I32.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context

        kind = VALUE_I32
        if (node%base_expr_index <= 0) return
        if (.not. node_exists(arena, node%base_expr_index)) return
        select type (acc => arena%entries(node%base_expr_index)%node)
        type is (component_access_node)
            kind = derived_component_access_kind(arena, acc, context)
        end select
    end function component_element_access_kind

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
        true_name = ffc_unit_global_name( &
            context, 'logical.', context%string_literal_count)
        if (.not. set_liric_block(context%session, true_block, error_msg)) return
        context%current_block_id = true_block
        if (.not. emit_liric_print_string_value(context%session, &
                                                context%str_print_format_id, &
                                                trim(true_name), 'T', &
                                                error_msg)) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        context%string_literal_count = context%string_literal_count + 1
        false_name = ffc_unit_global_name( &
            context, 'logical.', context%string_literal_count)
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
