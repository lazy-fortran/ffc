    logical function is_internal_write(node, context)
        ! True when write's unit is a fixed-length character variable rather
        ! than a unit number or '*'.
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_internal_write = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*') return
        symbol_index = find_symbol(context, trim(node%unit_spec))
        if (symbol_index <= 0) return
        is_internal_write = &
            context%symbols(symbol_index)%value_kind == VALUE_CHARACTER .and. &
            context%symbols(symbol_index)%character_length > 0
    end function is_internal_write

    subroutine lower_internal_write(arena, node, context, error_msg)
        ! write (buf, fmt) value: format value through a single edit descriptor
        ! into the fixed-length character variable buf, blank-padded to its
        ! declared length. fmt is a literal '(I0)' / '(Iw)' / '(A)'.
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: printf_fmt
        character :: kind_char
        integer :: symbol_index, buflen
        integer(c_int32_t) :: field_id, pad_id
        type(lr_operand_desc_t) :: dest, tmp, field_ptr, pad_ptr, value
        type(lr_operand_desc_t) :: data_ptr, length, args(6), size_op, blen
        character(len=64) :: fmt_name

        if (.not. allocated(node%format_spec)) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'internal write requires a literal format', error_msg)
            return
        end if
        call parse_single_edit_descriptor(node%format_spec, kind_char, printf_fmt, &
                                          error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('internal write', node%line, &
                                           node%column, trim(error_msg), error_msg)
            return
        end if
        if (.not. allocated(node%arg_indices)) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'internal write requires one value', error_msg)
            return
        end if
        if (size(node%arg_indices) /= 1) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'only a single-value internal write is supported', &
                error_msg)
            return
        end if

        symbol_index = find_symbol(context, trim(node%unit_spec))
        if (context%symbols(symbol_index)%is_deferred_character) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'internal write target must be a fixed-length '// &
                'character variable', error_msg)
            return
        end if
        buflen = context%symbols(symbol_index)%character_length

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), dest, &
                error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), tmp, &
                error_msg)) return

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.iwf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         printf_fmt, field_id, error_msg)
        if (len_trim(error_msg) > 0) return
        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.iwpad.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         '%-*.*s', pad_id, error_msg)
        if (len_trim(error_msg) > 0) return
        field_ptr = printf_format_ptr(context%session, field_id)
        pad_ptr = printf_format_ptr(context%session, pad_id)
        size_op = i64_immediate(context%session, int(buflen + 1, c_int64_t))

        ! snprintf(tmp, buflen+1, field_fmt, value)
        args(1) = tmp
        args(2) = size_op
        args(3) = field_ptr
        if (kind_char == 'I') then
            call lower_i32_expression(arena, node%arg_indices(1), context, value, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            args(4) = value
        else
            call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                    length, error_msg)
            if (len_trim(error_msg) > 0) return
            args(4) = data_ptr
        end if
        if (.not. emit_snprintf(context%session, args(1:4), error_msg)) return

        ! snprintf(dest, buflen+1, "%-*.*s", buflen, buflen, tmp): left-justify
        ! the field, blank-pad and truncate to buflen.
        blen = i32_immediate(context%session, int(buflen, c_int64_t))
        args(1) = dest
        args(2) = size_op
        args(3) = pad_ptr
        args(4) = blen
        args(5) = blen
        args(6) = tmp
        if (.not. emit_snprintf(context%session, args(1:6), error_msg)) return

        context%symbols(symbol_index)%value = dest
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_internal_write
