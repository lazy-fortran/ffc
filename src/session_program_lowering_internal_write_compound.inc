    subroutine lower_compound_internal_write(arena, node, context, format_body, &
                                             error_msg)
        ! write (buf, '(d1,d2,...)') v1, v2, ...: walk the comma-separated I/A
        ! edit descriptors in order, formatting each value into a scratch
        ! buffer and appending it to a growing accumulator, then blank-pad and
        ! truncate the accumulator into buf's declared length.
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index, buflen, item_index, item_total, pos
        type(lr_operand_desc_t) :: dest_tmp, dest

        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices)) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'internal write requires at least one value', &
                error_msg)
            return
        end if
        item_total = size(node%arg_indices)

        symbol_index = find_symbol(context, trim(node%unit_spec))
        if (context%symbols(symbol_index)%is_deferred_character) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'internal write target must be a fixed-length '// &
                'character variable', error_msg)
            return
        end if
        buflen = context%symbols(symbol_index)%character_length

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(max(buflen*4, 256), &
                                                    c_int64_t)), &
                dest_tmp, error_msg)) return
        if (.not. emit_i8_store(context%session, &
                i8_immediate(context%session, 0_c_int64_t), dest_tmp, &
                error_msg)) return

        pos = 1
        item_index = 1
        do
            call skip_format_separators(format_body, pos)
            if (pos > len_trim(format_body)) exit
            if (item_index > item_total) then
                call unsupported_feature_error('internal write', node%line, &
                    node%column, 'internal write has more format '// &
                    'descriptors than values', error_msg)
                return
            end if
            call lower_compound_write_descriptor(arena, node, context, &
                format_body, pos, item_index, dest_tmp, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (item_index <= item_total) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'internal write has more values than format '// &
                'descriptors', error_msg)
            return
        end if

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), &
                dest, error_msg)) return
        call emit_blank_pad_string(context, buflen, dest_tmp, dest, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(symbol_index)%value = dest
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_compound_internal_write

    subroutine lower_compound_write_descriptor(arena, node, context, &
                                               format_body, pos, item_index, &
                                               dest_tmp, error_msg)
        ! Format one I/A edit descriptor's value into a scratch buffer and
        ! strcat it onto dest_tmp; advances pos past the descriptor and
        ! item_index past the consumed value.
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        integer, intent(inout) :: item_index
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: width
        character(len=:), allocatable :: printf_fmt
        character :: kind_char
        integer :: width_value
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, scratch, value, data_ptr, length
        type(lr_operand_desc_t) :: args(4), strcat_result
        character(len=64) :: fmt_name
        character(len=32) :: width_text

        call set_empty(error_msg)
        kind_char = format_body(pos:pos)
        if (kind_char >= 'a' .and. kind_char <= 'z') &
            kind_char = char(ichar(kind_char) - 32)
        pos = pos + 1

        select case (kind_char)
        case ('I')
            call parse_decimal_digits(format_body, pos, width)
            call skip_dot_modifier(format_body, pos)
            if (len(width) == 0 .or. width == '0') then
                printf_fmt = '%d'
            else
                call read_decimal_value(width, width_value, error_msg)
                if (len_trim(error_msg) > 0) return
                write (width_text, '(I0)') width_value
                printf_fmt = '%'//trim(width_text)//'d'
            end if
        case ('A')
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                printf_fmt = '%s'
            else
                call read_decimal_value(width, width_value, error_msg)
                if (len_trim(error_msg) > 0) return
                write (width_text, '(I0)') width_value
                printf_fmt = '%'//trim(width_text)//'s'
            end if
        case default
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'unsupported edit descriptor in compound '// &
                'format: '//kind_char, error_msg)
            return
        end select

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.iwcf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         printf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 128_c_int64_t), scratch, &
                error_msg)) return

        args(1) = scratch
        args(2) = i64_immediate(context%session, 128_c_int64_t)
        args(3) = fmt_ptr
        if (kind_char == 'I') then
            call lower_i32_expression(arena, node%arg_indices(item_index), &
                                      context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            args(4) = value
        else
            call char_expr_operands(arena, node%arg_indices(item_index), &
                                    context, data_ptr, length, error_msg)
            if (len_trim(error_msg) > 0) return
            args(4) = data_ptr
        end if
        if (.not. emit_snprintf(context%session, args, error_msg)) return

        if (.not. emit_ptr_call(context%session, 'strcat', &
                [dest_tmp, scratch], strcat_result, error_msg)) return

        item_index = item_index + 1
        call set_empty(error_msg)
    end subroutine lower_compound_write_descriptor
