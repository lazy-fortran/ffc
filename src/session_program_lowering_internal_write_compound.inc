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

        symbol_index = find_symbol_compat(context, trim(node%unit_spec))
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
            ! Value exhaustion is diagnosed per data descriptor: control
            ! descriptors such as nX consume no value and may trail the list.
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
        ! Format one edit descriptor into a scratch buffer and strcat it onto
        ! dest_tmp; advances pos past the descriptor and item_index past every
        ! value the descriptor consumes. A leading decimal prefix is the
        ! descriptor repeat count (for nX it is the blank count).
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        integer, intent(inout) :: item_index
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prefix
        character(len=:), allocatable :: printf_fmt
        character :: kind_char
        integer :: repeat_count, i
        integer :: width_value, precision_value, exponent_digits

        call set_empty(error_msg)
        call parse_decimal_digits(format_body, pos, prefix)
        repeat_count = 1
        if (len(prefix) > 0) then
            call read_decimal_value(prefix, repeat_count, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (pos > len_trim(format_body)) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, 'dangling repeat count in compound format', &
                error_msg)
            return
        end if
        kind_char = format_body(pos:pos)
        if (kind_char >= 'a' .and. kind_char <= 'z') &
            kind_char = char(ichar(kind_char) - 32)
        pos = pos + 1

        ! nX advances the cursor by n blanks and consumes no value.
        if (kind_char == 'X') then
            call append_internal_blanks(context, repeat_count, dest_tmp, &
                                        error_msg)
            return
        end if

        if (kind_char == 'E') then
            call parse_internal_e_descriptor(format_body, pos, width_value, &
                                             precision_value, &
                                             exponent_digits, error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('internal write', node%line, &
                                               node%column, trim(error_msg), &
                                               error_msg)
                return
            end if
            do i = 1, repeat_count
                if (item_index > size(node%arg_indices)) then
                    call unsupported_feature_error('internal write', node%line, &
                        node%column, 'internal write has more format '// &
                        'descriptors than values', error_msg)
                    return
                end if
                call append_internal_e_field(arena, node%arg_indices(item_index), &
                                             context, width_value, &
                                             precision_value, exponent_digits, &
                                             dest_tmp, error_msg)
                if (len_trim(error_msg) > 0) return
                item_index = item_index + 1
            end do
            return
        end if

        call parse_internal_ia_descriptor(format_body, pos, kind_char, &
                                          printf_fmt, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('internal write', node%line, &
                                           node%column, trim(error_msg), error_msg)
            return
        end if
        do i = 1, repeat_count
            if (item_index > size(node%arg_indices)) then
                call unsupported_feature_error('internal write', node%line, &
                    node%column, 'internal write has more format '// &
                    'descriptors than values', error_msg)
                return
            end if
            call append_internal_ia_field(arena, node%arg_indices(item_index), &
                                          context, kind_char, printf_fmt, &
                                          dest_tmp, error_msg)
            if (len_trim(error_msg) > 0) return
            item_index = item_index + 1
        end do
    end subroutine lower_compound_write_descriptor

    subroutine parse_internal_ia_descriptor(format_body, pos, kind_char, &
                                            printf_fmt, error_msg)
        ! Iw[.m] -> %wd (I0 -> %d), A[w] -> %[w]s.
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        character, intent(in) :: kind_char
        character(len=:), allocatable, intent(out) :: printf_fmt
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: width
        integer :: width_value
        character(len=32) :: width_text

        call set_empty(error_msg)
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
            error_msg = 'unsupported edit descriptor in compound format: '// &
                        kind_char
        end select
    end subroutine parse_internal_ia_descriptor

    subroutine parse_internal_e_descriptor(format_body, pos, width_value, &
                                           precision_value, exponent_digits, &
                                           error_msg)
        ! Ew.d[Ee]: field width w, d fraction digits, optional exponent digit
        ! count e (default 2). ES/EN are not handled here.
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        integer, intent(out) :: width_value, precision_value, exponent_digits
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: digits

        call set_empty(error_msg)
        width_value = 0
        precision_value = 0
        exponent_digits = 2
        call parse_decimal_digits(format_body, pos, digits)
        if (len(digits) == 0) then
            error_msg = 'E edit descriptor requires width and precision'
            return
        end if
        if (pos > len_trim(format_body)) then
            error_msg = 'E edit descriptor requires width and precision'
            return
        end if
        if (format_body(pos:pos) /= '.') then
            error_msg = 'E edit descriptor requires width and precision'
            return
        end if
        call read_decimal_value(digits, width_value, error_msg)
        if (len_trim(error_msg) > 0) return
        pos = pos + 1
        call parse_decimal_digits(format_body, pos, digits)
        if (len(digits) == 0) then
            error_msg = 'E edit descriptor requires precision'
            return
        end if
        call read_decimal_value(digits, precision_value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (pos > len_trim(format_body)) return
        if (format_body(pos:pos) /= 'E' .and. format_body(pos:pos) /= 'e') return
        pos = pos + 1
        call parse_decimal_digits(format_body, pos, digits)
        if (len(digits) == 0) then
            error_msg = 'E edit descriptor requires exponent digits after Ee'
            return
        end if
        call read_decimal_value(digits, exponent_digits, error_msg)
    end subroutine parse_internal_e_descriptor

    subroutine append_internal_blanks(context, count, dest_tmp, error_msg)
        ! Append count blanks to the accumulator: the nX cursor advance in an
        ! internal write that only ever moves forward.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: count
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: blanks_ptr, strcat_result
        character(len=64) :: fmt_name
        character(len=:), allocatable :: blanks
        integer :: i

        call set_empty(error_msg)
        if (count <= 0) return
        allocate (character(len=count) :: blanks)
        do i = 1, count
            blanks(i:i) = ' '
        end do
        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'iwx.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         blanks, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        blanks_ptr = printf_format_ptr(context%session, fmt_id)
        if (.not. emit_ptr_call(context%session, 'strcat', &
                                [dest_tmp, blanks_ptr], strcat_result, &
                                error_msg)) return
        call set_empty(error_msg)
    end subroutine append_internal_blanks

    subroutine append_internal_e_field(arena, node_index, context, width, &
                                       precision, exponent_digits, dest_tmp, &
                                       error_msg)
        ! Build the Ew.dEe field through the shared .ffc.fmt_e_en runtime helper
        ! (the same one formatted print uses) and append it to the accumulator.
        use liric_session_format_bindings, only: emit_e_en_format_call
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, width, precision, exponent_digits
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, value_f64, field, strcat_result

        if (is_f32_expression(arena, node_index, context)) then
            call lower_f32_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_to_f64(context%session, value, value_f64, &
                                            error_msg)) return
            value = value_f64
        else
            call lower_f64_expression(arena, node_index, context, value, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 256_c_int64_t), field, &
                error_msg)) return
        if (.not. emit_e_en_format_call(context%session, value, 0, precision, &
                                        width, field, error_msg, &
                                        exp_digits=exponent_digits)) return
        if (.not. emit_ptr_call(context%session, 'strcat', [dest_tmp, field], &
                                strcat_result, error_msg)) return
        call set_empty(error_msg)
    end subroutine append_internal_e_field

    subroutine append_internal_ia_field(arena, node_index, context, kind_char, &
                                        printf_fmt, dest_tmp, error_msg)
        ! snprintf one I/A value into a scratch buffer and append it.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character, intent(in) :: kind_char
        character(len=*), intent(in) :: printf_fmt
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, scratch, value, data_ptr, length
        type(lr_operand_desc_t) :: args(4), strcat_result
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'iwcf.', context%string_literal_count)
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
            call lower_i32_expression(arena, node_index, context, value, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            args(4) = value
        else
            call char_expr_operands(arena, node_index, context, data_ptr, &
                                    length, error_msg)
            if (len_trim(error_msg) > 0) return
            args(4) = data_ptr
        end if
        if (.not. emit_snprintf(context%session, args, error_msg)) return

        if (.not. emit_ptr_call(context%session, 'strcat', &
                                [dest_tmp, scratch], strcat_result, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine append_internal_ia_field
