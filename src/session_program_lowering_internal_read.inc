    logical function is_internal_read(node, context)
        ! True when read's unit is a character variable rather than a unit
        ! number or '*'.
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_internal_read = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*') return
        symbol_index = find_symbol(context, trim(node%unit_spec))
        if (symbol_index <= 0) return
        is_internal_read = &
            context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
    end function is_internal_read

    subroutine lower_internal_read(arena, node, context, error_msg)
        ! read (buf, '(I0)'/'(Iw)') value: parse an integer from the character
        ! variable buf with sscanf and store it into the integer target.
        ! read (buf, *) value: list-directed read of an integer, real, or
        ! character scalar target from the character variable buf.
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: printf_fmt
        character :: kind_char
        character(len=:), allocatable :: target_name
        integer :: buf_index, target_index
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: slot, fmt_ptr, value, args(3)
        character(len=64) :: fmt_name

        if (.not. allocated(node%format_spec)) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read requires a literal format', error_msg)
            return
        end if
        if (.not. allocated(node%var_indices) .or. size(node%var_indices) /= 1) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'only a single-value internal read is supported', &
                error_msg)
            return
        end if

        buf_index = find_symbol(context, trim(node%unit_spec))
        if (.not. context%symbols(buf_index)%has_character_value) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read source must be assigned first', &
                error_msg)
            return
        end if

        call identifier_name(arena, node%var_indices(1), target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        target_index = find_symbol(context, target_name)
        if (target_index <= 0) then
            error_msg = 'internal read target was not declared: '//trim(target_name)
            return
        end if
        if (context%symbols(target_index)%is_array .or. &
            context%symbols(target_index)%is_derived) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read target must be a scalar', &
                error_msg)
            return
        end if

        if (trim(adjustl(node%format_spec)) == '*') then
            call lower_list_directed_internal_read(context, buf_index, &
                                                    target_index, node%line, &
                                                    node%column, error_msg)
            return
        end if

        call parse_single_edit_descriptor(node%format_spec, kind_char, printf_fmt, &
                                          error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('internal read', node%line, &
                                           node%column, trim(error_msg), error_msg)
            return
        end if
        if (kind_char /= 'I') then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read only supports integer edit '// &
                'descriptors', error_msg)
            return
        end if
        if (context%symbols(target_index)%value_kind /= VALUE_I32) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read target must be an integer scalar', &
                error_msg)
            return
        end if

        if (.not. emit_i32_alloca(context%session, slot, error_msg)) return
        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.irf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), '%d', &
                                         fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        ! sscanf(buf_data, "%d", &slot)
        args(1) = context%symbols(buf_index)%value
        args(2) = fmt_ptr
        args(3) = slot
        if (.not. emit_sscanf(context%session, args, error_msg)) return

        if (.not. emit_i32_load(context%session, slot, value, error_msg)) return
        context%symbols(target_index)%value = value
        call set_empty(error_msg)
    end subroutine lower_internal_read

    subroutine lower_list_directed_internal_read(context, buf_index, &
                                                 target_index, line, col, &
                                                 error_msg)
        ! read (buf, *) value: sscanf the character buffer with a conversion
        ! chosen from the target's kind (integer/real via file_read_slot_for_
        ! kind's numeric slots, character via a blank-padded token read).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: buf_index, target_index, line, col
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: scanf_fmt
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, slot, args(3)
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        if (context%symbols(target_index)%value_kind == VALUE_CHARACTER) then
            call lower_list_directed_internal_read_char(context, buf_index, &
                                                         target_index, line, &
                                                         col, error_msg)
            return
        end if

        call file_read_slot_for_kind(context, target_index, scanf_fmt, slot, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        if (len_trim(scanf_fmt) == 0) then
            call unsupported_feature_error('internal read', line, col, &
                'list-directed internal read supports integer, real, and '// &
                'character scalars only', error_msg)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.ilrf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = context%symbols(buf_index)%value
        args(2) = fmt_ptr
        args(3) = slot
        if (.not. emit_sscanf(context%session, args, error_msg)) return

        select case (context%symbols(target_index)%value_kind)
        case (VALUE_I32)
            if (.not. emit_i32_load(context%session, slot, &
                                    context%symbols(target_index)%value, &
                                    error_msg)) return
        case (VALUE_I64)
            if (.not. emit_i64_load(context%session, slot, &
                                    context%symbols(target_index)%value, &
                                    error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, slot, &
                                          context%symbols(target_index)%value, &
                                          error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, slot, &
                                          context%symbols(target_index)%value, &
                                          error_msg)) return
        end select
        call set_empty(error_msg)
    end subroutine lower_list_directed_internal_read

    subroutine lower_list_directed_internal_read_char(context, buf_index, &
                                                       target_index, line, &
                                                       col, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: buf_index, target_index, line, col
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: buflen
        type(lr_operand_desc_t) :: tmp, dest, fmt_ptr, args(3)
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name
        character(len=16) :: width_text
        character(len=:), allocatable :: scanf_fmt

        call set_empty(error_msg)
        if (context%symbols(target_index)%is_deferred_character) then
            call unsupported_feature_error('internal read', line, col, &
                'internal read target must be a fixed-length character '// &
                'variable', error_msg)
            return
        end if
        buflen = context%symbols(target_index)%character_length

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), &
                tmp, error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), &
                dest, error_msg)) return

        write (width_text, '(I0)') buflen
        scanf_fmt = ' %'//trim(width_text)//'s'
        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.ilrfc.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = context%symbols(buf_index)%value
        args(2) = fmt_ptr
        args(3) = tmp
        if (.not. emit_sscanf(context%session, args, error_msg)) return

        call emit_blank_pad_string(context, buflen, tmp, dest, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(target_index)%value = dest
        context%symbols(target_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_list_directed_internal_read_char
