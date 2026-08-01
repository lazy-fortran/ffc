    ! Fixed-width A and L edit descriptors on a formatted file-unit READ
    ! (#434). The numeric path in read_ops lets fscanf skip whitespace, which
    ! cannot honour column-oriented A/L fields. Here each field is taken from
    ! the current record with fscanf("%<w>[^\n]"), so blanks inside the field
    ! are preserved and the record cursor advances by exactly the field width.

    logical function format_has_al_descriptor(format_spec)
        ! True when an explicit format contains an A or L edit descriptor.
        character(len=*), intent(in) :: format_spec
        character(len=:), allocatable :: body
        integer :: i
        character :: ch

        format_has_al_descriptor = .false.
        if (trim(format_spec) == '*') return
        call normalize_format_body(format_spec, body)
        do i = 1, len(body)
            ch = body(i:i)
            if (ch >= 'a' .and. ch <= 'z') ch = char(ichar(ch) - 32)
            if (ch == 'A' .or. ch == 'L') then
                format_has_al_descriptor = .true.
                return
            end if
        end do
    end function format_has_al_descriptor

    subroutine next_al_descriptor(body, pos, repeat_left, code, width, &
                                  ok, error_msg)
        ! Pull the next A or L descriptor out of a normalized format body.
        ! repeat_left carries an unconsumed repeat count between calls; code
        ! and width describe the descriptor. ok is false at end of format.
        character(len=*), intent(in) :: body
        integer, intent(inout) :: pos
        integer, intent(inout) :: repeat_left
        character, intent(inout) :: code
        integer, intent(inout) :: width
        logical, intent(out) :: ok
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: repeat_count
        character :: ch

        call set_empty(error_msg)
        ok = .false.
        if (repeat_left > 0) then
            repeat_left = repeat_left - 1
            ok = .true.
            return
        end if
        do while (pos <= len(body))
            ch = body(pos:pos)
            if (ch /= ' ' .and. ch /= ',') exit
            pos = pos + 1
        end do
        if (pos > len(body)) return
        call read_al_int(body, pos, repeat_count)
        if (repeat_count <= 0) repeat_count = 1
        if (pos > len(body)) then
            error_msg = 'formatted read format ends in a repeat count'
            return
        end if
        code = body(pos:pos)
        if (code >= 'a' .and. code <= 'z') code = char(ichar(code) - 32)
        if (code /= 'A' .and. code /= 'L') then
            error_msg = 'formatted A/L file-unit read supports only A and L '// &
                        'edit descriptors'
            return
        end if
        pos = pos + 1
        call read_al_int(body, pos, width)
        repeat_left = repeat_count - 1
        ok = .true.
    end subroutine next_al_descriptor

    subroutine read_al_int(text, pos, value)
        ! Read a non-negative integer at pos; value is 0 when no digit follows.
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos
        integer, intent(out) :: value

        value = 0
        do while (pos <= len(text))
            if (text(pos:pos) < '0' .or. text(pos:pos) > '9') exit
            value = value * 10 + (ichar(text(pos:pos)) - ichar('0'))
            pos = pos + 1
        end do
    end subroutine read_al_int

    subroutine lower_read_file_al(arena, node, fp, context, error_msg)
        ! Walk the item list against the format's A/L descriptors, reading one
        ! fixed-width field per item, then advance past the rest of the record.
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        type(lr_operand_desc_t), intent(in) :: fp
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: body
        type(lr_operand_desc_t) :: err_slot, err_value
        integer :: pos, repeat_left, width, i
        character :: code
        logical :: ok

        call set_empty(error_msg)
        call normalize_format_body(node%format_spec, body)
        call emit_al_error_slot(context, err_slot, error_msg)
        if (len_trim(error_msg) > 0) return
        pos = 1
        repeat_left = 0
        code = ' '
        width = 0
        if (allocated(node%var_indices)) then
            do i = 1, size(node%var_indices)
                call next_al_descriptor(body, pos, repeat_left, code, width, &
                                        ok, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. ok) then
                    error_msg = 'formatted read format has fewer edit '// &
                                'descriptors than input items'
                    return
                end if
                call lower_read_al_item(arena, node%var_indices(i), fp, code, &
                                        width, err_slot, node%line, &
                                        node%column, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        call emit_al_record_advance(context, fp, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_i32_load_operand(context, err_slot, err_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call store_read_iostat_from_eof(node, fp, context, error_msg, err_value)
        if (len_trim(error_msg) > 0) return
        call branch_on_eof_to_end_label(node, fp, context, error_msg)
    end subroutine lower_read_file_al

    subroutine emit_al_error_slot(context, err_slot, error_msg)
        ! Allocate and zero the per-statement field-error code slot.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i32_store
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: err_slot
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. emit_i32_alloca(context%session, err_slot, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                                 i32_immediate(context%session, 0_c_int64_t), &
                                 err_slot, error_msg)) return
    end subroutine emit_al_error_slot

    subroutine emit_i32_load_operand(context, slot, value, error_msg)
        use liric_session_memory_bindings, only: emit_i32_load
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: slot
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. emit_i32_load(context%session, slot, value, error_msg)) return
    end subroutine emit_i32_load_operand

    subroutine emit_al_record_advance(context, fp, error_msg)
        ! Consume the remainder of the current record plus its newline, so the
        ! next READ starts at column one of the next record.
        use liric_session_complex_print_bindings, only: emit_extern_i32_call
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fmt_ptr, args(2), fp_arg(1), unused
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'alskip.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         '%*[^'//achar(10)//']', fmt_id, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)
        args(1) = fp
        args(2) = fmt_ptr
        if (.not. emit_fscanf(context%session, args, error_msg)) return
        fp_arg(1) = fp
        if (.not. emit_extern_i32_call(context%session, 'fgetc', fp_arg, &
                                       unused, error_msg)) return
    end subroutine emit_al_record_advance

    subroutine emit_al_field_buffer(context, fp, width, buf, error_msg)
        ! Read up to width characters of the current record into a fresh
        ! null-terminated buffer, stopping at the record end.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        integer, intent(in) :: width
        type(lr_operand_desc_t), intent(out) :: buf
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fmt_ptr, args(3)
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name
        character(len=16) :: width_text

        call set_empty(error_msg)
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(width + 1, c_int64_t)), &
                buf, error_msg)) return
        ! An empty field leaves the scanset unmatched, so pre-terminate.
        if (.not. emit_liric_store_char_byte(context%session, buf, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return

        write (width_text, '(I0)') width
        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'alfield.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
            '%'//trim(width_text)//'[^'//achar(10)//']', fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)
        args(1) = fp
        args(2) = fmt_ptr
        args(3) = buf
        if (.not. emit_fscanf(context%session, args, error_msg)) return
    end subroutine emit_al_field_buffer

    subroutine lower_read_al_item(arena, node_index, fp, code, width, err_slot, &
                                  stmt_line, stmt_col, context, error_msg)
        ! Read one A or L field into the named scalar target.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lr_operand_desc_t), intent(in) :: fp
        character, intent(in) :: code
        integer, intent(in) :: width
        type(lr_operand_desc_t), intent(in) :: err_slot
        integer, intent(in) :: stmt_line, stmt_col
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name
        integer :: sym

        call set_empty(error_msg)
        call identifier_name(arena, node_index, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol_compat(context, var_name)
        if (sym <= 0) then
            error_msg = 'read target was not declared: '//trim(var_name)
            return
        end if

        if (code == 'A') then
            if (context%symbols(sym)%value_kind /= VALUE_CHARACTER) then
                call unsupported_feature_error('read from file unit', &
                    stmt_line, stmt_col, 'A editing needs a character '// &
                    'target (#434)', error_msg)
                return
            end if
            call lower_read_al_char(context, fp, sym, width, stmt_line, &
                                    stmt_col, error_msg)
        else
            if (context%symbols(sym)%value_kind /= VALUE_LOGICAL) then
                call unsupported_feature_error('read from file unit', &
                    stmt_line, stmt_col, 'L editing needs a logical '// &
                    'target (#434)', error_msg)
                return
            end if
            call lower_read_al_logical(context, fp, sym, width, err_slot, &
                                       error_msg)
        end if
    end subroutine lower_read_al_item

    subroutine lower_read_al_char(context, fp, sym, width, stmt_line, &
                                  stmt_col, error_msg)
        ! A editing: a field narrower than the target is left-justified and
        ! blank-padded; a wider field keeps its rightmost characters.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        integer, intent(in) :: sym, width, stmt_line, stmt_col
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: buf, padded, src, dest
        integer :: buflen, field_width, span

        call set_empty(error_msg)
        if (context%symbols(sym)%is_deferred_character) then
            call unsupported_feature_error('read from file unit', stmt_line, &
                stmt_col, 'file-unit read target must be a fixed-length '// &
                'character variable (#434)', error_msg)
            return
        end if
        buflen = context%symbols(sym)%character_length
        field_width = width
        if (field_width <= 0) field_width = buflen
        span = max(field_width, buflen)

        call emit_al_field_buffer(context, fp, field_width, buf, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(span + 1, c_int64_t)), &
                padded, error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), &
                dest, error_msg)) return
        call emit_blank_pad_string(context, span, buf, padded, error_msg)
        if (len_trim(error_msg) > 0) return
        if (field_width > buflen) then
            if (.not. emit_ptr_offset(context%session, padded, &
                    int(field_width - buflen, c_int64_t), src, error_msg)) return
        else
            src = padded
        end if
        call emit_blank_pad_string(context, buflen, src, dest, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(sym)%value = dest
        context%symbols(sym)%has_character_value = .true.
    end subroutine lower_read_al_char

    subroutine lower_read_al_logical(context, fp, sym, width, err_slot, &
                                     error_msg)
        ! L editing: skip leading blanks and an optional decimal point, then
        ! accept T or F in either case. Anything else is a bad-value IOSTAT.
        use liric_session_complex_print_bindings, only: emit_extern_i32_call
        use liric_session_memory_bindings, only: emit_i32_store
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        integer, intent(in) :: sym, width
        type(lr_operand_desc_t), intent(in) :: err_slot
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: buf, skip_set, span_args(2), skipped
        type(lr_operand_desc_t) :: byte_val, folded, is_true, is_false
        integer(c_int32_t) :: set_id
        integer(c_int32_t) :: true_blk, not_true_blk, false_blk, bad_blk, done_blk
        character(len=64) :: set_name
        integer :: field_width

        call set_empty(error_msg)
        field_width = width
        if (field_width <= 0) field_width = 1
        call emit_al_field_buffer(context, fp, field_width, buf, error_msg)
        if (len_trim(error_msg) > 0) return

        context%string_literal_count = context%string_literal_count + 1
        set_name = ffc_unit_global_name( &
            context, 'alskipset.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(set_name), &
                                         ' .', set_id, error_msg)
        if (len_trim(error_msg) > 0) return
        skip_set = printf_format_ptr(context%session, set_id)
        span_args(1) = buf
        span_args(2) = skip_set
        if (.not. emit_extern_i32_call(context%session, 'strspn', span_args, &
                                       skipped, error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, buf, skipped, &
                                            byte_val, error_msg)) return
        ! Fold ASCII case: 'T'|32 == 't', 'F'|32 == 'f'.
        if (.not. emit_i32_binary(context%session, LR_OP_OR, byte_val, &
                i32_immediate(context%session, 32_c_int64_t), folded, &
                error_msg)) return

        true_blk = create_liric_block(context%session)
        not_true_blk = create_liric_block(context%session)
        false_blk = create_liric_block(context%session)
        bad_blk = create_liric_block(context%session)
        done_blk = create_liric_block(context%session)

        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, folded, &
                i32_immediate(context%session, 116_c_int64_t), is_true, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, is_true, true_blk, &
                                    not_true_blk, error_msg)) return

        if (.not. set_liric_block(context%session, true_blk, error_msg)) return
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, 1_c_int64_t), &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, done_blk, error_msg)) return

        if (.not. set_liric_block(context%session, not_true_blk, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, folded, &
                i32_immediate(context%session, 102_c_int64_t), is_false, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, is_false, false_blk, &
                                    bad_blk, error_msg)) return

        if (.not. set_liric_block(context%session, false_blk, error_msg)) return
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, 0_c_int64_t), &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, done_blk, error_msg)) return

        if (.not. set_liric_block(context%session, bad_blk, error_msg)) return
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, 0_c_int64_t), &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 5010_c_int64_t), err_slot, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, done_blk, error_msg)) return

        if (.not. set_liric_block(context%session, done_blk, error_msg)) return
        context%current_block_id = done_blk
        context%current_block_terminated = .false.
        call reload_i32_symbol(context, sym, error_msg)
    end subroutine lower_read_al_logical
