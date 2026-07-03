    ! File-unit READ and REWIND lowering (#247 file I/O).
    !
    ! A READ from an opened unit loads the unit's FILE* and emits one fscanf per
    ! list-directed scalar target, mirroring the stdin scanf path in write_ops.
    ! REWIND maps to the C runtime rewind(FILE*).

    logical function is_file_unit_read(node, context)
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer :: sym, ios, unit_number

        is_file_unit_read = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*' .or. &
            trim(node%unit_spec) == '5') return
        read (node%unit_spec, *, iostat=ios) unit_number
        if (ios == 0) then
            ! A numeric unit can be implicitly preconnected on first use.
            is_file_unit_read = .true.
            return
        end if
        sym = find_symbol(context, trim(node%unit_spec))
        if (sym > 0) is_file_unit_read = context%symbols(sym)%is_file_unit .or. &
                                         context%symbols(sym)%has_unit_const
    end function is_file_unit_read

    function file_unit_pseudo_name(unit_number) result(name)
        integer, intent(in) :: unit_number
        character(len=24) :: name
        write (name, '(A,I0)') '.unit.', unit_number
    end function file_unit_pseudo_name

    subroutine lower_read_file(arena, node, context, error_msg)
        ! List-directed READ from an opened file unit into scalar targets.
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp
        integer :: i

        call set_empty(error_msg)
        ! List-directed (*) and explicit edit-descriptor formats both reduce to
        ! a per-item fscanf with a kind-driven conversion (%d / %lf / ...).
        ! Field widths are not enforced; whitespace- and newline-separated
        ! records read correctly, which covers the formatted-read corpus.
        if (allocated(node%format_spec)) then
            call check_read_format_supported(node%format_spec, node%line, &
                                             node%column, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call load_unit_file_ptr(node%unit_spec, context, fp, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = ''
            call unsupported_feature_error('read from file unit', node%line, &
                node%column, 'unit not opened in same scope (#247)', error_msg)
            return
        end if
        if (.not. allocated(node%var_indices)) then
            call store_read_iostat_success(node, context, error_msg)
            return
        end if
        do i = 1, size(node%var_indices)
            call lower_read_file_item(arena, node%var_indices(i), fp, node%line, &
                                      node%column, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call store_read_iostat_from_eof(node, fp, context, error_msg)
        if (len_trim(error_msg) > 0) return
        call branch_on_eof_to_end_label(node, fp, context, error_msg)
    end subroutine lower_read_file

    subroutine branch_on_eof_to_end_label(node, fp, context, error_msg)
        ! Honour a READ end= label: when the read reaches end-of-file, transfer
        ! control to the named statement label instead of falling through.
        ! Requires a labeled program body, where the label owns a LIRIC block.
        use liric_session_complex_print_bindings, only: emit_extern_i32_call
        type(read_statement_node), intent(in) :: node
        type(lr_operand_desc_t), intent(in) :: fp
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: end_label
        type(lr_operand_desc_t) :: eof_flag, at_eof, fp_arg(1)
        integer(c_int32_t) :: target_block, continue_block
        logical :: found

        call set_empty(error_msg)
        if (.not. allocated(node%io_control_list)) return
        call io_control_value(node%io_control_list, 'end', end_label)
        if (len_trim(end_label) == 0) return
        if (.not. context%in_labeled_body) then
            call unsupported_feature_error('read from file unit', node%line, &
                node%column, 'end= label is only lowered inside a labeled '// &
                'program body (#247)', error_msg)
            return
        end if
        target_block = find_label_block(context, trim(end_label), found)
        if (.not. found) then
            error_msg = 'read end= targets undefined statement label: '// &
                        trim(end_label)
            return
        end if

        fp_arg(1) = fp
        if (.not. emit_extern_i32_call(context%session, 'feof', fp_arg, eof_flag, &
                                       error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, eof_flag, &
                i32_immediate(context%session, 0_c_int64_t), at_eof, &
                error_msg)) return
        continue_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, at_eof, target_block, &
                                     continue_block, error_msg)) return
        if (.not. set_liric_block(context%session, continue_block, error_msg)) &
            return
        context%current_block_id = continue_block
        context%current_block_terminated = .false.
    end subroutine branch_on_eof_to_end_label

    subroutine store_read_iostat_success(node, context, error_msg)
        ! No item list: a READ with no targets always succeeds; iostat = 0.
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym

        call set_empty(error_msg)
        call read_iostat_symbol(node, context, sym, error_msg)
        if (len_trim(error_msg) > 0 .or. sym <= 0) return
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, 0_c_int64_t), &
                                  error_msg)
    end subroutine store_read_iostat_success

    subroutine store_read_iostat_from_eof(node, fp, context, error_msg)
        ! Map end-of-file to a Fortran iostat: feof(fp) == 0 -> 0 (success),
        ! otherwise -1 (end-of-file), matching gfortran's iostat_end.
        use liric_session_complex_print_bindings, only: emit_extern_i32_call
        type(read_statement_node), intent(in) :: node
        type(lr_operand_desc_t), intent(in) :: fp
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ok, eof_flag, fp_arg(1)
        integer(c_int32_t) :: ok_blk, eof_blk, done_blk
        integer :: sym

        call set_empty(error_msg)
        call read_iostat_symbol(node, context, sym, error_msg)
        if (len_trim(error_msg) > 0 .or. sym <= 0) return

        ! feof(fp) is set only once a read hits end-of-file. The vararg fscanf
        ! return is not reliably captured by the backend, so use feof to detect
        ! EOF: iostat = -1 (gfortran's iostat_end) at EOF, 0 otherwise.
        fp_arg(1) = fp
        if (.not. emit_extern_i32_call(context%session, 'feof', fp_arg, eof_flag, &
                                       error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, eof_flag, &
                i32_immediate(context%session, 0_c_int64_t), ok, error_msg)) return
        ok_blk = create_liric_block(context%session)
        eof_blk = create_liric_block(context%session)
        done_blk = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, ok, ok_blk, eof_blk, &
                                     error_msg)) return

        if (.not. set_liric_block(context%session, ok_blk, error_msg)) return
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, 0_c_int64_t), &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, done_blk, error_msg)) return

        if (.not. set_liric_block(context%session, eof_blk, error_msg)) return
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, -1_c_int64_t), &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, done_blk, error_msg)) return

        if (.not. set_liric_block(context%session, done_blk, error_msg)) return
        ! Both branches stored to the same slot; reload so later uses of the
        ! iostat variable see the merged value rather than a per-branch register.
        call reload_i32_symbol(context, sym, error_msg)
    end subroutine store_read_iostat_from_eof

    subroutine reload_i32_symbol(context, sym, error_msg)
        use liric_session_memory_bindings, only: emit_i32_load
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. context%symbols(sym)%has_address) return
        if (.not. emit_i32_load(context%session, context%symbols(sym)%address, &
                                context%symbols(sym)%value, error_msg)) return
    end subroutine reload_i32_symbol

    subroutine read_iostat_symbol(node, context, sym, error_msg)
        ! Resolve the read statement's iostat= target to a declared i32 symbol.
        ! The variable name lives in the io_control_list text (iostat=<name>);
        ! sym is 0 when the statement has no iostat= specifier.
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: sym
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name

        call set_empty(error_msg)
        sym = 0
        if (.not. allocated(node%io_control_list)) return
        call io_control_value(node%io_control_list, 'iostat', var_name)
        if (len_trim(var_name) == 0) return
        sym = find_symbol(context, var_name)
        if (sym <= 0) then
            error_msg = 'iostat target was not declared: '//trim(var_name)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_I32) then
            error_msg = 'iostat target must be a default integer: '//trim(var_name)
            sym = 0
            return
        end if
    end subroutine read_iostat_symbol

    subroutine io_control_value(control_list, keyword, value)
        ! Extract the value of "<keyword>=<value>" from an io_control_list text
        ! such as "iostat=ios, iomsg=msg". value is empty when absent.
        character(len=*), intent(in) :: control_list
        character(len=*), intent(in) :: keyword
        character(len=:), allocatable, intent(out) :: value
        character(len=:), allocatable :: lowered
        integer :: kw_pos, eq_pos, start, stop_pos, i

        value = ''
        lowered = lower_ascii(control_list)
        kw_pos = index(lowered, trim(keyword)//'=')
        if (kw_pos <= 0) return
        eq_pos = kw_pos + len_trim(keyword)
        start = eq_pos + 1
        do while (start <= len(control_list))
            if (control_list(start:start) /= ' ') exit
            start = start + 1
        end do
        stop_pos = start - 1
        do i = start, len(control_list)
            if (control_list(i:i) == ',' .or. control_list(i:i) == ' ') exit
            stop_pos = i
        end do
        if (stop_pos >= start) value = control_list(start:stop_pos)
    end subroutine io_control_value

    function lower_ascii(s) result(t)
        character(len=*), intent(in) :: s
        character(len=len(s)) :: t
        integer :: i, c
        do i = 1, len(s)
            c = ichar(s(i:i))
            if (c >= ichar('A') .and. c <= ichar('Z')) then
                t(i:i) = char(c + 32)
            else
                t(i:i) = s(i:i)
            end if
        end do
    end function lower_ascii

    subroutine assign_i32_to_symbol(context, sym, value, error_msg)
        ! Set an i32 symbol's current value and write it back to memory when the
        ! symbol is materialised, so later reads see the stored iostat.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i32_store
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        context%symbols(sym)%value = value
        if (.not. context%symbols(sym)%has_address) then
            if (.not. emit_i32_alloca(context%session, &
                                      context%symbols(sym)%address, error_msg)) return
            context%symbols(sym)%has_address = .true.
            context%symbols(sym)%is_reference = .true.
        end if
        if (.not. emit_i32_store(context%session, value, &
                                 context%symbols(sym)%address, error_msg)) return
    end subroutine assign_i32_to_symbol

    subroutine check_read_format_supported(format_spec, line, col, error_msg)
        ! Accept list-directed (*) and numeric edit descriptors (I/F/E/D/G plus
        ! repeat counts, widths, separators). Reject character/string/logical
        ! formats, which the kind-driven fscanf path cannot read yet.
        character(len=*), intent(in) :: format_spec
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: body
        integer :: i
        character :: ch

        call set_empty(error_msg)
        if (trim(format_spec) == '*') return
        call normalize_format_body(format_spec, body)
        do i = 1, len(body)
            ch = body(i:i)
            if (ch >= 'a' .and. ch <= 'z') ch = char(ichar(ch) - 32)
            select case (ch)
            case ('I', 'F', 'E', 'D', 'G', 'P', 'X', '.', ',', ' ', '/', &
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-')
                ! Numeric edit descriptor, repeat count, width, or separator.
            case default
                call unsupported_feature_error('read from file unit', line, &
                    col, 'formatted file-unit read supports numeric edit '// &
                    'descriptors only (#247)', error_msg)
                return
            end select
        end do
    end subroutine check_read_format_supported

    subroutine lower_read_file_item(arena, node_index, fp, stmt_line, stmt_col, &
                                    context, error_msg)
        ! Emit fscanf(fp, fmt, &slot) for one scalar target, then reload it.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i64_alloca, &
                                                  emit_i32_load, emit_i64_load
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lr_operand_desc_t), intent(in) :: fp
        integer, intent(in) :: stmt_line, stmt_col
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, scanf_fmt
        integer :: sym
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, slot, args(3)
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        call identifier_name(arena, node_index, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, var_name)
        if (sym <= 0) then
            error_msg = 'read target was not declared: '//trim(var_name)
            return
        end if

        if (context%symbols(sym)%value_kind == VALUE_CHARACTER) then
            call lower_read_file_char_item(context, fp, sym, stmt_line, &
                                           stmt_col, error_msg)
            return
        end if

        call file_read_slot_for_kind(context, sym, scanf_fmt, slot, error_msg)
        if (len_trim(error_msg) > 0) return
        if (len_trim(scanf_fmt) == 0) then
            call unsupported_feature_error('read from file unit', stmt_line, &
                stmt_col, 'list-directed file-unit read supports integer and '// &
                'real scalars only (#247)', error_msg)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.fscanf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = fp
        args(2) = fmt_ptr
        args(3) = slot
        if (.not. emit_fscanf(context%session, args, error_msg)) return

        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32)
            if (.not. emit_i32_load(context%session, slot, &
                                    context%symbols(sym)%value, error_msg)) return
        case (VALUE_I64)
            if (.not. emit_i64_load(context%session, slot, &
                                    context%symbols(sym)%value, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, slot, &
                                          context%symbols(sym)%value, &
                                          error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, slot, &
                                          context%symbols(sym)%value, &
                                          error_msg)) return
        end select
    end subroutine lower_read_file_item

    subroutine lower_read_file_char_item(context, fp, sym, stmt_line, &
                                         stmt_col, error_msg)
        ! List-directed READ of a fixed-length character scalar from an
        ! opened file unit: fscanf a whitespace-delimited token, then
        ! blank-pad and truncate it to the target's declared length.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        integer, intent(in) :: sym, stmt_line, stmt_col
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: buflen
        type(lr_operand_desc_t) :: tmp, dest, fmt_ptr, args(3)
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name
        character(len=16) :: width_text
        character(len=:), allocatable :: scanf_fmt

        call set_empty(error_msg)
        if (context%symbols(sym)%is_deferred_character) then
            call unsupported_feature_error('read from file unit', stmt_line, &
                stmt_col, 'file-unit read target must be a fixed-length '// &
                'character variable (#247)', error_msg)
            return
        end if
        buflen = context%symbols(sym)%character_length

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), &
                tmp, error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(buflen + 1, c_int64_t)), &
                dest, error_msg)) return

        write (width_text, '(I0)') buflen
        scanf_fmt = ' %'//trim(width_text)//'s'
        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.fscanfc.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = fp
        args(2) = fmt_ptr
        args(3) = tmp
        if (.not. emit_fscanf(context%session, args, error_msg)) return

        call emit_blank_pad_string(context, buflen, tmp, dest, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(sym)%value = dest
        context%symbols(sym)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_read_file_char_item

    subroutine file_read_slot_for_kind(context, sym, scanf_fmt, slot, error_msg)
        ! Resolve the scanf conversion and ensure the target has an alloca slot.
        ! scanf_fmt is empty for unsupported kinds.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i64_alloca
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        character(len=:), allocatable, intent(out) :: scanf_fmt
        type(lr_operand_desc_t), intent(out) :: slot
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        scanf_fmt = ''
        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32)
            scanf_fmt = ' %d'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i32_alloca(context%session, slot, error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_I64)
            scanf_fmt = ' %ld'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i64_alloca(context%session, slot, error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_F32)
            scanf_fmt = ' %f'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_liric_f32_alloca(context%session, slot, &
                                                error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_F64)
            scanf_fmt = ' %lf'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_liric_f64_alloca(context%session, slot, &
                                                error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        end select
    end subroutine file_read_slot_for_kind

    subroutine set_read_slot(context, sym, slot)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        type(lr_operand_desc_t), intent(in) :: slot
        context%symbols(sym)%address = slot
        context%symbols(sym)%has_address = .true.
        context%symbols(sym)%is_reference = .true.
    end subroutine set_read_slot

    subroutine lower_rewind(node, context, error_msg)
        type(rewind_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp, fa(1)

        call set_empty(error_msg)
        if (.not. allocated(node%unit_spec)) then
            call unsupported_feature_error('rewind statement', node%line, &
                node%column, 'missing unit specifier (#247)', error_msg)
            return
        end if
        call load_unit_file_ptr(node%unit_spec, context, fp, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = ''
            call unsupported_feature_error('rewind statement', node%line, &
                node%column, 'unit not opened in same scope (#247)', error_msg)
            return
        end if
        fa(1) = fp
        if (.not. emit_void_call(context%session, 'rewind', fa, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_rewind
