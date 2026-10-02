submodule (session_program_lowering_impl) internal_write_compound
    !! `internal_write_compound` procedures, moved out of `session_program_lowering_internal_write_compound.inc` so that this
    !! unit has a name, a checked interface, and can be compiled, edited
    !! and pointed at on its own instead of only inside its includer.
    implicit none

contains

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
        integer :: ndesc, di, expanded
        character(len=1) :: kinds(256)
        character(len=64) :: fmts(256)
        integer :: wv(256), pv(256), ed(256)
        logical :: any_written
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

        ! Flatten the format (groups and repeat counts expanded at compile
        ! time) and walk the value list once, expanding constant-bounds
        ! implied-dos with their loop variables bound per iteration.
        ! Format reversion restarts the descriptor list after a newline
        ! (F2018 13.10.2.1 internal-write records).
        kinds = ''
        fmts = ''
        wv = 0
        pv = 0
        ed = 0
        ndesc = 0
        pos = 1
        call collect_format_descriptors(format_body, pos, kinds, fmts, wv, pv, &
                                         ed, ndesc, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('internal write', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if
        if (pos <= len_trim(format_body)) then
            if (format_body(pos:pos) /= ')') then
                call unsupported_feature_error('internal write', node%line, &
                    node%column, 'unbalanced parentheses in format', error_msg)
                return
            end if
        end if

        di = 1
        expanded = 0
        any_written = .false.
        do item_index = 1, item_total
            call emit_write_item(arena, node%arg_indices(item_index), context, &
                                 di, kinds, fmts, wv, pv, ed, ndesc, dest_tmp, &
                                 expanded, any_written, node, error_msg)
            if (len_trim(error_msg) > 0) return
        end do

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

    recursive subroutine collect_format_descriptors(text, pos, kinds, fmts, &
                                                    wv, pv, ed, ndesc, error_msg)
        ! Expand a format body into a flat list of single-use descriptors:
        ! r(...) groups and rX repeats are unrolled at compile time.
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos
        character(len=1), intent(inout) :: kinds(:)
        character(len=64), intent(inout) :: fmts(:)
        integer, intent(inout) :: wv(:), pv(:), ed(:)
        integer, intent(inout) :: ndesc
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: digits
        character(len=64) :: pf
        character :: kind
        integer :: repeat_count, r, gstart, gi, gw, gp, ge

        call set_empty(error_msg)
        do while (pos <= len_trim(text))
            if (text(pos:pos) == ',') then
                pos = pos + 1
                cycle
            end if
            if (text(pos:pos) == ')') return
            call parse_decimal_digits(text, pos, digits)
            repeat_count = 1
            if (len(digits) > 0) then
                call read_decimal_value(digits, repeat_count, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (pos > len_trim(text)) then
                error_msg = 'dangling repeat count in format'
                return
            end if
            kind = text(pos:pos)
            if (kind >= 'a' .and. kind <= 'z') kind = char(ichar(kind) - 32)
            pos = pos + 1
            select case (kind)
            case ('(')
                gstart = ndesc + 1
                call collect_format_descriptors(text, pos, kinds, fmts, wv, pv, &
                                                ed, ndesc, error_msg)
                if (len_trim(error_msg) > 0) return
                if (pos > len_trim(text) .or. text(pos:pos) /= ')') then
                    error_msg = 'unterminated format group'
                    return
                end if
                pos = pos + 1
                do r = 2, repeat_count
                    do gi = gstart, ndesc
                        call push_descriptor(kinds, fmts, wv, pv, ed, ndesc, &
                                            kinds(gi), fmts(gi), wv(gi), pv(gi), &
                                            ed(gi), error_msg)
                        if (len_trim(error_msg) > 0) return
                    end do
                end do
            case ('X')
                do r = 1, repeat_count
                    call push_descriptor(kinds, fmts, wv, pv, ed, ndesc, 'X', &
                                        '', 0, 0, 0, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            case ('E')
                gw = 0
                gp = 0
                ge = 2
                call parse_internal_e_descriptor(text, pos, gw, gp, ge, error_msg)
                if (len_trim(error_msg) > 0) return
                do r = 1, repeat_count
                    call push_descriptor(kinds, fmts, wv, pv, ed, ndesc, 'E', &
                                        '', gw, gp, ge, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            case ('I', 'A')
                call parse_decimal_digits(text, pos, digits)
                if (kind == 'I') call skip_dot_modifier(text, pos)
                if (kind == 'I') then
                    if (len(digits) == 0 .or. digits == '0') then
                        pf = '%d'
                    else
                        call read_decimal_value(digits, gw, error_msg)
                        if (len_trim(error_msg) > 0) return
                        write (pf, '(A,I0,A)') '%', gw, 'd'
                    end if
                else
                    if (len(digits) == 0) then
                        pf = '%s'
                    else
                        call read_decimal_value(digits, gw, error_msg)
                        if (len_trim(error_msg) > 0) return
                        write (pf, '(A,I0,A)') '%', gw, 's'
                    end if
                end if
                do r = 1, repeat_count
                    call push_descriptor(kinds, fmts, wv, pv, ed, ndesc, kind, &
                                        pf, 0, 0, 0, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            case default
                error_msg = 'unsupported edit descriptor in compound format: '// &
                            kind
                return
            end select
        end do
    end subroutine collect_format_descriptors

    subroutine push_descriptor(kinds, fmts, wv, pv, ed, ndesc, kind, f, w, p, &
                              e, error_msg)
        character(len=1), intent(inout) :: kinds(:)
        character(len=64), intent(inout) :: fmts(:)
        integer, intent(inout) :: wv(:), pv(:), ed(:), ndesc
        character(len=1), intent(in) :: kind
        character(len=*), intent(in) :: f
        integer, intent(in) :: w, p, e
        character(len=:), allocatable, intent(out) :: error_msg

        if (ndesc + 1 > size(kinds)) then
            error_msg = 'format expands beyond 256 descriptors'
            return
        end if
        ndesc = ndesc + 1
        kinds(ndesc) = kind
        fmts(ndesc) = f
        wv(ndesc) = w
        pv(ndesc) = p
        ed(ndesc) = e
        call set_empty(error_msg)
    end subroutine push_descriptor

    recursive subroutine emit_write_item(arena, item_idx, context, di, kinds, &
                                          fmts, wv, pv, ed, ndesc, dest_tmp, &
                                          expanded, any_written, node, error_msg)
        ! Emit one output item: expand constant-bounds implied-dos with the
        ! loop variable bound per iteration; consume one data descriptor for
        ! every other expression. X descriptors are skipped (no value); a
        ! format restart writes a record-separating newline first.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: item_idx
        type(lowering_context_t), intent(inout) :: context
        integer, intent(inout) :: di, expanded
        character(len=1), intent(inout) :: kinds(:)
        character(len=64), intent(inout) :: fmts(:)
        integer, intent(inout) :: wv(:), pv(:), ed(:)
        integer, intent(in) :: ndesc
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        logical, intent(inout) :: any_written
        type(write_statement_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lo, hi, step, ival
        integer :: vsym, i, oi
        logical :: created_temp
        type(symbol_t) :: saved_sym
        integer, allocatable :: objs(:)

        call set_empty(error_msg)
        select type (idn => arena%entries(item_idx)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%var_name)) then
                call unsupported_feature_error('internal write implied-do', &
                    idn%line, idn%column, 'implied-do without loop variable', &
                    error_msg)
                return
            end if
            call eval_i32_constant(arena, idn%start_expr_index, context, lo, &
                                   error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('internal write implied-do', &
                    idn%line, idn%column, &
                    'implied-do bounds must be compile-time constants', &
                    error_msg)
                return
            end if
            call eval_i32_constant(arena, idn%end_expr_index, context, hi, &
                                   error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('internal write implied-do', &
                    idn%line, idn%column, &
                    'implied-do bounds must be compile-time constants', &
                    error_msg)
                return
            end if
            step = 1_c_int64_t
            if (idn%step_expr_index > 0) then
                call eval_i32_constant(arena, idn%step_expr_index, context, &
                                       step, error_msg)
                if (len_trim(error_msg) > 0) then
                    call unsupported_feature_error('internal write implied-do', &
                        idn%line, idn%column, &
                        'implied-do step must be a compile-time constant', &
                        error_msg)
                    return
                end if
            end if
            if (step == 0_c_int64_t) then
                error_msg = 'io implied-do step is zero'
                return
            end if
            if (allocated(idn%object_indices)) then
                objs = idn%object_indices
            else if (idn%expr_index > 0) then
                allocate (objs(1))
                objs(1) = idn%expr_index
            else
                call unsupported_feature_error('internal write implied-do', &
                    idn%line, idn%column, 'implied-do without objects', &
                    error_msg)
                return
            end if
            call bind_io_implied_do_var(context, idn%var_name, vsym, &
                                         created_temp, saved_sym)
            ival = lo
            do while ((step > 0_c_int64_t .and. ival <= hi) .or. &
                      (step < 0_c_int64_t .and. ival >= hi))
                context%symbols(vsym)%i32_constant = ival
                context%symbols(vsym)%value = i32_immediate(context%session, &
                                                             ival)
                context%symbols(vsym)%has_address = .false.
                context%symbols(vsym)%is_reference = .false.
                do oi = 1, size(objs)
                    call emit_write_item(arena, objs(oi), context, di, kinds, &
                                         fmts, wv, pv, ed, ndesc, dest_tmp, &
                                         expanded, any_written, node, error_msg)
                    if (len_trim(error_msg) > 0) then
                        call release_io_implied_do_var(context, vsym, &
                                                       created_temp, saved_sym)
                        return
                    end if
                end do
                ival = ival + step
            end do
            call release_io_implied_do_var(context, vsym, created_temp, &
                                            saved_sym)
        class default
            if (expanded >= 64) then
                call unsupported_feature_error('internal write', node%line, &
                    node%column, 'internal write expands beyond 64 values', &
                    error_msg)
                return
            end if
            ! Advance past X descriptors; restart the format (with a record
            ! separator once anything has been written) when exhausted.
            do
                if (di > ndesc) then
                    if (any_written) then
                        call append_internal_newline(context, dest_tmp, &
                                                    error_msg)
                        if (len_trim(error_msg) > 0) return
                    end if
                    di = 1
                end if
                if (kinds(di) /= 'X') exit
                call append_internal_blanks(context, 1, dest_tmp, error_msg)
                if (len_trim(error_msg) > 0) return
                any_written = .true.
                di = di + 1
            end do
            if (kinds(di) == 'I' .or. kinds(di) == 'A') then
                call append_internal_ia_field(arena, item_idx, context, &
                                              kinds(di), trim(fmts(di)), dest_tmp, &
                                              error_msg)
            else
                call append_internal_e_field(arena, item_idx, context, wv(di), &
                                             pv(di), ed(di), dest_tmp, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            any_written = .true.
            di = di + 1
            expanded = expanded + 1
        end select
    end subroutine emit_write_item

    subroutine append_internal_newline(context, dest_tmp, error_msg)
        ! Record separator between format reversions of an internal write.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: dest_tmp
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: nl_ptr, strcat_result
        type(lr_operand_desc_t) :: strcat_args(2)
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name(context, 'iwnl.', &
                                         context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         char(10), fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        nl_ptr = printf_format_ptr(context%session, fmt_id)
        strcat_args(1) = dest_tmp
        strcat_args(2) = nl_ptr
        if (.not. emit_ptr_call(context%session, 'strcat', strcat_args, &
                                strcat_result, error_msg)) return
        call set_empty(error_msg)
    end subroutine append_internal_newline

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
        type(lr_operand_desc_t) :: strcat_args(2)
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
        strcat_args(1) = dest_tmp
        strcat_args(2) = blanks_ptr
        if (.not. emit_ptr_call(context%session, 'strcat', &
                                strcat_args, strcat_result, &
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
        type(lr_operand_desc_t) :: strcat_args(2)

        if (scalar_real_expr_kind(arena, node_index, context) == VALUE_F32) then
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
        strcat_args(1) = dest_tmp
        strcat_args(2) = field
        if (.not. emit_ptr_call(context%session, 'strcat', strcat_args, &
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
        type(lr_operand_desc_t) :: strcat_args(2)
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

        strcat_args(1) = dest_tmp
        strcat_args(2) = scratch
        if (.not. emit_ptr_call(context%session, 'strcat', &
                                strcat_args, strcat_result, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine append_internal_ia_field


end submodule internal_write_compound
