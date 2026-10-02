submodule (session_program_lowering_impl) internal_read
    !! `internal_read` procedures, moved out of `session_program_lowering_internal_read.inc` so that this
    !! unit has a name, a checked interface, and can be compiled, edited
    !! and pointed at on its own instead of only inside its includer.
    implicit none

contains

    logical function is_internal_read(node, context)
        ! True when read's unit is a character variable rather than a unit
        ! number or '*'.
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index, open_pos
        character(len=:), allocatable :: base_name

        is_internal_read = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*') return
        symbol_index = find_symbol_compat(context, trim(node%unit_spec))
        if (symbol_index > 0) then
            is_internal_read = context%symbols(symbol_index)%value_kind == &
                VALUE_CHARACTER
            return
        end if
        open_pos = index(trim(node%unit_spec), '(')
        if (open_pos <= 1) return
        base_name = trim(node%unit_spec(:open_pos - 1))
        symbol_index = find_symbol_compat(context, base_name)
        if (symbol_index <= 0) return
        is_internal_read = context%symbols(symbol_index)%value_kind == &
            VALUE_CHARACTER
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
        type(lr_operand_desc_t) :: buffer_value, base_value, offset
        character(len=64) :: fmt_name
        character(len=:), allocatable :: unit_text, base_name, bounds_text
        integer :: open_pos, colon_pos, close_pos, lower_bound, ios

        if (.not. allocated(node%format_spec)) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read requires a literal format', error_msg)
            return
        end if
        if (.not. allocated(node%var_indices) .or. size(node%var_indices) < 1) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'only a single-value internal read is supported', &
                error_msg)
            return
        end if
        if (node%var_indices(1) > 0 .and. node_exists(arena, &
            node%var_indices(1))) then
            select type (idn => arena%entries(node%var_indices(1))%node)
            type is (io_implied_do_node)
                call lower_internal_read_implied_do(arena, node, context, &
                                                    error_msg)
                return
            class default
            end select
        end if
        if (size(node%var_indices) /= 1) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'only a single-value internal read is supported', &
                error_msg)
            return
        end if

        unit_text = trim(node%unit_spec)
        buf_index = find_symbol_compat(context, unit_text)
        if (buf_index <= 0) then
            open_pos = index(unit_text, '(')
            if (open_pos <= 1) then
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'internal read unit must be a character variable '// &
                    'or constant substring', error_msg)
                return
            end if
            close_pos = index(unit_text(open_pos + 1:), ')')
            if (close_pos <= 0) then
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'internal read unit must be a character variable '// &
                    'or constant substring', error_msg)
                return
            end if
            close_pos = close_pos + open_pos
            colon_pos = index(unit_text(open_pos + 1:close_pos - 1), ':')
            if (colon_pos <= 0) then
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'internal read substring bounds must be constant', &
                    error_msg)
                return
            end if
            base_name = trim(unit_text(:open_pos - 1))
            buf_index = find_symbol_compat(context, base_name)
            if (buf_index <= 0) then
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'internal read source was not declared', error_msg)
                return
            end if
            bounds_text = trim(unit_text(open_pos + 1:open_pos + colon_pos - 1))
            read (bounds_text, *, iostat=ios) lower_bound
            if (ios /= 0 .or. lower_bound < 1) then
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'internal read substring bounds must be constant', &
                    error_msg)
                return
            end if
            base_value = context%symbols(buf_index)%value
            offset = i32_immediate(context%session, &
                                   int(lower_bound - 1, c_int64_t))
            call ptr_plus_i32(context, base_value, offset, buffer_value, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            buffer_value = context%symbols(buf_index)%value
        end if
        if (.not. context%symbols(buf_index)%has_character_value) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read source must be assigned first', &
                error_msg)
            return
        end if

        call identifier_name(arena, node%var_indices(1), target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        target_index = find_symbol_compat(context, target_name)
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
                                                    node%column, buffer_value, &
                                                    error_msg)
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
        fmt_name = ffc_unit_global_name( &
            context, 'irf.', context%string_literal_count)
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

    subroutine lower_internal_read_implied_do(arena, node, context, error_msg)
        ! read (buf, fmt) ((a(i,j), i=..), j=..): flatten the implied-do
        ! targets with loop variables bound per value, then sscanf one integer
        ! per target in descriptor order. List-directed '*' reads every
        ! flattened target with whitespace skipping; format reversion needs no
        ! explicit record handling because '%d' skips newlines.
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: unit_text, fmt_spec, expanded, token
        character(len=64) :: fmt_name, sfmt, wtext
        integer :: buf_index, targets(64), nt, isym(64, 8), ibcnt(64), k
        integer(c_int64_t) :: ival(64, 8)
        integer :: bsym(8), bd, pos, tok_start, tok_end, digit_end
        logical :: btemp(8)
        type(symbol_t) :: bsaved(8)
        type(lr_operand_desc_t) :: buffer_value, addr, fmt_ptr, value
        type(lr_operand_desc_t), allocatable :: slots(:)
        type(lr_operand_desc_t), allocatable :: sargs(:)
        integer(c_int32_t) :: fmt_id

        call set_empty(error_msg)
        call normalize_format_body(node%format_spec, fmt_spec)
        fmt_spec = trim(adjustl(fmt_spec))
        if (fmt_spec /= '*' .and. len_trim(fmt_spec) == 0) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read requires a literal format', &
                error_msg)
            return
        end if
        unit_text = trim(node%unit_spec)
        buf_index = find_symbol_compat(context, unit_text)
        if (buf_index <= 0) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read unit must be a character variable', &
                error_msg)
            return
        end if
        if (.not. context%symbols(buf_index)%has_character_value) then
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'internal read source must be assigned first', &
                error_msg)
            return
        end if
        buffer_value = context%symbols(buf_index)%value

        targets = 0
        isym = 0
        ibcnt = 0
        ival = 0
        nt = 0
        bd = 0
        call flatten_read_targets(arena, node%var_indices(1), context, &
                                  targets, nt, isym, ival, ibcnt, bd, bsym, &
                                  btemp, bsaved, 0, error_msg)
        if (len_trim(error_msg) > 0) then
            call unbind_read_stack(context, bd, bsym, btemp, bsaved)
            return
        end if
        if (nt > 64) then
            call unbind_read_stack(context, bd, bsym, btemp, bsaved)
            call unsupported_feature_error('internal read', node%line, &
                node%column, 'implied-do expands beyond 64 targets', error_msg)
            return
        end if

        expanded = ''
        if (fmt_spec /= '*') then
            call expand_format_groups(fmt_spec, expanded, error_msg)
            if (len_trim(error_msg) > 0) then
                call unbind_read_stack(context, bd, bsym, btemp, bsaved)
                return
            end if
            if (len_trim(expanded) == 0) then
                call unbind_read_stack(context, bd, bsym, btemp, bsaved)
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'format has no data descriptor', error_msg)
                return
            end if
        end if


        ! One sscanf call, one conversion per flattened target: the format
        ! is the concatenation of per-target conversions with separator
        ! skips, so no runtime cursor threading is needed. Widths: Fortran
        ! Iw reads an exact field; sscanf %wd reads UP TO w chars, which
        ! matches for space-separated input (the same stance as the
        ! single-value path that maps I0 -> %d).
        allocate (slots(nt))
        sfmt = ''
        if (fmt_spec == '*') then
            do k = 1, nt
                if (k > 1) sfmt = trim(sfmt)//'%*[ ,]'
                sfmt = trim(sfmt)//'%d'
            end do
        else
            pos = 1
            do k = 1, nt
                call skip_format_separators(expanded, pos)
                if (pos > len_trim(expanded)) pos = 1
                tok_start = pos
                tok_end = pos
                do while (tok_end <= len_trim(expanded) .and. &
                         expanded(tok_end:tok_end) /= ',')
                    tok_end = tok_end + 1
                end do
                token = expanded(tok_start:tok_end - 1)
                if (len_trim(token) == 0 .or. &
                    (token(1:1) /= 'I' .and. token(1:1) /= 'i')) then
                    call unbind_read_stack(context, bd, bsym, btemp, bsaved)
                    call unsupported_feature_error('internal read', &
                        node%line, node%column, &
                        'internal read only supports integer edit descriptors', &
                        error_msg)
                    return
                end if
                wtext = token_digits(token)
                if (k > 1) sfmt = trim(sfmt)//'%*[ ,]'
                if (len_trim(wtext) == 0 .or. trim(wtext) == '0') then
                    sfmt = trim(sfmt)//'%d'
                else
                    sfmt = trim(sfmt)//'%'//trim(wtext)//'d'
                end if
                pos = tok_end
            end do
        end if
        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name(context, 'ird.', &
                                         context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         trim(sfmt), fmt_id, error_msg)
        if (len_trim(error_msg) > 0) then
            call unbind_read_stack(context, bd, bsym, btemp, bsaved)
            return
        end if
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        allocate (sargs(2 + nt))
        sargs(1) = buffer_value
        sargs(2) = fmt_ptr
        do k = 1, nt
            if (.not. emit_i32_alloca(context%session, slots(k), &
                                       error_msg)) then
                call unbind_read_stack(context, bd, bsym, btemp, bsaved)
                return
            end if
            sargs(2 + k) = slots(k)
        end do
        if (.not. emit_sscanf(context%session, sargs, error_msg)) then
            call unbind_read_stack(context, bd, bsym, btemp, bsaved)
            return
        end if
        do k = 1, nt
            call rebind_read_target(context, targets(k), isym(k, :), &
                                    ival(k, :), ibcnt(k), error_msg)
            if (len_trim(error_msg) > 0) exit
            select type (tn => arena%entries(targets(k))%node)
            type is (call_or_subscript_node)
                call lower_i32_array_element_address(arena, tn, context, &
                                                     addr, error_msg)
            class default
                call unsupported_feature_error('internal read', node%line, &
                    node%column, 'implied-do read targets must be array '// &
                    'elements', error_msg)
            end select
            if (len_trim(error_msg) > 0) exit
            if (.not. emit_i32_load(context%session, slots(k), value, &
                                     error_msg)) exit
            if (.not. emit_i32_store(context%session, value, addr, &
                                     error_msg)) exit
        end do
        call unbind_read_stack(context, bd, bsym, btemp, bsaved)
    end subroutine lower_internal_read_implied_do

    function token_digits(token) result(digits)
        character(len=*), intent(in) :: token
        character(len=:), allocatable :: digits
        integer :: i

        digits = ''
        i = 2
        do while (i <= len_trim(token) .and. token(i:i) >= '0' .and. &
                  token(i:i) <= '9')
            digits = trim(digits)//token(i:i)
            i = i + 1
        end do
    end function token_digits

    subroutine rebind_read_target(context, tidx, syms, vals, cnt, error_msg)
        ! Restore the loop-variable bindings for one read target before its
        ! element address is lowered.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: tidx, syms(8), cnt
        integer(c_int64_t), intent(in) :: vals(8)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: d

        call set_empty(error_msg)
        do d = 1, cnt
            context%symbols(syms(d))%i32_constant = vals(d)
            context%symbols(syms(d))%value = &
                i32_immediate(context%session, vals(d))
            context%symbols(syms(d))%has_i32_constant = .true.
            context%symbols(syms(d))%is_transient_i32_constant = .true.
            context%symbols(syms(d))%has_address = .false.
            context%symbols(syms(d))%is_reference = .false.
        end do
    end subroutine rebind_read_target

    subroutine unbind_read_stack(context, bd, bsym, btemp, bsaved)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: bd, bsym(8)
        logical, intent(in) :: btemp(8)
        type(symbol_t), intent(in) :: bsaved(8)
        integer :: d

        do d = bd, 1, -1
            call release_io_implied_do_var(context, bsym(d), btemp(d), &
                                            bsaved(d))
        end do
    end subroutine unbind_read_stack

    recursive subroutine flatten_read_targets(arena, idx, context, targets, &
                                              nt, isym, ival, ibcnt, bd, &
                                              bsym, btemp, bsaved, depth, &
                                              error_msg)
        ! Same shape as the print flattener: bind each implied-do variable,
        ! iterate constant bounds, recurse into nesting, and record plain
        ! element targets with a snapshot of the active bindings.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(inout) :: context
        integer, intent(inout) :: targets(64), nt
        integer, intent(inout) :: isym(64, 8), ibcnt(64)
        integer(c_int64_t), intent(inout) :: ival(64, 8)
        integer, intent(inout) :: bd
        integer, intent(inout) :: bsym(8)
        logical, intent(inout) :: btemp(8)
        type(symbol_t), intent(inout) :: bsaved(8)
        integer, intent(in) :: depth
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lo, hi, step, v
        integer :: vsym, oi, d
        logical :: created
        type(symbol_t) :: saved
        integer, allocatable :: objs(:)

        call set_empty(error_msg)
        if (depth > 8) then
            error_msg = 'implied-do nesting beyond depth 8'
            return
        end if
        if (idx <= 0 .or. .not. node_exists(arena, idx)) then
            error_msg = 'implied-do node missing from arena'
            return
        end if
        select type (idn => arena%entries(idx)%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%var_name)) then
                error_msg = 'implied-do without loop variable'
                return
            end if
            call eval_i32_constant(arena, idn%start_expr_index, context, lo, &
                                   error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'implied-do bounds must be compile-time constants'
                return
            end if
            call eval_i32_constant(arena, idn%end_expr_index, context, hi, &
                                   error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'implied-do bounds must be compile-time constants'
                return
            end if
            step = 1_c_int64_t
            if (idn%step_expr_index > 0) then
                call eval_i32_constant(arena, idn%step_expr_index, context, &
                                       step, error_msg)
                if (len_trim(error_msg) > 0) then
                    error_msg = 'implied-do step must be a constant'
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
                error_msg = 'implied-do without objects'
                return
            end if
            bd = bd + 1
            if (bd > size(bsym)) then
                bd = bd - 1
                error_msg = 'implied-do binding stack overflow'
                return
            end if
            call bind_io_implied_do_var(context, idn%var_name, vsym, created, &
                                         saved)
            bsym(bd) = vsym
            btemp(bd) = created
            bsaved(bd) = saved
            v = lo
            do while ((step > 0_c_int64_t .and. v <= hi) .or. &
                      (step < 0_c_int64_t .and. v >= hi))
                context%symbols(vsym)%i32_constant = v
                context%symbols(vsym)%value = &
                    i32_immediate(context%session, v)
                context%symbols(vsym)%has_i32_constant = .true.
                context%symbols(vsym)%is_transient_i32_constant = .true.
                context%symbols(vsym)%has_address = .false.
                context%symbols(vsym)%is_reference = .false.
                do oi = 1, size(objs)
                    if (objs(oi) <= 0 .or. .not. node_exists(arena, &
                        objs(oi))) then
                        error_msg = 'implied-do object missing from arena'
                        exit
                    end if
                    select type (ob => arena%entries(objs(oi))%node)
                    type is (io_implied_do_node)
                        call flatten_read_targets(arena, objs(oi), context, &
                                                 targets, nt, isym, ival, &
                                                 ibcnt, bd, bsym, btemp, &
                                                 bsaved, depth + 1, error_msg)
                    class default
                        nt = nt + 1
                        if (nt > size(targets)) then
                            error_msg = 'implied-do expansion overflow'
                            exit
                        end if
                        targets(nt) = objs(oi)
                        ibcnt(nt) = bd
                        do d = 1, bd
                            isym(nt, d) = bsym(d)
                            ival(nt, d) = context%symbols(bsym(d))%&
                                          i32_constant
                        end do
                    end select
                    if (len_trim(error_msg) > 0) exit
                end do
                if (len_trim(error_msg) > 0) exit
                v = v + step
            end do
            if (len_trim(error_msg) > 0) then
                bd = bd - 1
                return
            end if
            bd = bd - 1
        class default
            error_msg = 'flatten target is not an implied-do'
        end select
    end subroutine flatten_read_targets

    subroutine lower_list_directed_internal_read(context, buf_index, &
                                                 target_index, line, col, &
                                                 buffer_value, error_msg)
        ! read (buf, *) value: sscanf the character buffer with a conversion
        ! chosen from the target's kind (integer/real via file_read_slot_for_
        ! kind's numeric slots, character via a blank-padded token read).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: buf_index, target_index, line, col
        type(lr_operand_desc_t), intent(in) :: buffer_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: scanf_fmt
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, slot, args(3)
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        if (context%symbols(target_index)%value_kind == VALUE_CHARACTER) then
            call lower_list_directed_internal_read_char(context, buf_index, &
                                                         target_index, line, &
                                                         col, buffer_value, error_msg)
            return
        end if

        if (context%symbols(target_index)%value_kind == VALUE_LOGICAL) then
            call lower_list_directed_internal_read_logical(context, buf_index, &
                                                           target_index, buffer_value, error_msg)
            return
        end if

        call file_read_slot_for_kind(context, target_index, scanf_fmt, slot, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        if (len_trim(scanf_fmt) == 0) then
            call unsupported_feature_error('internal read', line, col, &
                'list-directed internal read supports integer, real, '// &
                'logical, and character scalars only', error_msg)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'ilrf.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = buffer_value
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
                                                       col, buffer_value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: buf_index, target_index, line, col
        type(lr_operand_desc_t), intent(in) :: buffer_value
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
        fmt_name = ffc_unit_global_name( &
            context, 'ilrfc.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = buffer_value
        args(2) = fmt_ptr
        args(3) = tmp
        if (.not. emit_sscanf(context%session, args, error_msg)) return

        call emit_blank_pad_string(context, buflen, tmp, dest, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(target_index)%value = dest
        context%symbols(target_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_list_directed_internal_read_char

    subroutine lower_list_directed_internal_read_logical(context, buf_index, &
                                                          target_index, buffer_value, error_msg)
        ! read (buf, *) flag: scan the logical field out of the character
        ! buffer and convert it exactly like the file-unit path does.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: buf_index, target_index
        type(lr_operand_desc_t), intent(in) :: buffer_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), parameter :: token_fmt = ' %15s'
        type(lr_operand_desc_t) :: token, fmt_ptr, args(3)
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        call emit_logical_token_buffer(context, token, error_msg)
        if (len_trim(error_msg) > 0) return

        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'ilrfl.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         token_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = buffer_value
        args(2) = fmt_ptr
        args(3) = token
        if (.not. emit_sscanf(context%session, args, error_msg)) return

        call assign_logical_from_token(context, token, target_index, error_msg)
    end subroutine lower_list_directed_internal_read_logical


end submodule internal_read
