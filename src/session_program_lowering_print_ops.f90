submodule(session_program_lowering_impl) session_program_lowering_print_ops
contains
    module subroutine lower_print(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: expr_count
        logical :: prev_is_char
        logical :: cur_is_char

        if (allocated(node%format_spec)) then
            if (trim(node%format_spec) /= '*') then
                call lower_formatted_print(arena, node, context, error_msg)
                return
            end if
        end if

        if (.not. allocated(node%expression_indices)) then
            if (.not. emit_liric_print_newline(context%session, error_msg)) return
            call set_empty(error_msg)
            return
        end if
        if (size(node%expression_indices) == 0) then
            if (.not. emit_liric_print_newline(context%session, error_msg)) return
            call set_empty(error_msg)
            return
        end if

        ! gfortran list-directed output writes one separating blank before
        ! every value; the first blank is also the record's carriage control.
        ! Each value field carries no leading blank of its own. The
        ! one exception: no blank is written between two consecutive character
        ! values, so they print concatenated.
        expr_count = size(node%expression_indices)
        prev_is_char = .false.
        do i = 1, expr_count
            ! A bare whole array or array section among other items prints each
            ! element inline (own leading spaces, no record break), so
            ! 'tag', arr and 'tag', arr(lo:hi) both work.
            block
                logical :: array_handled
                array_handled = .false.
                select type (item => arena%entries(node%expression_indices(i))%node)
                type is (array_slice_node)
                    if (.not. is_character_substring(arena, &
                            node%expression_indices(i), context)) then
                        call emit_array_section_print_items(arena, item, context, &
                                                            array_handled, error_msg)
                    end if
                type is (array_literal_node)
                    call emit_array_literal_print_items(arena, &
                        node%expression_indices(i), context, array_handled, &
                        error_msg)
                type is (io_implied_do_node)
                    call emit_io_implied_do_print_items(arena, &
                        node%expression_indices(i), context, array_handled, &
                        error_msg, prev_is_char)
                class default
                    call emit_whole_array_print_items(arena, &
                        node%expression_indices(i), context, array_handled, &
                        error_msg, prev_is_char)
                end select
                if (len_trim(error_msg) > 0) return
                if (array_handled) then
                    ! An implied-do reports the separator state of its last
                    ! object through prev_is_char; every other array item ends
                    ! with a value of the item's own type.
                    select type (item => arena%entries( &
                            node%expression_indices(i))%node)
                    type is (io_implied_do_node)
                    class default
                        prev_is_char = char_print_item(arena, &
                            node%expression_indices(i), context)
                    end select
                    cycle
                end if
            end block
            cur_is_char = char_print_item(arena, node%expression_indices(i), &
                                          context)
            if (.not. (i > 1 .and. cur_is_char .and. prev_is_char)) then
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
            end if
            call lower_print_expression_value(arena, node%expression_indices(i), &
                                              context, error_msg)
            if (len_trim(error_msg) > 0) return
            prev_is_char = cur_is_char
        end do
        if (.not. emit_liric_print_newline(context%session, error_msg)) return

        call set_empty(error_msg)
    end subroutine lower_print

    module subroutine lower_formatted_print(arena, node, context, error_msg)
        ! Formatted print with an explicit literal format string. The compound
        ! lowerer walks every edit descriptor (data, control, string-literal),
        ! applies repeat counts, and reverts the format across records, so it is
        ! a strict superset of the single-descriptor case.
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: format_body

        call check_formatted_io_objects(arena, node, error_msg)
        if (len_trim(error_msg) > 0) return
        call normalize_format_body(node%format_spec, format_body)
        if (len_trim(format_body) == 0) then
            ! An empty format with no items still terminates one record; an
            ! empty format beside items leaves them without a data descriptor
            ! (gfortran fails at runtime), so refuse with items present.
            if (allocated(node%expression_indices) .and. &
                size(node%expression_indices) > 0) then
                call unsupported_feature_error('formatted print', node%line, &
                    node%column, 'format has no data descriptor', error_msg)
                return
            end if
            if (.not. emit_liric_print_newline(context%session, error_msg)) return
            call set_empty(error_msg)
            return
        end if
        call lower_compound_formatted_print(arena, node, context, format_body, &
                                            error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('formatted print', node%line, &
                                           node%column, trim(error_msg), error_msg)
        end if
    end subroutine lower_formatted_print

    module subroutine parse_single_edit_descriptor(spec, kind_char, printf_fmt, &
                                                    error_msg)
        ! Translate a single Fortran edit descriptor to a printf conversion.
        ! Iw -> %wd (I0 -> %d), A[w] -> %[w]s. Compound formats are rejected.
        character(len=*), intent(in) :: spec
        character, intent(out) :: kind_char
        character(len=:), allocatable, intent(out) :: printf_fmt
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: s, width

        call set_empty(error_msg)
        kind_char = ' '
        call normalize_format_body(spec, s)
        if (index(s, ',') > 0) then
            error_msg = 'compound format strings are not supported'
            return
        end if
        if (len(s) == 0) then
            error_msg = 'empty format string'
            return
        end if
        kind_char = s(1:1)
        if (kind_char == 'i') kind_char = 'I'
        if (kind_char == 'a') kind_char = 'A'
        if (len(s) > 1) then
            width = trim(adjustl(s(2:)))
        else
            width = ''
        end if
        select case (kind_char)
        case ('I')
            if (len(width) == 0 .or. width == '0') then
                printf_fmt = '%d'
            else
                printf_fmt = '%'//width//'d'
            end if
        case ('A')
            if (len(width) == 0) then
                printf_fmt = '%s'
            else
                printf_fmt = '%'//width//'s'
            end if
        case default
            error_msg = 'unsupported edit descriptor: '//s
        end select
    end subroutine parse_single_edit_descriptor

    module subroutine lower_compound_formatted_print(arena, node, context, &
                                                     format_body, error_msg)
        ! Walk the compound format once per record. When the format is exhausted
        ! but data items remain (format reversion, F2018 13.4), terminate the
        ! record with a newline and restart the format from the beginning until
        ! every item is consumed.
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: pos
        integer :: item_index
        integer :: pass_start_index
        integer :: item_total
        logical :: exhausted
        logical :: implied_do_handled

        call set_empty(error_msg)
        if (len_trim(format_body) == 0 .or. &
            trim(format_body) == "()") then
            ! An empty format has no data descriptor: gfortran fails at
            ! runtime on any output item; refuse the statement instead.
            call unsupported_feature_error('formatted print', node%line, &
                node%column, 'format has no data descriptor', error_msg)
            return
        end if
        implied_do_handled = .false.
        if (allocated(node%expression_indices)) then
            if (size(node%expression_indices) == 1) then
                call lower_formatted_io_implied_do(arena, node, context, format_body, &
                                                   implied_do_handled, error_msg)
                if (len_trim(error_msg) > 0) return
                if (implied_do_handled) then
                    if (.not. emit_liric_print_newline(context%session, error_msg)) &
                        return
                    return
                end if
            end if
        end if
        ! Any remaining implied-do sits beside other top-level items: the
        ! flattened walk owns sole-control statements only. Refuse by name so
        ! the diagnostic says what is unsupported.
        if (allocated(node%expression_indices)) then
            do item_index = 1, size(node%expression_indices)
                if (node_exists(arena, node%expression_indices(item_index))) then
                    select type (dn => arena%entries(node%expression_indices(item_index))%node)
                    type is (io_implied_do_node)
                        call unsupported_feature_error('formatted I/O implied-do', &
                            node%line, node%column, &
                            'implied-do beside other output items is not yet lowered', &
                            error_msg)
                        return
                    class default
                    end select
                end if
            end do
        end if
        item_index = 1
        exhausted = .false.
        if (allocated(node%expression_indices)) then
            item_total = size(node%expression_indices)
        else
            item_total = 0
        end if
        do
            pos = 1
            pass_start_index = item_index
            do
                call skip_format_separators(format_body, pos)
                if (pos > len_trim(format_body)) exit
                ! A trailing nX prints blanks the record would omit; once no
                ! values remain, skip X steps instead of padding (13.10.2).
                if (item_index > item_total) then
                    block
                        character :: kc
                        integer :: p2
                        p2 = pos
                        do while (p2 <= len_trim(format_body) .and. &
                                 format_body(p2:p2) >= '0' .and. &
                                 format_body(p2:p2) <= '9')
                            p2 = p2 + 1
                        end do
                        kc = ' '
                        if (p2 <= len_trim(format_body)) &
                            kc = format_body(p2:p2)
                        if (kc == 'X' .or. kc == 'x') then
                            pos = p2 + 1
                            cycle
                        end if
                    end block
                end if
                call lower_next_compound_descriptor(arena, node, context, &
                                                    format_body, pos, item_index, &
                                                    exhausted, error_msg)
                if (len_trim(error_msg) > 0) return
                ! A data descriptor with no remaining item terminates format
                ! processing immediately (F2018 13.4): write the record and stop.
                if (exhausted) exit
            end do
            if (.not. emit_liric_print_newline(context%session, error_msg)) return
            if (exhausted .or. item_index > item_total) exit
            ! No item consumed in a full pass with items still pending means the
            ! format has no data descriptor: reversion would loop forever.
            if (item_index == pass_start_index) then
                error_msg = 'format has no data descriptor for remaining items'
                return
            end if
        end do
        call set_empty(error_msg)
    end subroutine lower_compound_formatted_print

    module subroutine lower_formatted_io_implied_do(arena, node, context, &
                                                    format_body, handled, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lo, hi, step, ival
        integer :: item_index, pos, vsym
        logical :: created_temp, exhausted
        type(symbol_t) :: saved_sym
        type(print_statement_node) :: one_node
        integer, allocatable :: objects(:)

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%expression_indices)) return
        if (size(node%expression_indices) /= 1) return
        if (.not. node_exists(arena, node%expression_indices(1))) return

        select type (idn => arena%entries(node%expression_indices(1))%node)
        type is (io_implied_do_node)
            if (.not. allocated(idn%var_name)) return
            if (allocated(idn%object_indices)) then
                objects = idn%object_indices
            else if (idn%expr_index > 0) then
                allocate (objects(1))
                objects(1) = idn%expr_index
            else
                return
            end if
            if (size(objects) == 0) return
            if (.true.) then
                ! Multi-object and nested implied-dos: flatten the value list
                ! with loop variables recorded per value, expand the format
                ! into single-descriptor steps, then walk both in step.
                call lower_flattened_implied_do_print(arena, node, context, &
                                                      format_body, error_msg)
                if (len_trim(error_msg) > 0) return
                handled = .true.
                return
            end if
            call eval_i32_constant(arena, idn%start_expr_index, context, lo, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, idn%end_expr_index, context, hi, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
            step = 1_c_int64_t
            if (idn%step_expr_index > 0) then
                call eval_i32_constant(arena, idn%step_expr_index, context, step, &
                                       error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (step == 0_c_int64_t) then
                error_msg = 'io implied-do step is zero'
                return
            end if

            one_node%expression_indices = objects
            call bind_io_implied_do_var(context, idn%var_name, vsym, &
                                        created_temp, saved_sym)
            ival = lo
            do while ((step > 0_c_int64_t .and. ival <= hi) .or. &
                      (step < 0_c_int64_t .and. ival >= hi))
                context%symbols(vsym)%i32_constant = ival
                context%symbols(vsym)%value = i32_immediate(context%session, ival)
                context%symbols(vsym)%has_address = .false.
                context%symbols(vsym)%is_reference = .false.
                item_index = 1
                pos = 1
                call lower_next_compound_descriptor(arena, one_node, context, &
                                                    format_body, pos, item_index, &
                                                    exhausted, error_msg)
                if (len_trim(error_msg) > 0) then
                    call release_io_implied_do_var(context, vsym, created_temp, &
                                                   saved_sym)
                    return
                end if
                ival = ival + step
            end do
            call release_io_implied_do_var(context, vsym, created_temp, saved_sym)
            handled = .true.
        class default
        end select
    end subroutine lower_formatted_io_implied_do

    subroutine lower_flattened_implied_do_print(arena, node, context, &
                                                  format_body, error_msg)
        ! Multi-object / nested implied-do print: flatten the value list with
        ! loop variables bound per value, expand the format (groups and repeat
        ! counts) into single-descriptor steps, then walk descriptor and
        ! value in step. Format reversion terminates the record with a
        ! newline and restarts (F2018 13.4).
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: items(64), isym(64, 8), ibcnt(64), nitems
        integer(c_int64_t) :: ival(64, 8)
        integer :: bsym(8), bd, k, pos, item_index, rc, i
        logical :: btemp(8), exhausted
        type(symbol_t) :: bsaved(8)
        character(len=:), allocatable :: expanded
        type(print_statement_node) :: one_node

        call set_empty(error_msg)
        items = 0
        isym = 0
        ibcnt = 0
        ival = 0
        nitems = 0
        bd = 0
        call flatten_implied_do(arena, node%expression_indices(1), context, &
                                items, nitems, isym, ival, ibcnt, bd, bsym, &
                                btemp, bsaved, 0, error_msg)
        if (len_trim(error_msg) > 0) then
            call unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
            return
        end if
        if (nitems > 64) then
            call unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
            call unsupported_feature_error('formatted I/O implied-do', &
                node%line, node%column, &
                'implied-do expands beyond 64 values', error_msg)
            return
        end if
        if (nitems == 0) then
            ! Zero iterations: the print statement still terminates its
            ! record; the caller emits the newline.
            call unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
            return
        end if
        call expand_format_groups(format_body, expanded, error_msg)
        if (len_trim(error_msg) > 0) then
            call unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
            return
        end if
        ! A record omits its trailing blanks (F2018 13.10.2): X steps that
        ! trail the last data descriptor of every pass are dropped up front,
        ! which trims every reversion record the same way.
        do
            if (len_trim(expanded) >= 2 .and. &
                expanded(len_trim(expanded) - 1:) == ',X') then
                expanded = expanded(:len_trim(expanded) - 2)
                cycle
            end if
            if (trim(expanded) == 'X') expanded = ''
            exit
        end do
        if (len_trim(expanded) == 0) then
            call unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
            call unsupported_feature_error('formatted I/O implied-do', &
                node%line, node%column, &
                'format has no data descriptor for implied-do values', &
                error_msg)
            return
        end if

        allocate (one_node%expression_indices(nitems))
        one_node%expression_indices(1:nitems) = items(1:nitems)
        pos = 1
        item_index = 1
        do
            call skip_format_separators(expanded, pos)
            if (pos > len_trim(expanded)) then
                ! Record boundary: the caller emits the terminating newline,
                ! so only inter-record separators are written here.
                if (item_index > nitems) exit
                if (.not. emit_liric_print_newline(context%session, &
                                                    error_msg)) exit
                pos = 1
                cycle
            end if
            do i = 1, ibcnt(item_index)
                context%symbols(isym(item_index, i))%i32_constant = &
                    ival(item_index, i)
                context%symbols(isym(item_index, i))%value = &
                    i32_immediate(context%session, ival(item_index, i))
                context%symbols(isym(item_index, i))%has_address = .false.
                context%symbols(isym(item_index, i))%is_reference = .false.
            end do
            if (item_index > nitems) then
                ! All values emitted: the record ends here. Trailing X steps
                ! would print blanks, and a formatted print record omits its
                ! trailing blanks (F2018 13.10.2), so stop, do not pad.
                exit
            end if
            call lower_next_compound_descriptor(arena, one_node, context, &
                                                expanded, pos, item_index, &
                                                exhausted, error_msg)
            if (len_trim(error_msg) > 0) exit
            if (exhausted) exit
        end do
        call unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
    end subroutine lower_flattened_implied_do_print

    subroutine unbind_implied_do_stack(context, bd, bsym, btemp, bsaved)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: bd, bsym(8)
        logical, intent(in) :: btemp(8)
        type(symbol_t), intent(in) :: bsaved(8)
        integer :: d

        do d = bd, 1, -1
            call release_io_implied_do_var(context, bsym(d), btemp(d), &
                                            bsaved(d))
        end do
    end subroutine unbind_implied_do_stack

    recursive subroutine flatten_implied_do(arena, idx, context, items, &
                                             nitems, isym, ival, ibcnt, bd, &
                                             bsym, btemp, bsaved, depth, &
                                             error_msg)
        ! Expand one implied-do at a time: bind its loop variable, iterate the
        ! constant bounds, and recurse into nested implied-dos. Every plain
        ! value records a snapshot of the active loop-variable bindings.
        ! Explicit shapes everywhere: call sites use identically sized arrays,
        ! implicit interfaces carry no descriptors.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(inout) :: context
        integer, intent(inout) :: items(64), nitems
        integer, intent(inout) :: isym(64, 8), ibcnt(64)
        integer(c_int64_t), intent(inout) :: ival(64, 8)
        integer, intent(inout) :: bd
        integer, intent(inout) :: bsym(8)
        logical, intent(inout) :: btemp(8)
        type(symbol_t), intent(inout) :: bsaved(8)
        integer, intent(in) :: depth
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lo, hi, step, v
        integer :: vsym, i, oi, d
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
                context%symbols(vsym)%value = i32_immediate(context%session, v)
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
                        call flatten_implied_do(arena, objs(oi), context, &
                                                items, nitems, isym, ival, &
                                                ibcnt, bd, bsym, btemp, bsaved, &
                                                depth + 1, error_msg)
                    class default
                        nitems = nitems + 1
                        if (nitems > size(items)) then
                            error_msg = 'implied-do expansion overflow'
                            exit
                        end if
                        items(nitems) = objs(oi)
                        ibcnt(nitems) = bd
                        do d = 1, bd
                            isym(nitems, d) = bsym(d)
                            ival(nitems, d) = context%symbols(bsym(d))%&
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
    end subroutine flatten_implied_do

    module subroutine expand_format_groups(format_body, expanded, error_msg)
        ! Expand r(...) groups to rep comma-separated copies of the expanded
        ! body and rX to rep bare X steps; other descriptor tokens are copied
        ! verbatim up to the next separator. A comma inside a quoted literal
        ! is refused so the flattener never splits a string in half.
        character(len=*), intent(in) :: format_body
        character(len=:), allocatable, intent(out) :: expanded
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=512) :: out, body
        integer :: pos, rep, r, start
        logical :: quoted

        call set_empty(error_msg)
        out = ''
        pos = 1
        call expand_fmt_into(format_body, pos, out, quoted, error_msg)
        if (len_trim(error_msg) > 0) then
            expanded = ''
            return
        end if
        expanded = trim(out)
    end subroutine expand_format_groups

    recursive subroutine expand_fmt_into(text, pos, out, quoted, error_msg)
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos
        character(len=512), intent(inout) :: out
        logical, intent(inout) :: quoted
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: digits, body
        character(len=512) :: local_out
        integer :: rep, r, start
        character :: q

        call set_empty(error_msg)
        do while (pos <= len_trim(text))
            if (text(pos:pos) == ',') then
                pos = pos + 1
                cycle
            end if
            if (text(pos:pos) == ')') return
            call parse_decimal_digits(text, pos, digits)
            rep = 1
            if (len(digits) > 0) call read_decimal_value(digits, rep, &
                                                          error_msg)
            if (len_trim(error_msg) > 0) return
            if (pos > len_trim(text)) then
                error_msg = 'dangling repeat count in format'
                return
            end if
            if (text(pos:pos) == '(') then
                pos = pos + 1
                local_out = ''
                call expand_fmt_into(text, pos, local_out, quoted, error_msg)
                if (len_trim(error_msg) > 0) return
                if (pos > len_trim(text) .or. text(pos:pos) /= ')') then
                    error_msg = 'unterminated format group'
                    return
                end if
                pos = pos + 1
                do r = 1, rep
                    if (len_trim(out) > 0) out = trim(out)//','
                    out = trim(out)//trim(local_out)
                end do
                cycle
            end if
            ! Plain descriptor: copy the token up to the next separator,
            ! respecting quoted literals (a comma inside quotes is refused).
            start = pos
            quoted = .false.
            q = ' '
            do while (pos <= len_trim(text))
                if (quoted) then
                    if (text(pos:pos) == q) quoted = .false.
                    pos = pos + 1
                    cycle
                end if
                if (text(pos:pos) == '"' .or. text(pos:pos) == "'") then
                    quoted = .true.
                    q = text(pos:pos)
                    pos = pos + 1
                    cycle
                end if
                if (text(pos:pos) == ',') exit
                if (text(pos:pos) == ')') exit
                pos = pos + 1
            end do
            if (quoted) then
                error_msg = 'unterminated quoted literal in format'
                return
            end if
            body = text(start:pos - 1)
            if (body(1:1) == 'X' .or. body(1:1) == 'x') then
                ! nX becomes n bare X steps, unless the token carried letters
                ! beyond X (then it is not a plain skip).
                if (len_trim(body) /= 1) then
                    error_msg = 'cannot expand descriptor token: '// &
                                trim(body)
                    return
                end if
                do r = 1, rep
                    if (len_trim(out) > 0) out = trim(out)//','
                    out = trim(out)//'X'
                end do
            else
                do r = 1, rep
                    if (len_trim(out) > 0) out = trim(out)//','
                    out = trim(out)//trim(body)
                end do
            end if
        end do
    end subroutine expand_fmt_into

    logical function multi_or_nested_objects(arena, objects) result(flag)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: objects(:)
        integer :: i

        flag = size(objects) /= 1
        if (flag) return
        if (node_exists(arena, objects(1))) then
            select type (object => arena%entries(objects(1))%node)
            type is (io_implied_do_node)
                flag = .true.
            class default
            end select
        end if
    end function multi_or_nested_objects

    subroutine check_formatted_io_objects(arena, node, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, item, single_object(1)

        call set_empty(error_msg)
        if (.not. allocated(node%expression_indices)) return
        do i = 1, size(node%expression_indices)
            item = node%expression_indices(i)
            if (.not. node_exists(arena, item)) cycle
            select type (loop => arena%entries(item)%node)
                type is (io_implied_do_node)
                if (allocated(loop%object_indices)) then
                    if (.not. formatted_io_objects_supported(arena, &
                        loop%object_indices, loop%line, loop%column, &
                        error_msg)) return
                else if (loop%expr_index > 0) then
                    single_object(1) = loop%expr_index
                    if (.not. formatted_io_objects_supported(arena, &
                        single_object, loop%line, loop%column, &
                        error_msg)) return
                end if
            end select
        end do
    end subroutine check_formatted_io_objects

    logical function formatted_io_objects_supported(arena, objects, line, column, &
            error_msg) result(supported)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: objects(:), line, column
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        supported = .false.
        call set_empty(error_msg)
        supported = .true.
        ! Multiple and nested implied-do objects lower through the flattened
        ! walk (multi_or_nested_objects route); every object must exist.
        do i = 1, size(objects)
            if (.not. node_exists(arena, objects(i))) then
                call unsupported_feature_error('formatted I/O implied-do', &
                    line, column, 'implied-do object missing from arena', &
                    error_msg)
                return
            end if
        end do
    end function formatted_io_objects_supported

    recursive module subroutine lower_next_compound_descriptor(arena, node, context, &
                                                              format_body, pos, &
                                                              item_index, exhausted, &
                                                              error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        integer, intent(inout) :: item_index
        logical, intent(out) :: exhausted
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: prefix
        character(len=:), allocatable :: width
        character(len=:), allocatable :: precision
        character(len=:), allocatable :: printf_fmt
        character :: kind_char
        integer :: repeat_count
        integer :: width_value
        integer :: precision_value
        integer :: buffer_size
        integer :: e_mode
        integer :: i
        character(len=32) :: width_text
        character(len=32) :: precision_text
        character(len=:), allocatable :: min_digits
        character(len=32) :: min_digits_text

        call set_empty(error_msg)
        exhausted = .false.
        ! A '/' edit descriptor terminates the current record; no repeat prefix.
        if (format_body(pos:pos) == '/') then
            pos = pos + 1
            if (.not. emit_liric_print_newline(context%session, error_msg)) return
            return
        end if
        ! A quoted character-string edit descriptor prints its literal text.
        if (format_body(pos:pos) == "'" .or. format_body(pos:pos) == '"') then
            call lower_format_string_literal(context, format_body, pos, error_msg)
            return
        end if
        call parse_decimal_digits(format_body, pos, prefix)
        if (pos > len_trim(format_body)) then
            error_msg = 'dangling repeat count in compound format'
            return
        end if
        repeat_count = 1
        if (len(prefix) > 0) then
            call read_decimal_value(prefix, repeat_count, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        ! A parenthesized group r(...) repeats its inner descriptor list r times
        ! (F2018 13.3.3): recurse over the group body once per repeat.
        if (format_body(pos:pos) == '(') then
            call lower_compound_group(arena, node, context, format_body, pos, &
                                      repeat_count, item_index, exhausted, &
                                      error_msg)
            return
        end if
        kind_char = format_body(pos:pos)
        if (kind_char >= 'a' .and. kind_char <= 'z') &
            kind_char = char(ichar(kind_char) - 32)
        pos = pos + 1

        select case (kind_char)
        case ('X')
            ! nX writes n blanks; the count is the X repeat, not a data repeat.
            do i = 1, repeat_count
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
            end do
        case ('B', 'O', 'Z')
            call lower_boz_descriptor(arena, node, context, format_body, pos, &
                                      kind_char, repeat_count, item_index, &
                                      exhausted, error_msg)
        case ('I')
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                error_msg = 'I edit descriptor requires width'
                return
            end if
            ! Iw.m: m is the minimum number of digits, zero-padded. printf gets
            ! the same shape from a precision field: `%w.md` prints at least m
            ! digits right-justified in a field of w, which is Fortran's Iw.m.
            ! Verified against gfortran: I6.3/42 -> "   042", I6.3/0 -> "   000",
            ! I4.4/7 -> "0007", I5.3/-42 -> " -042". Discarding m (the previous
            ! behaviour) silently printed "    42" and "     0".
            if (allocated(min_digits)) deallocate (min_digits)
            if (pos <= len_trim(format_body)) then
                if (format_body(pos:pos) == '.') then
                    pos = pos + 1
                    call parse_decimal_digits(format_body, pos, min_digits)
                end if
            end if
            call read_decimal_value(width, width_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (width_value == 0) then
                printf_fmt = '%d'
            else
                write (width_text, '(I0)') width_value
                printf_fmt = '%'//trim(width_text)//'d'
                if (allocated(min_digits)) then
                    if (len_trim(min_digits) > 0 .and. min_digits /= '0') then
                        call read_decimal_value(min_digits, precision_value, &
                                               error_msg)
                        if (len_trim(error_msg) > 0) return
                        write (min_digits_text, '(I0)') precision_value
                        printf_fmt = '%'//trim(width_text)//'.'// &
                                    trim(min_digits_text)//'d'
                    end if
                end if
            end if
            call repeat_data_descriptor(arena, node, context, kind_char, &
                                        printf_fmt, 0, repeat_count, item_index, &
                                        exhausted, error_msg)
        case ('F')
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                error_msg = 'F edit descriptor requires width and precision'
                return
            end if
            if (pos > len_trim(format_body)) then
                error_msg = 'F edit descriptor requires width and precision'
                return
            end if
            if (format_body(pos:pos) /= '.') then
                error_msg = 'F edit descriptor requires width and precision'
                return
            end if
            pos = pos + 1
            call parse_decimal_digits(format_body, pos, precision)
            if (len(precision) == 0) then
                error_msg = 'F edit descriptor requires precision'
                return
            end if
            call read_decimal_value(width, width_value, error_msg)
            if (len_trim(error_msg) > 0) return
            call read_decimal_value(precision, precision_value, error_msg)
            if (len_trim(error_msg) > 0) return
            write (width_text, '(I0)') width_value
            write (precision_text, '(I0)') precision_value
            ! stdout's SIGN='PLUS' connection mode (#280) forces a leading '+'
            ! on non-negative values, matching printf's own '+' flag.
            if (context%stdout_force_plus_sign) then
                printf_fmt = '%+'//trim(width_text)//'.'//trim(precision_text)//'f'
            else
                printf_fmt = '%'//trim(width_text)//'.'//trim(precision_text)//'f'
            end if
            buffer_size = max(64, width_value + precision_value + 32)
            call repeat_data_descriptor(arena, node, context, kind_char, &
                                        printf_fmt, buffer_size, repeat_count, &
                                        item_index, exhausted, error_msg)
        case ('E')
            if (pos > len_trim(format_body)) then
                error_msg = 'E edit descriptor requires width and precision'
                return
            end if
            e_mode = 0
            if (format_body(pos:pos) == 'S' .or. format_body(pos:pos) == 's') then
                e_mode = -1
                pos = pos + 1
            else if (format_body(pos:pos) == 'N' .or. &
                     format_body(pos:pos) == 'n') then
                e_mode = 1
                pos = pos + 1
            end if
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
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
            pos = pos + 1
            call parse_decimal_digits(format_body, pos, precision)
            if (len(precision) == 0) then
                error_msg = 'E edit descriptor requires precision'
                return
            end if
            call read_decimal_value(width, width_value, error_msg)
            if (len_trim(error_msg) > 0) return
            call read_decimal_value(precision, precision_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (e_mode == -1) then
                write (width_text, '(I0)') width_value
                ! gfortran ESw.d prints d fraction digits and an uppercase
                ! two-digit exponent: printf %w.dE matches.
                write (precision_text, '(I0)') precision_value
                ! The "#" flag keeps the decimal point when precision is zero
                ! (gfortran's ESw.0 prints "3.E+00"; plain %.0E drops the dot).
                printf_fmt = '%#'//trim(width_text)//'.'//trim(precision_text)//'E'
                buffer_size = max(64, width_value + precision_value + 32)
                call repeat_data_descriptor(arena, node, context, 'E', &
                                            printf_fmt, buffer_size, &
                                            repeat_count, item_index, exhausted, &
                                            error_msg)
            else
                call repeat_e_en_descriptor(arena, node, context, e_mode, &
                                            width_value, precision_value, &
                                            repeat_count, item_index, exhausted, &
                                            error_msg)
            end if
        case ('A')
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                printf_fmt = '%s'
            else
                call read_decimal_value(width, width_value, error_msg)
                if (len_trim(error_msg) > 0) return
                write (width_text, '(I0)') width_value
                ! Aw truncates to w characters and right-justifies; printf's
                ! precision does the truncation, the width the padding.
                printf_fmt = '%'//trim(width_text)//'.'//trim(width_text)//'s'
            end if
            call repeat_data_descriptor(arena, node, context, kind_char, &
                                        printf_fmt, 0, repeat_count, item_index, &
                                        exhausted, error_msg)
        case ('L')
            ! Lw prints right-justified T/F in a field of width w (min 1).
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                width_value = 1
            else
                call read_decimal_value(width, width_value, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            do i = 1, repeat_count
                call lower_compound_logical_descriptor(arena, node, context, &
                                                       width_value, item_index, &
                                                       exhausted, error_msg)
                if (len_trim(error_msg) > 0) return
                if (exhausted) exit
            end do
        case default
            error_msg = 'unsupported edit descriptor in compound format: '// &
                        kind_char
        end select
    end subroutine lower_next_compound_descriptor

    recursive module subroutine lower_compound_group(arena, node, context, &
                                                     format_body, pos, repeat_count, &
                                                     item_index, exhausted, error_msg)
        ! Lower a parenthesized group r(...). pos points at the opening '('.
        ! On entry the group body is walked repeat_count times; each walk runs
        ! the same descriptor list, stopping when the data items are exhausted.
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        integer, intent(in) :: repeat_count
        integer, intent(inout) :: item_index
        logical, intent(out) :: exhausted
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: group_body
        integer :: close_pos, inner_pos, rep

        call set_empty(error_msg)
        exhausted = .false.
        call find_group_close(format_body, pos, close_pos, error_msg)
        if (len_trim(error_msg) > 0) return
        group_body = format_body(pos + 1:close_pos - 1)
        pos = close_pos + 1
        do rep = 1, repeat_count
            inner_pos = 1
            do
                call skip_format_separators(group_body, inner_pos)
                if (inner_pos > len_trim(group_body)) exit
                call lower_next_compound_descriptor(arena, node, context, &
                                                    group_body, inner_pos, &
                                                    item_index, exhausted, &
                                                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (exhausted) return
            end do
        end do
    end subroutine lower_compound_group

    module subroutine find_group_close(text, open_pos, close_pos, error_msg)
        ! Find the ')' matching the '(' at open_pos, tracking nesting depth and
        ! skipping parentheses that appear inside quoted string descriptors.
        character(len=*), intent(in) :: text
        integer, intent(in) :: open_pos
        integer, intent(out) :: close_pos
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, depth, n
        character :: ch, quote

        call set_empty(error_msg)
        close_pos = 0
        n = len_trim(text)
        depth = 0
        i = open_pos
        do while (i <= n)
            ch = text(i:i)
            if (ch == "'" .or. ch == '"') then
                quote = ch
                i = i + 1
                do while (i <= n)
                    if (text(i:i) == quote) then
                        ! A doubled quote is an embedded delimiter, not the end.
                        if (i < n) then
                            if (text(i + 1:i + 1) == quote) then
                                i = i + 2
                                cycle
                            end if
                        end if
                        exit
                    end if
                    i = i + 1
                end do
            else if (ch == '(') then
                depth = depth + 1
            else if (ch == ')') then
                depth = depth - 1
                if (depth == 0) then
                    close_pos = i
                    return
                end if
            end if
            i = i + 1
        end do
        error_msg = 'unterminated group in compound format'
    end subroutine find_group_close

    module subroutine skip_dot_modifier(format_body, pos)
        ! Skip a trailing ".m" modifier (e.g. I5.3) without consuming the next
        ! descriptor.
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        character(len=:), allocatable :: dummy

        if (pos <= len_trim(format_body)) then
            if (format_body(pos:pos) == '.') then
                pos = pos + 1
                call parse_decimal_digits(format_body, pos, dummy)
            end if
        end if
    end subroutine skip_dot_modifier

    module subroutine repeat_data_descriptor(arena, node, context, kind_char, &
                                             printf_fmt, buffer_size, repeat_count, &
                                             item_index, exhausted, error_msg)
        ! Apply one data edit descriptor to repeat_count consecutive items,
        ! stopping early when the item list runs out.
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character, intent(in) :: kind_char
        character(len=*), intent(in) :: printf_fmt
        integer, intent(in) :: buffer_size
        integer, intent(in) :: repeat_count
        integer, intent(inout) :: item_index
        logical, intent(out) :: exhausted
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        exhausted = .false.
        do i = 1, repeat_count
            call lower_compound_data_descriptor(arena, node, context, kind_char, &
                                                printf_fmt, buffer_size, &
                                                item_index, exhausted, error_msg)
            if (len_trim(error_msg) > 0) return
            if (exhausted) return
        end do
    end subroutine repeat_data_descriptor

    module subroutine lower_format_string_literal(context, format_body, pos, &
                                                  error_msg)
        ! Emit a quoted character-string edit descriptor. Fortran doubles the
        ! delimiter to embed it ('' inside '...'); collapse those to one.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        character(len=:), allocatable, intent(out) :: error_msg
        character :: quote
        character(len=:), allocatable :: text
        integer :: n
        character(len=64) :: string_name

        call set_empty(error_msg)
        quote = format_body(pos:pos)
        pos = pos + 1
        text = ''
        n = len_trim(format_body)
        do
            if (pos > n) then
                error_msg = 'unterminated string literal in format'
                return
            end if
            if (format_body(pos:pos) == quote) then
                if (pos < n) then
                    if (format_body(pos + 1:pos + 1) == quote) then
                        text = text//quote
                        pos = pos + 2
                        cycle
                    end if
                end if
                pos = pos + 1
                exit
            end if
            text = text//format_body(pos:pos)
            pos = pos + 1
        end do
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name(context, 'str.', &
                                           context%string_literal_count)
        if (.not. emit_liric_print_string_value(context%session, &
                context%str_print_format_id, trim(string_name), text, &
                error_msg)) return
    end subroutine lower_format_string_literal

    module function ffc_unit_global_name(context, kind_tag, counter) result(name)
        ! Build a per-unit .ffc content-global symbol name. When the unit carries
        ! a symbol prefix (a separately compiled module object), the prefix is
        ! inserted after '.ffc.' so the counter-numbered global does not collide
        ! with the main or a sibling module object at link time (#284). The fixed
        ! shared runtime helpers keep their stable names and are emitted elsewhere.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: kind_tag
        integer, intent(in) :: counter
        character(len=:), allocatable :: name
        character(len=32) :: num

        write (num, '(I0)') counter
        if (allocated(context%unit_symbol_prefix)) then
            name = '.ffc.'//context%unit_symbol_prefix//trim(kind_tag)// &
                   trim(num)
        else
            name = '.ffc.'//trim(kind_tag)//trim(num)
        end if
    end function ffc_unit_global_name

    module subroutine lower_compound_logical_descriptor(arena, node, context, width, &
                                                        item_index, exhausted, &
                                                        error_msg)
        ! Lw output: width-1 leading blanks then T/F (gfortran right-justifies).
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: width
        integer, intent(inout) :: item_index
        logical, intent(out) :: exhausted
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i

        call set_empty(error_msg)
        exhausted = .false.
        if (.not. allocated(node%expression_indices)) then
            exhausted = .true.
            return
        end if
        if (item_index > size(node%expression_indices)) then
            exhausted = .true.
            return
        end if
        do i = 1, max(width - 1, 0)
            if (.not. emit_liric_print_space(context%session, error_msg)) return
        end do
        call lower_logical_expression(arena, node%expression_indices(item_index), &
                                      context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_print_logical_value(context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        item_index = item_index + 1
    end subroutine lower_compound_logical_descriptor

    module subroutine lower_compound_data_descriptor(arena, node, context, kind_char, &
                                                     printf_fmt, buffer_size, &
                                                     item_index, exhausted, error_msg)
        ! Encountering a data descriptor with no remaining item terminates the
        ! format (exhausted=.true.); it is not an error (F2018 13.4).
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character, intent(in) :: kind_char
        character(len=*), intent(in) :: printf_fmt
        integer, intent(in) :: buffer_size
        integer, intent(inout) :: item_index
        logical, intent(out) :: exhausted
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        exhausted = .false.
        if (.not. allocated(node%expression_indices)) then
            exhausted = .true.
            return
        end if
        if (item_index > size(node%expression_indices)) then
            exhausted = .true.
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        fmt_name = ffc_unit_global_name( &
            context, 'fmt.user.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         printf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return

        if (kind_char == 'I') then
            call lower_formatted_int_item(arena, node%expression_indices(item_index), &
                                          context, fmt_id, error_msg)
        else if (kind_char == 'A') then
            call lower_formatted_char_item(arena, &
                                           node%expression_indices(item_index), &
                                           context, fmt_id, error_msg)
        else
            call lower_formatted_real_item(arena, node%expression_indices(item_index), &
                                           context, fmt_id, buffer_size, &
                                           error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        item_index = item_index + 1
        call set_empty(error_msg)
    end subroutine lower_compound_data_descriptor

    module subroutine repeat_e_en_descriptor(arena, node, context, mode, width, &
                                             precision, repeat_count, item_index, &
                                             exhausted, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: mode, width, precision, repeat_count
        integer, intent(inout) :: item_index
        logical, intent(out) :: exhausted
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        exhausted = .false.
        do i = 1, repeat_count
            if (item_index > size(node%expression_indices)) then
                exhausted = .true.
                return
            end if
            call lower_formatted_e_en_real_item(arena, &
                                                node%expression_indices(item_index), &
                                                context, mode, width, precision, &
                                                error_msg)
            if (len_trim(error_msg) > 0) return
            item_index = item_index + 1
        end do
        call set_empty(error_msg)
    end subroutine repeat_e_en_descriptor

    module subroutine lower_formatted_real_item(arena, node_index, context, fmt_id, &
                                               buffer_size, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: fmt_id
        integer, intent(in) :: buffer_size
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: value_f64
        type(lr_operand_desc_t) :: field
        type(lr_operand_desc_t) :: args(4)

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
                i64_immediate(context%session, int(buffer_size, c_int64_t)), &
                field, error_msg)) return

        args(1) = field
        args(2) = i64_immediate(context%session, int(buffer_size, c_int64_t))
        args(3) = printf_format_ptr(context%session, fmt_id)
        args(4) = value
        if (.not. emit_snprintf(context%session, args, error_msg)) return
        if (.not. emit_liric_print_string_operand_value(context%session, &
                context%str_print_format_id, field, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_formatted_real_item

    module subroutine lower_formatted_e_en_real_item(arena, node_index, context, mode, &
                                                     width, precision, error_msg)
        use liric_session_format_bindings, only: emit_e_en_format_call
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, mode, width, precision
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: value_f64
        type(lr_operand_desc_t) :: field

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
        if (.not. emit_e_en_format_call(context%session, value, mode, precision, &
                                        width, field, error_msg)) return
        if (.not. emit_liric_print_string_operand_value(context%session, &
                context%str_print_format_id, field, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_formatted_e_en_real_item

    module subroutine read_decimal_value(digits, value, error_msg)
        character(len=*), intent(in) :: digits
        integer, intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_stat

        read (digits, *, iostat=io_stat) value
        if (io_stat /= 0) then
            error_msg = 'invalid decimal value in format descriptor'
            return
        end if
        call set_empty(error_msg)
    end subroutine read_decimal_value

    module subroutine parse_decimal_digits(text, pos, digits)
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos
        character(len=:), allocatable, intent(out) :: digits
        integer :: start

        start = pos
        do while (pos <= len_trim(text))
            if (.not. is_decimal_digit(text(pos:pos))) exit
            pos = pos + 1
        end do
        if (pos > start) then
            digits = text(start:pos - 1)
        else
            digits = ''
        end if
    end subroutine parse_decimal_digits

    module function is_decimal_digit(ch)
        character, intent(in) :: ch
        logical :: is_decimal_digit

        is_decimal_digit = ch >= '0' .and. ch <= '9'
    end function is_decimal_digit

    module subroutine skip_format_separators(text, pos)
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos
        do while (pos <= len_trim(text))
            if (text(pos:pos) /= ',' .and. text(pos:pos) /= ' ') exit
            pos = pos + 1
        end do
    end subroutine skip_format_separators

    module subroutine normalize_format_body(spec, body)
        character(len=*), intent(in) :: spec
        character(len=:), allocatable, intent(out) :: body
        character :: outer_quote
        integer :: n

        body = trim(adjustl(spec))
        outer_quote = ' '
        n = len(body)
        if (n >= 2) then
            if ((body(1:1) == "'" .and. body(n:n) == "'") .or. &
                (body(1:1) == '"' .and. body(n:n) == '"')) then
                outer_quote = body(1:1)
                if (n == 2) then
                    body = ''
                else
                    body = body(2:n - 1)
                end if
            end if
        end if
        ! The format literal's own delimiter is doubled inside the source token
        ! (e.g. '(...''...)' ); collapse it now so descriptor parsing sees the
        ! real text.
        if (outer_quote /= ' ') call collapse_doubled_quote(body, outer_quote)
        body = trim(adjustl(body))
        n = len(body)
        if (n >= 2 .and. body(1:1) == '(' .and. body(n:n) == ')') then
            if (n == 2) then
                body = ''
            else
                body = body(2:n - 1)
            end if
        end if
        body = trim(adjustl(body))
    end subroutine normalize_format_body

    module subroutine collapse_doubled_quote(text, quote)
        ! Replace every doubled quote (quote//quote) with a single quote.
        character(len=:), allocatable, intent(inout) :: text
        character, intent(in) :: quote
        character(len=:), allocatable :: out
        integer :: i, n

        n = len(text)
        out = ''
        i = 1
        do while (i <= n)
            if (i < n) then
                if (text(i:i) == quote .and. text(i + 1:i + 1) == quote) then
                    out = out//quote
                    i = i + 2
                    cycle
                end if
            end if
            out = out//text(i:i)
            i = i + 1
        end do
        text = out
    end subroutine collapse_doubled_quote

    module subroutine lower_formatted_int_item(arena, node_index, context, fmt_id, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: fmt_id
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32_value(context%session, fmt_id, value, &
                                             error_msg)) return
    end subroutine lower_formatted_int_item

    module subroutine lower_formatted_char_item(arena, node_index, context, fmt_id, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: fmt_id
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        integer :: symbol_index
        type(lr_operand_desc_t) :: data_ptr, length
        character(len=:), allocatable :: lit_value, lit_type

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'formatted print item does not reference an AST node'
            return
        end if
        if (is_literal(arena, node_index)) then
            if (is_character_literal(arena, node_index)) then
                call get_literal_info(arena, node_index, lit_value, lit_type, &
                                      error_msg)
                if (len_trim(error_msg) > 0) return
                call strip_literal_quotes(lit_value, character_value)
                context%string_literal_count = context%string_literal_count + 1
                string_name = ffc_unit_global_name(context, 'str.', &
                                                   context%string_literal_count)
                if (.not. emit_liric_print_string_value(context%session, fmt_id, &
                        trim(string_name), character_value, error_msg)) return
                return
            end if
            error_msg = 'unsupported character argument for the A edit descriptor'
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, lit_value, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, lit_value)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER &
                    .and. context%symbols(symbol_index)%has_character_value) then
                    call char_expr_operands(arena, node_index, context, &
                                            data_ptr, length, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (context%symbols(symbol_index)%is_dummy_argument) then
                        block
                            type(lr_operand_desc_t) :: view
                            call materialize_character_print_view(context, data_ptr, &
                                length, view, error_msg)
                            if (len_trim(error_msg) > 0) return
                            data_ptr = view
                        end block
                    end if
                    if (.not. emit_liric_print_string_operand_value( &
                        context%session, fmt_id, data_ptr, error_msg)) return
                    return
                end if
            end if
        end if
        if (is_char_expr_call(arena, node_index, context) .or. &
            is_character_concat(arena, node_index, context)) then
            block
                logical :: array_handled
                call emit_formatted_character_array_expression(arena, node_index, &
                    context, fmt_id, array_handled, error_msg)
                if (len_trim(error_msg) > 0) return
                if (array_handled) return
            end block
            call char_expr_operands(arena, node_index, context, data_ptr, &
                                    length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_string_operand_value(context%session, &
                fmt_id, data_ptr, error_msg)) return
            return
        end if
        error_msg = 'unsupported character argument for the A edit descriptor'
    end subroutine lower_formatted_char_item

    module procedure char_print_item
        ! Whether a print item evaluates to a character value. Used to suppress
        ! the list-directed separator between two consecutive character values.
        integer :: symbol_index
        character(len=:), allocatable :: id_name, id_err

        is_char = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_literal(arena, node_index)) then
            is_char = is_character_literal(arena, node_index)
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = find_symbol_compat(context, id_name)
            if (symbol_index > 0) &
                is_char = context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
            return
        end if
        if (is_character_concat(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_character_substring(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%is_array_access) then
                if (array_access_value_kind(node, context) == VALUE_CHARACTER) then
                    is_char = .true.
                    return
                end if
            end if
            ! A subscripted reference to a declared character array is a
            ! character element whether or not the parser marked the node as an
            ! array access; the symbol table is the authority on what the name
            ! is, so ask it rather than trusting the flag.
            if (allocated(node%name)) then
                block
                    integer :: sym
                    sym = find_symbol_compat(context, node%name)
                    if (sym > 0) then
                        if (context%symbols(sym)%is_array) then
                            if (context%symbols(sym)%value_kind == &
                                VALUE_CHARACTER) then
                                is_char = .true.
                                return
                            end if
                        end if
                    end if
                end block
            end if
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_deferred_char_function(context, node%name)) then
                    is_char = .true.
                    return
                end if
            end if
            is_char = is_char_expr_call(arena, node_index, context)
        type is (component_access_node)
            is_char = derived_component_access_kind(arena, node, context) == &
                      VALUE_CHARACTER
        class default
            is_char = .false.
        end select
    end procedure char_print_item
end submodule session_program_lowering_print_ops
