    subroutine lower_print(arena, node, context, error_msg)
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
        if (expr_count == 1) then
            select type (section_node => arena%entries( &
                    node%expression_indices(1))%node)
            type is (array_slice_node)
                call lower_array_section_print(arena, section_node, context, &
                                               error_msg)
                return
            end select
            block
                logical :: handled
                call emit_array_literal_print_items(arena, &
                    node%expression_indices(1), context, handled, error_msg)
                if (len_trim(error_msg) > 0) return
                if (handled) then
                    if (.not. emit_liric_print_newline(context%session, error_msg)) &
                        return
                    call set_empty(error_msg)
                    return
                end if
                call lower_whole_array_print(arena, node%expression_indices(1), &
                                             context, handled, error_msg)
                if (handled .or. len_trim(error_msg) > 0) return
            end block
        end if
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
                    call emit_array_section_print_items(arena, item, context, &
                                                        array_handled, error_msg)
                type is (array_literal_node)
                    call emit_array_literal_print_items(arena, &
                        node%expression_indices(i), context, array_handled, &
                        error_msg)
                class default
                    call emit_whole_array_print_items(arena, &
                        node%expression_indices(i), context, array_handled, &
                        error_msg, prev_is_char)
                end select
                if (len_trim(error_msg) > 0) return
                if (array_handled) then
                    prev_is_char = char_print_item(arena, &
                        node%expression_indices(i), context)
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

    subroutine lower_formatted_print(arena, node, context, error_msg)
        ! Formatted print with an explicit literal format string. The compound
        ! lowerer walks every edit descriptor (data, control, string-literal),
        ! applies repeat counts, and reverts the format across records, so it is
        ! a strict superset of the single-descriptor case.
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: format_body

        call normalize_format_body(node%format_spec, format_body)
        if (len_trim(format_body) == 0) then
            ! An empty format still terminates one record.
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

    subroutine parse_single_edit_descriptor(spec, kind_char, printf_fmt, error_msg)
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

    subroutine lower_compound_formatted_print(arena, node, context, format_body, &
                                              error_msg)
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

        call set_empty(error_msg)
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

    recursive subroutine lower_next_compound_descriptor(arena, node, context, &
                                              format_body, pos, item_index, &
                                              exhausted, error_msg)
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
        case ('I')
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                error_msg = 'I edit descriptor requires width'
                return
            end if
            ! Iw.m: ignore the minimum-digit count m for printf width mapping.
            call skip_dot_modifier(format_body, pos)
            call read_decimal_value(width, width_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (width_value == 0) then
                printf_fmt = '%d'
            else
                write (width_text, '(I0)') width_value
                printf_fmt = '%'//trim(width_text)//'d'
            end if
            call repeat_data_descriptor(arena, node, context, kind_char, &
                                        printf_fmt, 0, repeat_count, item_index, &
                                        exhausted, error_msg)
        case ('F')
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0 .or. pos > len_trim(format_body) .or. &
                format_body(pos:pos) /= '.') then
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
            if (len(width) == 0 .or. pos > len_trim(format_body) .or. &
                format_body(pos:pos) /= '.') then
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
                printf_fmt = '%'//trim(width_text)//'s'
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

    recursive subroutine lower_compound_group(arena, node, context, format_body, &
                                              pos, repeat_count, item_index, &
                                              exhausted, error_msg)
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

    subroutine find_group_close(text, open_pos, close_pos, error_msg)
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

    subroutine skip_dot_modifier(format_body, pos)
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

    subroutine repeat_data_descriptor(arena, node, context, kind_char, &
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

    subroutine lower_format_string_literal(context, format_body, pos, error_msg)
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
                if (pos < n .and. format_body(pos + 1:pos + 1) == quote) then
                    text = text//quote
                    pos = pos + 2
                    cycle
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

    function ffc_unit_global_name(context, kind_tag, counter) result(name)
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

    subroutine lower_compound_logical_descriptor(arena, node, context, width, &
                                                 item_index, exhausted, error_msg)
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

    subroutine lower_compound_data_descriptor(arena, node, context, kind_char, &
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

    subroutine repeat_e_en_descriptor(arena, node, context, mode, width, &
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

    subroutine lower_formatted_real_item(arena, node_index, context, fmt_id, &
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

    subroutine lower_formatted_e_en_real_item(arena, node_index, context, mode, &
                                             width, precision, error_msg)
        use liric_session_format_bindings, only: emit_e_en_format_call
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, mode, width, precision
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: value_f64
        type(lr_operand_desc_t) :: field

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
        if (.not. emit_e_en_format_call(context%session, value, mode, precision, &
                                        width, field, error_msg)) return
        if (.not. emit_liric_print_string_operand_value(context%session, &
                context%str_print_format_id, field, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_formatted_e_en_real_item

    subroutine read_decimal_value(digits, value, error_msg)
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

    subroutine parse_decimal_digits(text, pos, digits)
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

    logical function is_decimal_digit(ch)
        character, intent(in) :: ch

        is_decimal_digit = ch >= '0' .and. ch <= '9'
    end function is_decimal_digit

    subroutine skip_format_separators(text, pos)
        character(len=*), intent(in) :: text
        integer, intent(inout) :: pos

        do while (pos <= len_trim(text))
            if (text(pos:pos) /= ',' .and. text(pos:pos) /= ' ') exit
            pos = pos + 1
        end do
    end subroutine skip_format_separators

    subroutine normalize_format_body(spec, body)
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

    subroutine collapse_doubled_quote(text, quote)
        ! Replace every doubled quote (quote//quote) with a single quote.
        character(len=:), allocatable, intent(inout) :: text
        character, intent(in) :: quote
        character(len=:), allocatable :: out
        integer :: i, n

        n = len(text)
        out = ''
        i = 1
        do while (i <= n)
            if (i < n .and. text(i:i) == quote .and. text(i + 1:i + 1) == quote) then
                out = out//quote
                i = i + 2
            else
                out = out//text(i:i)
                i = i + 1
            end if
        end do
        text = out
    end subroutine collapse_doubled_quote

    subroutine lower_formatted_int_item(arena, node_index, context, fmt_id, &
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

    subroutine lower_formatted_char_item(arena, node_index, context, fmt_id, &
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
            symbol_index = find_symbol(context, lit_value)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER &
                    .and. context%symbols(symbol_index)%has_character_value) then
                    if (.not. emit_liric_print_string_operand_value( &
                        context%session, fmt_id, &
                        context%symbols(symbol_index)%value, error_msg)) return
                    return
                end if
            end if
        end if
        if (is_char_expr_call(arena, node_index) .or. &
            is_character_concat(arena, node_index, context)) then
            call char_expr_operands(arena, node_index, context, data_ptr, &
                                    length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_string_operand_value(context%session, &
                fmt_id, data_ptr, error_msg)) return
            return
        end if
        error_msg = 'unsupported character argument for the A edit descriptor'
    end subroutine lower_formatted_char_item

    logical function char_print_item(arena, node_index, context) result(is_char)
        ! Whether a print item evaluates to a character value. Used to suppress
        ! the list-directed separator between two consecutive character values.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
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
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) &
                is_char = context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
            return
        end if
        if (is_character_concat(arena, node_index, context)) then
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
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_deferred_char_function(context, node%name)) then
                    is_char = .true.
                    return
                end if
            end if
            is_char = is_char_expr_call(arena, node_index)
        type is (component_access_node)
            is_char = derived_component_access_kind(arena, node, context) == &
                      VALUE_CHARACTER
        end select
    end function char_print_item
