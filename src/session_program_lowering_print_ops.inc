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
        ! Formatted print with an explicit literal format string: a single
        ! edit descriptor (Iw / A[w]) applied to each item, one record newline.
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: printf_fmt
        character(len=:), allocatable :: format_body
        character :: kind_char
        integer(c_int32_t) :: fmt_id
        integer :: i
        character(len=64) :: fmt_name

        call normalize_format_body(node%format_spec, format_body)
        if (index(format_body, ',') > 0) then
            call lower_compound_formatted_print(arena, node, context, &
                                                format_body, error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('formatted print', node%line, &
                                               node%column, trim(error_msg), &
                                               error_msg)
            end if
            return
        end if

        call parse_single_edit_descriptor(node%format_spec, kind_char, &
                                          printf_fmt, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('formatted print', node%line, &
                                           node%column, trim(error_msg), error_msg)
            return
        end if

        if (.not. allocated(node%expression_indices)) then
            if (.not. emit_liric_print_newline(context%session, error_msg)) return
            call set_empty(error_msg)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.fmt.user.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         printf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 1, size(node%expression_indices)
            if (kind_char == 'I') then
                call lower_formatted_int_item(arena, node%expression_indices(i), &
                                              context, fmt_id, error_msg)
            else
                call lower_formatted_char_item(arena, node%expression_indices(i), &
                                               context, fmt_id, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do
        if (.not. emit_liric_print_newline(context%session, error_msg)) return
        call set_empty(error_msg)
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
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: pos
        integer :: item_index

        call set_empty(error_msg)
        pos = 1
        item_index = 1
        do
            call skip_format_separators(format_body, pos)
            if (pos > len_trim(format_body)) exit
            call lower_next_compound_descriptor(arena, node, context, &
                                                format_body, pos, item_index, &
                                                error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (allocated(node%expression_indices)) then
            if (item_index <= size(node%expression_indices)) then
                error_msg = 'format reversion is not supported'
                return
            end if
        end if
        if (.not. emit_liric_print_newline(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_compound_formatted_print

    subroutine lower_next_compound_descriptor(arena, node, context, format_body, &
                                              pos, item_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: format_body
        integer, intent(inout) :: pos
        integer, intent(inout) :: item_index
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
        integer :: i
        character(len=32) :: width_text
        character(len=32) :: precision_text

        call set_empty(error_msg)
        call parse_decimal_digits(format_body, pos, prefix)
        if (pos > len_trim(format_body)) then
            error_msg = 'dangling repeat count in compound format'
            return
        end if
        kind_char = format_body(pos:pos)
        if (kind_char == 'i') kind_char = 'I'
        if (kind_char == 'f') kind_char = 'F'
        if (kind_char == 'x') kind_char = 'X'
        if (kind_char == 'a') kind_char = 'A'
        pos = pos + 1

        select case (kind_char)
        case ('X')
            repeat_count = 1
            if (len(prefix) > 0) then
                call read_decimal_value(prefix, repeat_count, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            do i = 1, repeat_count
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
            end do
            call set_empty(error_msg)
        case ('I')
            if (len(prefix) > 0) then
                error_msg = 'repeat counts are not supported for I descriptors'
                return
            end if
            call parse_decimal_digits(format_body, pos, width)
            if (len(width) == 0) then
                error_msg = 'I edit descriptor requires width'
                return
            end if
            call read_decimal_value(width, width_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (width_value == 0) then
                printf_fmt = '%d'
            else
                write (width_text, '(I0)') width_value
                printf_fmt = '%'//trim(width_text)//'d'
            end if
            call lower_compound_data_descriptor(arena, node, context, kind_char, &
                                                printf_fmt, 0, item_index, &
                                                error_msg)
        case ('F')
            if (len(prefix) > 0) then
                error_msg = 'repeat counts are not supported for F descriptors'
                return
            end if
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
            printf_fmt = '%'//trim(width_text)//'.'//trim(precision_text)//'f'
            buffer_size = max(64, width_value + precision_value + 32)
            call lower_compound_data_descriptor(arena, node, context, kind_char, &
                                                printf_fmt, buffer_size, &
                                                item_index, error_msg)
        case default
            error_msg = 'unsupported edit descriptor in compound format: '// &
                        kind_char
        end select
    end subroutine lower_next_compound_descriptor

    subroutine lower_compound_data_descriptor(arena, node, context, kind_char, &
                                              printf_fmt, buffer_size, &
                                              item_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character, intent(in) :: kind_char
        character(len=*), intent(in) :: printf_fmt
        integer, intent(in) :: buffer_size
        integer, intent(inout) :: item_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: fmt_id
        character(len=64) :: fmt_name

        if (.not. allocated(node%expression_indices)) then
            error_msg = 'formatted print item is missing'
            return
        end if
        if (item_index > size(node%expression_indices)) then
            error_msg = 'formatted print item is missing'
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.fmt.user.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         printf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return

        if (kind_char == 'I') then
            call lower_formatted_int_item(arena, node%expression_indices(item_index), &
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

    subroutine lower_formatted_real_item(arena, node_index, context, fmt_id, &
                                         buffer_size, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: fmt_id
        integer, intent(in) :: buffer_size
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: field
        type(lr_operand_desc_t) :: args(4)

        call lower_f64_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
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
        integer :: n

        body = trim(adjustl(spec))
        n = len(body)
        if (n >= 2) then
            if ((body(1:1) == "'" .and. body(n:n) == "'") .or. &
                (body(1:1) == '"' .and. body(n:n) == '"')) then
                if (n == 2) then
                    body = ''
                else
                    body = body(2:n - 1)
                end if
            end if
        end if
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
                write (string_name, '(A,I0)') '.ffc.str.', &
                    context%string_literal_count
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
        if (is_char_expr_call(arena, node_index)) then
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
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. node%is_array_access .and. allocated(node%name)) then
                if (is_contained_deferred_char_function(context, node%name)) then
                    is_char = .true.
                    return
                end if
            end if
            is_char = is_char_expr_call(arena, node_index)
        end select
    end function char_print_item
