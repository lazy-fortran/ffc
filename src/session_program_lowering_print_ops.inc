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
        ! every value; the first blank also serves as the record's carriage
        ! control. Each value field carries no leading blank of its own. The
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
        character :: kind_char
        integer(c_int32_t) :: fmt_id
        integer :: i
        character(len=64) :: fmt_name

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
        s = trim(adjustl(spec))
        if (len(s) >= 2) then
            if ((s(1:1) == "'" .and. s(len(s):len(s)) == "'") .or. &
                (s(1:1) == '"' .and. s(len(s):len(s)) == '"')) s = s(2:len(s) - 1)
        end if
        s = trim(adjustl(s))
        if (len(s) >= 2) then
            if (s(1:1) == '(' .and. s(len(s):len(s)) == ')') s = s(2:len(s) - 1)
        end if
        s = trim(adjustl(s))
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
        width = trim(adjustl(s(2:)))
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
