    subroutine lower_print(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: expr_count
        logical :: prev_is_char
        logical :: cur_is_char

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

    logical function char_print_item(arena, node_index, context) result(is_char)
        ! Whether a print item evaluates to a character value. Used to suppress
        ! the list-directed separator between two consecutive character values.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_char = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            is_char = is_character_literal(node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) &
                is_char = context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
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
