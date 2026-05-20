    subroutine lower_print(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: expr_count

        if (.not. allocated(node%expression_indices)) then
            call set_empty(error_msg)
            return
        end if

        expr_count = size(node%expression_indices)
        do i = 1, expr_count
            call lower_print_expression_value(arena, node%expression_indices(i), &
                                              context, error_msg)
            if (len_trim(error_msg) > 0) return
            if (i < expr_count) then
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
            else
                if (.not. emit_liric_print_newline(context%session, error_msg)) &
                    return
            end if
        end do

        call set_empty(error_msg)
    end subroutine lower_print
