    subroutine lower_do_loop(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: start_value
        integer(c_int64_t) :: end_value
        integer(c_int64_t) :: step_value
        integer(c_int64_t) :: index_value
        integer :: loop_symbol_index
        integer :: i
        logical :: terminated

        value = context%session%i32_immediate(0_c_int64_t)
        if (.not. allocated(node%var_name)) then
            error_msg = 'direct LIRIC session DO requires a loop variable'
            return
        end if

        loop_symbol_index = find_symbol(context, node%var_name)
        if (loop_symbol_index <= 0) then
            error_msg = 'DO loop variable was not declared: '//trim(node%var_name)
            return
        end if

        call parse_i32_constant(arena, node%start_expr_index, start_value, &
                                'start', error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_i32_constant(arena, node%end_expr_index, end_value, 'end', &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_do_step(arena, node%step_expr_index, step_value, error_msg)
        if (len_trim(error_msg) > 0) return

        index_value = start_value
        terminated = .false.
        do while (loop_continues(index_value, end_value, step_value))
            context%symbols(loop_symbol_index)%value = &
                context%session%i32_immediate(index_value)
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call lower_statement(arena, node%body_indices(i), context, &
                                         value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (context%current_block_terminated) then
                        terminated = .true.
                        exit
                    end if
                end do
            end if
            if (terminated) return
            index_value = index_value + step_value
        end do

        context%symbols(loop_symbol_index)%value = &
            context%session%i32_immediate(index_value)
        value = context%session%i32_immediate(0_c_int64_t)
        call set_empty(error_msg)
    end subroutine lower_do_loop

    subroutine parse_do_step(arena, step_expr_index, step_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: step_expr_index
        integer(c_int64_t), intent(out) :: step_value
        character(len=:), allocatable, intent(out) :: error_msg

        if (step_expr_index <= 0) then
            step_value = 1_c_int64_t
            call set_empty(error_msg)
            return
        end if

        call parse_i32_constant(arena, step_expr_index, step_value, 'step', &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        if (step_value == 0_c_int64_t) then
            error_msg = 'direct LIRIC session DO does not support zero step'
        end if
    end subroutine parse_do_step

    subroutine parse_i32_constant(arena, node_index, value, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(out) :: value
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'DO loop '//trim(name)// &
                        ' index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            call parse_i32_literal(node%value, value, error_msg)
        class default
            error_msg = 'direct LIRIC session DO only supports literal bounds'
        end select
    end subroutine parse_i32_constant

    logical function loop_continues(value, end_value, step_value) result(continues)
        integer(c_int64_t), intent(in) :: value
        integer(c_int64_t), intent(in) :: end_value
        integer(c_int64_t), intent(in) :: step_value

        if (step_value > 0_c_int64_t) then
            continues = value <= end_value
        else
            continues = value >= end_value
        end if
    end function loop_continues
