    subroutine lower_do_loop(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: start_value
        type(lr_operand_desc_t) :: end_value
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: step_operand
        type(lr_operand_desc_t) :: next_index
        type(lr_operand_desc_t) :: copied_value
        type(lr_operand_desc_t) :: entry_values(MAX_SYMBOLS)
        type(lr_operand_desc_t) :: header_values(MAX_SYMBOLS)
        type(lr_operand_desc_t) :: backedge_values(MAX_SYMBOLS)
        integer(c_int64_t) :: step_value
        integer(c_int32_t) :: entry_block
        integer(c_int32_t) :: header_block
        integer(c_int32_t) :: body_block
        integer(c_int32_t) :: latch_block
        integer(c_int32_t) :: exit_block
        integer(c_int32_t) :: reserved_vreg
        integer :: loop_symbol_index
        integer :: carried_indices(MAX_SYMBOLS)
        integer :: carried_count
        integer :: initial_symbol_count
        integer :: i
        logical :: body_terminated

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
        if (context%symbols(loop_symbol_index)%value_kind /= VALUE_I32) then
            error_msg = 'direct LIRIC session DO loop variable must be integer'
            return
        end if

        call lower_i32_expression(arena, node%start_expr_index, context, &
                                  start_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%end_expr_index, context, &
                                  end_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_do_step(arena, node%step_expr_index, step_value, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(loop_symbol_index)%value = start_value
        entry_block = context%current_block_id
        initial_symbol_count = context%symbol_count
        call collect_i32_like_carried_symbols(context, carried_indices, &
                                              carried_count)
        do i = 1, carried_count
            entry_values(carried_indices(i)) = &
                context%symbols(carried_indices(i))%value
            call reserve_backedge_value(context, reserved_vreg, error_msg)
            if (len_trim(error_msg) > 0) return
            backedge_values(carried_indices(i)) = &
                context%session%i32_vreg(reserved_vreg)
        end do

        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.
        do i = 1, carried_count
            if (.not. emit_liric_i32_phi(context%session, &
                                         entry_values(carried_indices(i)), &
                                         entry_block, &
                                         backedge_values(carried_indices(i)), &
                                         latch_block, &
                                         header_values(carried_indices(i)), &
                                         error_msg)) return
            context%symbols(carried_indices(i))%value = &
                header_values(carried_indices(i))
        end do

        call lower_do_condition(context, loop_symbol_index, end_value, &
                                step_value, condition, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                                    exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        body_terminated = .false.
        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      body_terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (context%symbol_count /= initial_symbol_count) then
            error_msg = 'direct LIRIC session DO cannot merge loop declarations'
            return
        end if
        if (body_terminated) then
            call unsupported_feature_error('terminating do loop body', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support stop or return inside '// &
                                           'counted do loops', error_msg)
            return
        end if
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.

        step_operand = context%session%i32_immediate(step_value)
        reserved_vreg = int(backedge_values(loop_symbol_index)%payload, &
                            c_int32_t)
        if (.not. context%session%emit_i32_binary_into( &
            LR_OP_ADD, header_values(loop_symbol_index), step_operand, &
            reserved_vreg, next_index, error_msg)) return
        do i = 1, carried_count
            if (carried_indices(i) == loop_symbol_index) cycle
            if (.not. is_i32_like_kind( &
                context%symbols(carried_indices(i))%value_kind)) then
                error_msg = 'direct LIRIC session DO only carries integer '// &
                            'or logical symbols'
                return
            end if
            if (.not. context%session%emit_i32_copy_to( &
                context%symbols(carried_indices(i))%value, &
                int(backedge_values(carried_indices(i))%payload, c_int32_t), &
                copied_value, &
                error_msg)) return
        end do
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        do i = 1, carried_count
            context%symbols(carried_indices(i))%value = &
                header_values(carried_indices(i))
        end do
        value = context%session%i32_immediate(0_c_int64_t)
        call set_empty(error_msg)
    end subroutine lower_do_loop

    subroutine collect_i32_like_carried_symbols(context, indices, count)
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: indices(MAX_SYMBOLS)
        integer, intent(out) :: count
        integer :: i

        count = 0
        do i = 1, context%symbol_count
            if (is_i32_like_kind(context%symbols(i)%value_kind)) then
                count = count + 1
                indices(count) = i
            end if
        end do
    end subroutine collect_i32_like_carried_symbols

    logical function is_i32_like_kind(value_kind)
        integer, intent(in) :: value_kind

        is_i32_like_kind = value_kind == VALUE_I32 .or. &
                           value_kind == VALUE_LOGICAL
    end function is_i32_like_kind

    subroutine reserve_backedge_value(context, vreg, error_msg)
        type(lowering_context_t), intent(in) :: context
        integer(c_int32_t), intent(out) :: vreg
        character(len=:), allocatable, intent(out) :: error_msg

        vreg = context%session%reserve_i32_vreg()
        if (vreg == 0_c_int32_t) vreg = context%session%reserve_i32_vreg()
        if (vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a positive backedge vreg'
        else
            call set_empty(error_msg)
        end if
    end subroutine reserve_backedge_value

    subroutine lower_do_condition(context, loop_symbol_index, end_value, &
                                  step_value, condition, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: loop_symbol_index
        type(lr_operand_desc_t), intent(in) :: end_value
        integer(c_int64_t), intent(in) :: step_value
        type(lr_operand_desc_t), intent(out) :: condition
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int) :: pred

        if (step_value > 0_c_int64_t) then
            pred = LR_CMP_SLE
        else
            pred = LR_CMP_SGE
        end if
        if (.not. emit_liric_i32_icmp(context%session, pred, &
                                      context%symbols(loop_symbol_index)%value, &
                                      end_value, condition, error_msg)) return
    end subroutine lower_do_condition

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
            call unsupported_feature_error('do loop step', &
                                           arena%get_node_line(step_expr_index), &
                                           arena%get_node_column(step_expr_index), &
                                           'direct LIRIC session requires a '// &
                                           'nonzero literal step', error_msg)
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
            call unsupported_feature_error('do loop '//trim(name), &
                                           arena%get_node_line(node_index), &
                                           arena%get_node_column(node_index), &
                                           'direct LIRIC session only '// &
                                           'supports literal steps', error_msg)
        end select
    end subroutine parse_i32_constant
