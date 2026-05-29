    subroutine lower_do_while(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(do_while_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: copied_value
        type(lr_operand_desc_t), allocatable :: entry_values(:)
        type(lr_operand_desc_t), allocatable :: header_values(:)
        type(lr_operand_desc_t), allocatable :: backedge_values(:)
        integer(c_int32_t) :: entry_block
        integer(c_int32_t) :: header_block
        integer(c_int32_t) :: body_block
        integer(c_int32_t) :: exit_block
        integer(c_int32_t) :: reserved_vreg
        integer, allocatable :: carried_indices(:)
        integer :: carried_count
        integer :: initial_symbol_count
        integer :: i
        logical :: body_terminated

        value = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        initial_symbol_count = context%symbol_count
        call collect_i32_like_carried_symbols(context, carried_indices, &
                                               carried_count)
        if (carried_count > 0) then
            allocate(entry_values(context%symbol_count))
            allocate(header_values(context%symbol_count))
            allocate(backedge_values(context%symbol_count))
        end if
        do i = 1, carried_count
            entry_values(carried_indices(i)) = &
                context%symbols(carried_indices(i))%value
            call reserve_backedge_value(context, reserved_vreg, error_msg)
            if (len_trim(error_msg) > 0) return
            backedge_values(carried_indices(i)) = &
                i32_vreg(context%session, reserved_vreg)
        end do

        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
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
                                         body_block, &
                                         header_values(carried_indices(i)), &
                                         error_msg)) return
            context%symbols(carried_indices(i))%value = &
                header_values(carried_indices(i))
        end do

        call lower_i1_condition(arena, node%condition_index, context, &
                                condition, error_msg)
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
            error_msg = 'direct LIRIC session DO WHILE cannot merge '// &
                        'loop declarations'
            return
        end if
        if (body_terminated) then
            call unsupported_feature_error('terminating do while body', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support stop or return inside '// &
                                           'do while loops', error_msg)
            return
        end if
        do i = 1, carried_count
            if (.not. is_i32_like_kind( &
                context%symbols(carried_indices(i))%value_kind)) then
                error_msg = 'direct LIRIC session DO WHILE only carries '// &
                            'integer or logical symbols'
                return
            end if
            if (.not. emit_i32_copy_to(context%session, &
                context%symbols(carried_indices(i))%value, &
                int(backedge_values(carried_indices(i))%payload, c_int32_t), &
                copied_value, error_msg)) return
        end do
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        do i = 1, carried_count
            context%symbols(carried_indices(i))%value = &
                header_values(carried_indices(i))
        end do
        value = i32_immediate(context%session, 0_c_int64_t)
        if (carried_count > 0) then
            deallocate(entry_values)
            deallocate(header_values)
            deallocate(backedge_values)
        end if
        if (allocated(carried_indices)) then
            deallocate(carried_indices)
        end if
        call set_empty(error_msg)
    end subroutine lower_do_while
