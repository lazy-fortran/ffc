    ! FORALL construct: indexed elementwise array assignment.
    !
    ! FORALL (i=lo:hi[, ...][, mask]) body  lowers to a sequential loop nest,
    ! one counted loop per index, with the optional mask evaluated per iteration
    ! as an IF guard around the body. The standard permits any execution order
    ! for the index set, so serial nested loops give a conforming result for the
    ! supported single-target-assignment bodies. Array writes land in memory, so
    ! no cross-iteration value carrying is needed beyond the loop inductions that
    ! lower_counted_loop already threads.

    recursive subroutine lower_forall(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k, sym

        call set_empty(error_msg)
        if (node%num_indices < 1) then
            call unsupported_feature_error('forall construct', &
                                           node%line, node%column, &
                                           'FORALL requires at least one index', &
                                           error_msg)
            return
        end if
        if (.not. allocated(node%index_names)) then
            error_msg = 'FORALL node has no index names'
            return
        end if
        if (.not. allocated(node%lower_bound_indices) .or. &
            .not. allocated(node%upper_bound_indices)) then
            error_msg = 'FORALL node is missing bound indices'
            return
        end if

        ! Ensure every index has an i32 loop symbol before lowering any bounds.
        do k = 1, node%num_indices
            sym = find_symbol(context, trim(node%index_names(k)))
            if (sym <= 0) then
                call define_i32_symbol(context, trim(node%index_names(k)), &
                                       error_msg)
                if (len_trim(error_msg) > 0) return
            else if (context%symbols(sym)%value_kind /= VALUE_I32) then
                call unsupported_feature_error('forall index', node%line, &
                                               node%column, &
                                               'FORALL index must be integer', &
                                               error_msg)
                return
            end if
        end do

        call lower_forall_level(arena, node, 1, context, error_msg)
    end subroutine lower_forall

    recursive subroutine lower_forall_level(arena, node, level, context, error_msg)
        ! Emit the counted loop for index `level`; its body recurses to the next
        ! index, and the innermost level emits the masked body.
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        integer, intent(in) :: level
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: stride_index

        stride_index = 0
        if (allocated(node%stride_indices)) stride_index = node%stride_indices(level)
        call lower_counted_loop(arena, trim(node%index_names(level)), &
                                node%lower_bound_indices(level), &
                                node%upper_bound_indices(level), stride_index, &
                                node%line, node%column, emit_forall_inner, &
                                context, value, error_msg)
    contains
        subroutine emit_forall_inner(ctx, terminated, err)
            type(lowering_context_t), intent(inout) :: ctx
            logical, intent(out) :: terminated
            character(len=:), allocatable, intent(out) :: err

            terminated = .false.
            if (level < node%num_indices) then
                call lower_forall_level(arena, node, level + 1, ctx, err)
            else
                call lower_forall_body(arena, node, ctx, err)
            end if
        end subroutine emit_forall_inner
    end subroutine lower_forall_level

    subroutine lower_forall_body(arena, node, context, error_msg)
        ! Innermost body: lower each body assignment, guarded by the mask when
        ! present. The guard is an IF over the scalar mask; array stores in the
        ! taken branch reach memory directly.
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        logical :: terminated

        call set_empty(error_msg)
        if (.not. allocated(node%body_indices)) return

        if (node%has_mask .and. node%mask_expr_index > 0) then
            call lower_forall_masked_body(arena, node, context, error_msg)
            return
        end if

        call lower_statement_list(arena, node%body_indices, context, value, &
                                  terminated, error_msg)
        if (len_trim(error_msg) > 0) return
        if (terminated) then
            call unsupported_feature_error('forall body', node%line, node%column, &
                                           'direct LIRIC session does not support '// &
                                           'stop or return inside a FORALL', &
                                           error_msg)
        end if
    end subroutine lower_forall_body

    subroutine lower_forall_masked_body(arena, node, context, error_msg)
        ! Guard the body statements with a conditional branch on the mask, like
        ! a single-arm IF. The body runs only where the mask holds; merge_block
        ! continues the enclosing loop body.
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: value
        integer(c_int32_t) :: body_block, merge_block
        logical :: terminated

        call lower_i1_condition(arena, node%mask_expr_index, context, condition, &
                                error_msg)
        if (len_trim(error_msg) > 0) return

        body_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                                    merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        call lower_statement_list(arena, node%body_indices, context, value, &
                                  terminated, error_msg)
        if (len_trim(error_msg) > 0) return
        if (terminated) then
            call unsupported_feature_error('forall body', node%line, node%column, &
                                           'direct LIRIC session does not support '// &
                                           'stop or return inside a FORALL', &
                                           error_msg)
            return
        end if
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine lower_forall_masked_body
