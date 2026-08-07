submodule (session_program_lowering_impl) session_program_lowering_forall
    implicit none
contains
    ! FORALL construct: indexed elementwise array assignment.
    !
    ! FORALL (i=lo:hi[, ...][, mask]) uses a sequential loop nest, with one
    ! counted loop per index and the optional mask evaluated per iteration. A
    ! FORALL assignment is nevertheless an array operation: all RHS values see
    ! the target's pre-construct state. prepare_forall_snapshot therefore
    ! copies each fixed-size target before its statement loop and the array read
    ! path temporarily uses that copy (#673). Multiple body statements are
    ! emitted as separate loop nests, so statement 2 starts after statement 1
    ! has completed for the whole index set.

    recursive subroutine lower_forall(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k, sym, body_pos

        call set_empty(error_msg)
        context%forall_body_statement_index = 0
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
            sym = find_symbol_compat(context, trim(node%index_names(k)))
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

        if (.not. allocated(node%body_indices)) then
            call lower_forall_level(arena, node, 1, context, error_msg)
            return
        end if

        ! F2018 11.1.7.5 sequences FORALL body statements by statement, not by
        ! iteration. Select one body node at a time and give it an independent
        ! snapshot. This also keeps the common single-statement path compact.
        if (size(node%body_indices) <= 1) then
            context%forall_body_statement_index = 0
            call prepare_forall_snapshot(arena, node, context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_forall_level(arena, node, 1, context, error_msg)
            call clear_forall_snapshot(context)
            return
        end if

        do body_pos = 1, size(node%body_indices)
            context%forall_body_statement_index = node%body_indices(body_pos)
            call prepare_forall_snapshot(arena, node, context, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_forall_level(arena, node, 1, context, error_msg)
            call clear_forall_snapshot(context)
            if (len_trim(error_msg) > 0) return
        end do
        context%forall_body_statement_index = 0
    end subroutine lower_forall

    subroutine prepare_forall_snapshot(arena, node, context, error_msg)
        ! Materialise one fixed-size intrinsic target before lowering a FORALL
        ! statement. The snapshot is deliberately a raw stack copy: it uses
        ! the target's existing element address and byte width, so no second
        ! array descriptor or ownership protocol is introduced.
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: body_index, symbol_index, value_kind
        integer(c_int64_t) :: element_bytes, total_bytes
        character(len=:), allocatable :: target_name

        call clear_forall_snapshot(context)
        call set_empty(error_msg)
        body_index = context%forall_body_statement_index
        if (body_index <= 0) then
            if (.not. allocated(node%body_indices)) return
            if (size(node%body_indices) /= 1) return
            body_index = node%body_indices(1)
        end if
        if (.not. node_exists(arena, body_index)) return
        select type (statement => arena%entries(body_index)%node)
            type is (assignment_node)
            if (.not. node_exists(arena, statement%target_index)) return
            select type (target => arena%entries(statement%target_index)%node)
                type is (call_or_subscript_node)
                if (.not. allocated(target%name)) return
                target_name = trim(target%name)
            class default
                return
            end select
        class default
            return
        end select

        symbol_index = find_symbol_compat(context, target_name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_array) return
        if (context%symbols(symbol_index)%is_allocatable) return
        if (context%symbols(symbol_index)%array_size <= 0) return
        value_kind = context%symbols(symbol_index)%value_kind
        select case (value_kind)
        case (VALUE_I8)
            element_bytes = 1_c_int64_t
        case (VALUE_I16)
            element_bytes = 2_c_int64_t
        case (VALUE_I32, VALUE_F32, VALUE_LOGICAL)
            element_bytes = 4_c_int64_t
        case (VALUE_I64, VALUE_F64)
            element_bytes = 8_c_int64_t
        case default
            return
        end select
        total_bytes = int(context%symbols(symbol_index)%array_size, c_int64_t) * &
            element_bytes
        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, total_bytes), &
            context%forall_snapshot_address, error_msg)) return
        if (.not. emit_memcpy(context%session, context%forall_snapshot_address, &
            context%symbols(symbol_index)%element_address, &
            i64_immediate(context%session, total_bytes), error_msg)) return
        context%forall_snapshot_symbol = symbol_index
        context%forall_snapshot_reads = .true.
    end subroutine prepare_forall_snapshot

    subroutine clear_forall_snapshot(context)
        type(lowering_context_t), intent(inout) :: context
        context%forall_snapshot_reads = .false.
        context%forall_snapshot_writes = .false.
        context%forall_snapshot_symbol = 0
        context%forall_snapshot_address = lr_operand_desc_t()
    end subroutine clear_forall_snapshot

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
        integer :: selected_body(1)

        call set_empty(error_msg)
        if (.not. allocated(node%body_indices)) return

        if (node%has_mask .and. node%mask_expr_index > 0) then
            call lower_forall_masked_body(arena, node, context, error_msg)
            return
        end if

        if (context%forall_body_statement_index > 0) then
            selected_body(1) = context%forall_body_statement_index
            call lower_statement_list(arena, selected_body, context, value, &
                terminated, error_msg)
        else
            call lower_statement_list(arena, node%body_indices, context, value, &
                terminated, error_msg)
        end if
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
        integer :: selected_body(1)

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
        if (context%forall_body_statement_index > 0) then
            selected_body(1) = context%forall_body_statement_index
            call lower_statement_list(arena, selected_body, context, value, &
                terminated, error_msg)
        else
            call lower_statement_list(arena, node%body_indices, context, value, &
                terminated, error_msg)
        end if
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

end submodule session_program_lowering_forall
