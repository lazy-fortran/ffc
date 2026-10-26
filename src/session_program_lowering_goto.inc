    ! First pass: scan node_indices for any statement with a stmt_label and
    ! pre-allocate a LIRIC block for each distinct label found.  This must run
    ! before lowering so that forward GOTO references are always valid.
    subroutine collect_goto_labels(arena, node_indices, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer :: i
        character(len=:), allocatable :: lbl

        context%goto_label_count = 0
        do i = 1, size(node_indices)
            if (.not. node_exists(arena, node_indices(i))) cycle
            if (.not. allocated(arena%entries(node_indices(i))%node%stmt_label)) cycle
            lbl = trim(arena%entries(node_indices(i))%node%stmt_label)
            if (len(lbl) == 0) cycle
            if (find_goto_label_block(context, lbl) > 0) cycle
            if (context%goto_label_count >= 64) cycle
            context%goto_label_count = context%goto_label_count + 1
            context%goto_label_names(context%goto_label_count) = lbl
            context%goto_label_blocks(context%goto_label_count) = &
                create_liric_block(context%session)
        end do
    end subroutine collect_goto_labels

    ! Return the pre-allocated block id for a label, or 0 if not found.
    function find_goto_label_block(context, label) result(block_id)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: label
        integer(c_int32_t) :: block_id
        integer :: i

        block_id = 0_c_int32_t
        do i = 1, context%goto_label_count
            if (trim(context%goto_label_names(i)) == trim(label)) then
                block_id = context%goto_label_blocks(i)
                return
            end if
        end do
    end function find_goto_label_block

    ! Called before emitting a labeled statement. Closes the predecessor block
    ! (via an unconditional fallthrough branch) and switches the session to the
    ! pre-allocated block for this label.
    subroutine enter_labeled_block(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lbl
        integer(c_int32_t) :: target_block

        lbl = trim(arena%entries(node_index)%node%stmt_label)
        target_block = find_goto_label_block(context, lbl)
        if (target_block == 0_c_int32_t) then
            error_msg = 'goto label block not found for label: '//lbl
            return
        end if
        ! Emit fallthrough branch from the predecessor unless already terminated.
        if (.not. context%current_block_terminated) then
            if (.not. emit_liric_br(context%session, target_block, error_msg)) return
        end if
        if (.not. set_liric_block(context%session, target_block, error_msg)) return
        context%current_block_id = target_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine enter_labeled_block

    ! Emit an unconditional branch to the target label's block.
    subroutine lower_goto(node, context, error_msg)
        type(goto_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: target_block
        character(len=:), allocatable :: lbl

        if (.not. allocated(node%label) .or. len_trim(node%label) == 0) then
            error_msg = 'goto_node has no label (computed GOTO not supported)'
            return
        end if
        lbl = trim(node%label)
        target_block = find_goto_label_block(context, lbl)
        if (target_block == 0_c_int32_t) then
            error_msg = 'GOTO target label not found: '//lbl
            return
        end if
        if (.not. emit_liric_br(context%session, target_block, error_msg)) return
        context%current_block_terminated = .true.
        call set_empty(error_msg)
    end subroutine lower_goto

