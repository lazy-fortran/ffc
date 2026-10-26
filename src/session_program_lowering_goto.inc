    ! First pass: scan node_indices for any statement with a stmt_label and
    ! pre-allocate a LIRIC block for each distinct label found.  This must run
    ! before lowering so that forward GOTO references are always valid.
    subroutine collect_goto_labels(arena, node_indices, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer :: i
        character(len=:), allocatable :: lbl

        ! Do not reset count: accumulate labels across nested calls so that
        ! inner lower_statement_list calls for sub-bodies do not wipe labels
        ! collected for the enclosing body.
        do i = 1, size(node_indices)
            if (.not. node_exists(arena, node_indices(i))) cycle
            if (.not. allocated(arena%entries(node_indices(i))%node%stmt_label)) cycle
            lbl = trim(arena%entries(node_indices(i))%node%stmt_label)
            if (len_trim(lbl) == 0) cycle
            if (find_goto_label_block(context, lbl) > 0) cycle
            if (context%goto_label_count >= 64) cycle
            context%goto_label_count = context%goto_label_count + 1
            context%goto_label_names(context%goto_label_count) = lbl
            context%goto_label_blocks(context%goto_label_count) = &
                create_liric_block(context%session)
            write(0,*) 'DEBUG collect_goto_labels: found label: ', trim(lbl), &
                       ' block=', context%goto_label_blocks(context%goto_label_count)
        end do
        write(0,*) 'DEBUG collect_goto_labels: total labels=', context%goto_label_count
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

        if (node%selector_index > 0 .and. allocated(node%label_list)) then
            call lower_computed_goto(node, context, error_msg)
            return
        end if
        if (.not. allocated(node%label) .or. len_trim(node%label) == 0) then
            error_msg = 'goto_node has no label'
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

    ! Emit a computed GOTO: goto (l1, l2, ..., ln), expr
    ! Branches to label l_i when expr == i; falls through for out-of-range.
    subroutine lower_computed_goto(node, context, error_msg)
        type(goto_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: list, tok
        character(len=64) :: labels(64)
        integer :: label_count, pos, comma, i
        integer(c_int32_t) :: target_block, fall_block
        type(lr_operand_desc_t) :: selector, idx_val, cond

        call lower_i32_expression(context%arena, node%selector_index, context, &
                                  selector, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Parse comma-separated label_list into labels(:).
        list = trim(node%label_list)
        label_count = 0
        pos = 1
        do while (pos <= len(list))
            comma = index(list(pos:), ',')
            if (comma == 0) then
                tok = trim(adjustl(list(pos:)))
                if (len_trim(tok) > 0) then
                    label_count = label_count + 1
                    labels(label_count) = tok
                end if
                exit
            else
                tok = trim(adjustl(list(pos:pos + comma - 2)))
                if (len_trim(tok) > 0) then
                    label_count = label_count + 1
                    labels(label_count) = tok
                end if
                pos = pos + comma
            end if
        end do

        if (label_count == 0) then
            error_msg = 'computed GOTO has empty label list'
            return
        end if

        fall_block = create_liric_block(context%session)

        ! Emit compare-and-branch cascade: for i=1..n, if selector==i goto labels(i).
        do i = 1, label_count
            target_block = find_goto_label_block(context, trim(labels(i)))
            if (target_block == 0_c_int32_t) then
                error_msg = 'computed GOTO target label not found: '// &
                            trim(labels(i))
                return
            end if
            idx_val = i32_immediate(context%session, int(i, c_int64_t))
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, selector, &
                                          idx_val, cond, error_msg)) return
            if (i < label_count) then
                ! Branch to target or continue checking.
                block
                    integer(c_int32_t) :: next_check_block
                    next_check_block = create_liric_block(context%session)
                    if (.not. emit_liric_condbr(context%session, cond, target_block, &
                                                next_check_block, error_msg)) return
                    if (.not. set_liric_block(context%session, next_check_block, &
                                             error_msg)) return
                    context%current_block_id = next_check_block
                end block
            else
                ! Last entry: branch to target or fall through.
                if (.not. emit_liric_condbr(context%session, cond, target_block, &
                                            fall_block, error_msg)) return
            end if
        end do

        if (.not. set_liric_block(context%session, fall_block, error_msg)) return
        context%current_block_id = fall_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine lower_computed_goto
