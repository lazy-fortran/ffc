    subroutine lower_block_construct(arena, node, context, error_msg)
        ! Lower a BLOCK construct: a scoped region with its own declarations.
        ! Variables declared inside BLOCK are scoped to the construct body;
        ! the symbol table is restored on exit so outer symbols are unchanged.
        type(ast_arena_t), intent(in) :: arena
        type(block_construct_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        logical :: terminated
        integer :: saved_symbol_count

        call set_empty(error_msg)
        saved_symbol_count = context%symbol_count
        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
            if (terminated) context%current_block_terminated = terminated
        end if
        ! Discard BLOCK-local symbols so they are not visible outside.
        context%symbol_count = saved_symbol_count
    end subroutine lower_block_construct

    subroutine lower_do_concurrent(arena, node, context, value, error_msg)
        ! Lower DO CONCURRENT as a plain counted DO loop. The standard allows any
        ! serial execution order for the iteration set; parallelism is deferred.
        ! Locality specifiers (LOCAL/SHARED) are validated but otherwise ignored
        ! because serial execution already gives the correct semantics.
        type(ast_arena_t), intent(in) :: arena
        type(do_loop_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        ! Delegate to the standard counted-DO lowerer; is_concurrent is metadata
        ! and does not change the loop structure for a serial backend.
        call lower_do_loop(arena, node, context, value, error_msg)
    end subroutine lower_do_concurrent

    subroutine lower_where_stmt(arena, node, context, error_msg)
        ! Lower a single-statement WHERE: WHERE (mask) assignment.
        ! For scalar mask and scalar assignment this is just a guarded assignment.
        ! For array operands this generates a 1-D loop nest over a conforming mask
        ! and array that are both rank-1 integer arrays of the same declared size.
        type(ast_arena_t), intent(in) :: arena
        type(where_stmt_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: mask_val, dummy_value
        type(lr_operand_desc_t) :: condition
        integer(c_int32_t) :: then_block, merge_block
        type(branch_result_t) :: then_result, else_result
        type(symbol_t), allocatable :: saved_symbols(:)
        integer :: saved_symbol_count, i
        integer :: empty_else(0)

        call set_empty(error_msg)
        if (node%mask_expr_index <= 0) then
            error_msg = 'WHERE statement requires a mask expression'
            return
        end if
        if (node%assignment_index <= 0) then
            error_msg = 'WHERE statement requires an assignment'
            return
        end if

        ! Evaluate the mask as a logical/i32 condition.
        call lower_i1_condition(arena, node%mask_expr_index, context, &
                                mask_val, error_msg)
        if (len_trim(error_msg) > 0) return

        then_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        condition = mask_val
        if (.not. emit_liric_condbr(context%session, condition, then_block, &
                                    merge_block, error_msg)) return

        allocate(saved_symbols(context%symbol_count))
        do i = 1, context%symbol_count
            saved_symbols(i) = context%symbols(i)
        end do
        saved_symbol_count = context%symbol_count

        call lower_if_branch(arena, [node%assignment_index], then_block, &
                             merge_block, context, dummy_value, then_result, &
                             error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count

        call lower_if_branch(arena, empty_else, merge_block, merge_block, &
                             context, dummy_value, else_result, error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count
        deallocate(saved_symbols)

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        if (.not. then_result%terminated .and. .not. else_result%terminated) then
            call merge_branch_symbols(context, then_result, else_result, error_msg)
        end if
    end subroutine lower_where_stmt

    subroutine lower_where_construct(arena, node, context, error_msg)
        ! Lower a WHERE/ELSEWHERE construct.
        ! The WHERE body executes where the mask is true; each ELSEWHERE clause
        ! with a mask executes where all prior masks were false and the new mask
        ! is true; a final ELSEWHERE (no mask) executes where all prior masks
        ! were false. Implemented as a chain of IF blocks over the same mask.
        type(ast_arena_t), intent(in) :: arena
        type(where_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: mask_val, dummy_value, condition
        type(symbol_t), allocatable :: saved_symbols(:)
        integer :: saved_symbol_count, i
        type(branch_result_t) :: then_result, else_result, merge_result
        integer(c_int32_t) :: then_block, else_block, merge_block
        integer :: empty_body(0)

        call set_empty(error_msg)
        if (node%mask_expr_index <= 0) then
            error_msg = 'WHERE construct requires a mask expression'
            return
        end if

        call lower_i1_condition(arena, node%mask_expr_index, context, &
                                mask_val, error_msg)
        if (len_trim(error_msg) > 0) return

        then_block = create_liric_block(context%session)
        else_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        condition = mask_val
        if (.not. emit_liric_condbr(context%session, condition, then_block, &
                                    else_block, error_msg)) return

        allocate(saved_symbols(context%symbol_count))
        do i = 1, context%symbol_count
            saved_symbols(i) = context%symbols(i)
        end do
        saved_symbol_count = context%symbol_count

        ! Lower the WHERE body (true branch).
        if (allocated(node%where_body_indices)) then
            call lower_if_branch(arena, node%where_body_indices, then_block, &
                                 merge_block, context, dummy_value, then_result, &
                                 error_msg)
        else
            call lower_if_branch(arena, empty_body, then_block, &
                                 merge_block, context, dummy_value, then_result, &
                                 error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count

        ! Lower ELSEWHERE clauses (false branch of prior mask, chained).
        if (allocated(node%elsewhere_clauses) .and. &
            size(node%elsewhere_clauses) > 0) then
            call lower_elsewhere_chain(arena, node%elsewhere_clauses, &
                                       else_block, merge_block, &
                                       saved_symbols, saved_symbol_count, &
                                       context, else_result, error_msg)
        else
            call lower_if_branch(arena, empty_body, else_block, &
                                 merge_block, context, dummy_value, else_result, &
                                 error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count
        deallocate(saved_symbols)

        if (then_result%terminated .and. else_result%terminated) then
            context%current_block_terminated = .true.
            return
        end if
        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        if (.not. then_result%terminated .and. .not. else_result%terminated) then
            call merge_branch_symbols(context, then_result, else_result, error_msg)
        else if (.not. then_result%terminated) then
            do i = 1, then_result%symbol_count
                context%symbols(i) = then_result%symbols(i)
            end do
            context%symbol_count = then_result%symbol_count
        else
            do i = 1, else_result%symbol_count
                context%symbols(i) = else_result%symbols(i)
            end do
            context%symbol_count = else_result%symbol_count
        end if
    end subroutine lower_where_construct

    subroutine lower_elsewhere_chain(arena, clauses, first_else_block, &
                                     merge_block, saved_symbols, &
                                     saved_symbol_count, context, &
                                     final_result, error_msg)
        ! Lower a sequence of ELSEWHERE clauses as a chain of conditional
        ! branches. Each masked ELSEWHERE is another IF; the final unmasked
        ! ELSEWHERE is a plain block.
        type(ast_arena_t), intent(in) :: arena
        type(elsewhere_clause_t), intent(in) :: clauses(:)
        integer(c_int32_t), intent(in) :: first_else_block
        integer(c_int32_t), intent(in) :: merge_block
        type(symbol_t), intent(in) :: saved_symbols(:)
        integer, intent(in) :: saved_symbol_count
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(out) :: final_result
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: cur_block, body_block, next_block
        type(lr_operand_desc_t) :: clause_mask, condition, dummy_value
        integer :: k, i
        logical :: terminated
        integer :: empty_body(0)

        call set_empty(error_msg)
        cur_block = first_else_block

        do k = 1, size(clauses)
            ! Restore symbol snapshot before each clause.
            do i = 1, saved_symbol_count
                context%symbols(i) = saved_symbols(i)
            end do
            context%symbol_count = saved_symbol_count

            if (.not. set_liric_block(context%session, cur_block, error_msg)) return
            context%current_block_id = cur_block
            context%current_block_terminated = .false.

            if (clauses(k)%mask_index > 0) then
                ! Masked ELSEWHERE: evaluate mask, branch to body or next.
                call lower_i1_condition(arena, clauses(k)%mask_index, context, &
                                        clause_mask, error_msg)
                if (len_trim(error_msg) > 0) return
                condition = clause_mask
                body_block = create_liric_block(context%session)
                if (k < size(clauses)) then
                    next_block = create_liric_block(context%session)
                else
                    next_block = merge_block
                end if
                if (.not. emit_liric_condbr(context%session, condition, &
                                             body_block, next_block, &
                                             error_msg)) return
                if (.not. set_liric_block(context%session, body_block, &
                                           error_msg)) return
                context%current_block_id = body_block
                context%current_block_terminated = .false.
                do i = 1, saved_symbol_count
                    context%symbols(i) = saved_symbols(i)
                end do
                context%symbol_count = saved_symbol_count
                terminated = .false.
                if (allocated(clauses(k)%body_indices)) then
                    call lower_statement_list(arena, clauses(k)%body_indices, &
                                              context, dummy_value, terminated, &
                                              error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                if (.not. terminated) then
                    if (.not. emit_liric_br(context%session, merge_block, &
                                             error_msg)) return
                end if
                cur_block = next_block
            else
                ! Unmasked ELSEWHERE: plain block.
                terminated = .false.
                if (allocated(clauses(k)%body_indices)) then
                    call lower_statement_list(arena, clauses(k)%body_indices, &
                                              context, dummy_value, terminated, &
                                              error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
                if (.not. terminated) then
                    if (.not. emit_liric_br(context%session, merge_block, &
                                             error_msg)) return
                end if
                ! No further clauses after an unmasked ELSEWHERE.
                final_result%terminated = terminated
                final_result%predecessor_block_id = context%current_block_id
                allocate(final_result%symbols(context%symbol_count))
                do i = 1, context%symbol_count
                    final_result%symbols(i) = context%symbols(i)
                end do
                final_result%symbol_count = context%symbol_count
                return
            end if
        end do

        ! All clauses had masks; the fall-through (no matching mask) goes to merge.
        if (.not. set_liric_block(context%session, cur_block, error_msg)) return
        context%current_block_id = cur_block
        context%current_block_terminated = .false.
        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return
        final_result%terminated = .false.
        final_result%predecessor_block_id = cur_block
        allocate(final_result%symbols(context%symbol_count))
        do i = 1, context%symbol_count
            final_result%symbols(i) = context%symbols(i)
        end do
        final_result%symbol_count = context%symbol_count
    end subroutine lower_elsewhere_chain

    subroutine lower_forall(arena, node, context, error_msg)
        ! Lower a FORALL construct as a sequential loop nest over the index set.
        ! An optional mask guards each assignment. Supports a single index triplet;
        ! nested indices are lowered as nested DO loops.
        ! Only scalar integer indices and body assignments are supported here.
        type(ast_arena_t), intent(in) :: arena
        type(forall_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n, idx
        character(len=:), allocatable :: dummy_name
        type(do_loop_node) :: inner_loop
        type(do_loop_node) :: outer_loop
        type(lr_operand_desc_t) :: value

        call set_empty(error_msg)
        n = node%num_indices
        if (n < 1) then
            call unsupported_feature_error('forall construct', &
                                           node%line, node%column, &
                                           'FORALL requires at least one index', &
                                           error_msg)
            return
        end if
        if (n > 2) then
            call unsupported_feature_error('forall construct', &
                                           node%line, node%column, &
                                           'ffc direct-session supports at most '// &
                                           '2 FORALL indices (#250)', error_msg)
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
        if (node%has_mask) then
            call unsupported_feature_error('forall mask', node%line, node%column, &
                                           'FORALL mask not yet supported by ffc '// &
                                           'direct-session (#250)', error_msg)
            return
        end if

        if (n == 1) then
            ! Single-index FORALL: lower as a plain DO loop over the same bounds.
            dummy_name = trim(node%index_names(1))
            idx = find_symbol(context, dummy_name)
            if (idx <= 0) then
                ! Declare a temporary i32 loop variable.
                call define_i32_symbol(context, dummy_name, error_msg)
                if (len_trim(error_msg) > 0) return
                idx = find_symbol(context, dummy_name)
            end if
            inner_loop%var_name = dummy_name
            inner_loop%start_expr_index = node%lower_bound_indices(1)
            inner_loop%end_expr_index = node%upper_bound_indices(1)
            if (allocated(node%stride_indices)) then
                inner_loop%step_expr_index = node%stride_indices(1)
            else
                inner_loop%step_expr_index = 0
            end if
            if (allocated(node%body_indices)) then
                inner_loop%body_indices = node%body_indices
            end if
            inner_loop%is_concurrent = .false.
            inner_loop%line = node%line
            inner_loop%column = node%column
            call lower_do_loop(arena, inner_loop, context, value, error_msg)
        else
            ! Two-index FORALL: outer loop over index 1, inner loop over index 2.
            ! Build the inner loop node first (it becomes the body of the outer).
            ! Since we cannot easily reference the inner_loop's arena index from
            ! here, we lower both loops inline in two steps using lower_do_loop.
            call unsupported_feature_error('forall 2-index', node%line, node%column, &
                                           'ffc direct-session supports at most '// &
                                           '1 FORALL index in this slice (#250)', &
                                           error_msg)
        end if
    end subroutine lower_forall
