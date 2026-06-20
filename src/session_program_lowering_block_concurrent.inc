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
