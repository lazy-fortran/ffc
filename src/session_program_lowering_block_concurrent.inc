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
        integer :: saved_floor

        call set_empty(error_msg)
        saved_symbol_count = context%symbol_count
        saved_floor = context%block_scope_floor
        ! Everything declared so far is an enclosing scope for this block. A
        ! BLOCK-local declaration that reuses an outer name then lands in a fresh
        ! slot above the floor, so its writes never touch the outer storage,
        ! while writes to a non-shadowed outer variable still update it (#280).
        context%block_scope_floor = saved_symbol_count
        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
            if (terminated) context%current_block_terminated = terminated
        end if
        ! Discard BLOCK-local symbols so they are not visible outside; a shadowed
        ! outer name reverts to its (untouched) outer slot automatically.
        context%symbol_count = saved_symbol_count
        context%block_scope_floor = saved_floor
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
