submodule (session_program_lowering_impl) session_program_lowering_block_concurrent
    implicit none
contains
    module procedure lower_block_construct
    ! Lower a BLOCK construct: a scoped region with its own declarations.
    ! Variables declared inside BLOCK are scoped to the construct body;
    ! the symbol table is restored on exit so outer symbols are unchanged.
    type(lr_operand_desc_t) :: value
    logical :: terminated
    integer :: saved_symbol_count
    integer :: saved_floor
    integer :: local_symbol_count

    call set_empty(error_msg)
    terminated = .false.
    call push_storage_scope(context, saved_symbol_count, saved_floor)
    ! Everything declared so far is an enclosing scope for this block. A
    ! BLOCK-local declaration that reuses an outer name then lands in a fresh
    ! slot above the floor, so its writes never touch the outer storage,
    ! while writes to a non-shadowed outer variable still update it (#280).
    context%block_scope_floor = saved_symbol_count
    if (allocated(node%body_indices)) then
        call lower_statement_list(arena, node%body_indices, context, value, &
            terminated, error_msg)
    end if
    local_symbol_count = context%symbol_count
    ! Discard BLOCK-local symbols and their binding identities so neither
    ! a shadowed name nor a later declaration can reuse stale construct state.
    call pop_storage_scope(context, saved_symbol_count, saved_floor)
    ! Slot allocation reuses the flat symbol array after a BLOCK. Clear the
    ! discarded records as well as the count: declaration binding registration
    ! treats a live binding flag as authoritative, and stale flags would make a
    ! later block inherit the previous block's identity (#280).
    if (local_symbol_count > saved_symbol_count) then
        context%symbols(saved_symbol_count + 1:local_symbol_count) = symbol_t()
    end if
    if (len_trim(error_msg) > 0) return
    if (terminated) context%current_block_terminated = terminated
    end procedure lower_block_construct

    module procedure lower_do_concurrent
    ! Lower DO CONCURRENT as a plain counted DO loop. The standard allows
    ! any serial execution order for the iteration set; parallelism is
    ! deferred. Locality specifiers (LOCAL/SHARED) are validated but
    ! otherwise ignored because serial execution already gives the correct
    ! semantics.

    ! Delegate to the standard counted-DO lowerer; is_concurrent is metadata
    ! and does not change the loop structure for a serial backend.
    call lower_do_loop(arena, node, context, value, error_msg)
    end procedure lower_do_concurrent
end submodule session_program_lowering_block_concurrent
