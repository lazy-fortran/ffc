    subroutine lower_if(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(if_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: condition
        type(symbol_t), allocatable :: saved_symbols(:)
        integer :: saved_symbol_count
        integer(c_int32_t) :: then_block
        integer(c_int32_t) :: else_block
        integer(c_int32_t) :: merge_block
        type(branch_result_t) :: then_result
        type(branch_result_t) :: else_result
        integer :: i

        if (node%condition_index <= 0) then
            error_msg = 'direct LIRIC session IF requires a condition index'
            return
        end if
        if (.not. allocated(node%then_body_indices)) then
            error_msg = 'direct LIRIC session IF requires a THEN body'
            return
        end if

        call lower_i1_condition(arena, node%condition_index, context, &
                                condition, error_msg)
        if (len_trim(error_msg) > 0) return

        then_block = create_liric_block(context%session)
        else_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, condition, then_block, &
                                    else_block, error_msg)) return

        allocate(saved_symbols(context%symbol_count))
        do i = 1, context%symbol_count
            saved_symbols(i) = context%symbols(i)
        end do
        saved_symbol_count = context%symbol_count

        call lower_if_branch(arena, node%then_body_indices, then_block, &
                             merge_block, context, value, then_result, error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count
        if (allocated(node%else_body_indices)) then
            call lower_if_branch(arena, node%else_body_indices, else_block, &
                                 merge_block, context, value, else_result, &
                                 error_msg)
        else
            block
                integer :: empty_else(0)
                call lower_if_branch(arena, empty_else, else_block, &
                                     merge_block, context, value, else_result, &
                                     error_msg)
            end block
        end if
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count
        deallocate(saved_symbols)

        if (then_result%terminated .and. else_result%terminated) then
            context%current_block_terminated = .true.
            call set_empty(error_msg)
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
            call set_empty(error_msg)
        else
            do i = 1, else_result%symbol_count
                context%symbols(i) = else_result%symbols(i)
            end do
            context%symbol_count = else_result%symbol_count
            call set_empty(error_msg)
        end if
    end subroutine lower_if

    subroutine lower_if_branch(arena, node_indices, block_id, merge_block, &
                               context, value, result, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_indices(:)
        integer(c_int32_t), intent(in) :: block_id
        integer(c_int32_t), intent(in) :: merge_block
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        type(branch_result_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (.not. set_liric_block(context%session, block_id, error_msg)) return
        context%current_block_id = block_id

        call lower_statement_list(arena, node_indices, context, value, &
                                  result%terminated, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. allocated(result%symbols)) then
            allocate(result%symbols(context%symbol_count))
        end if
        do i = 1, context%symbol_count
            result%symbols(i) = context%symbols(i)
        end do
        result%symbol_count = context%symbol_count
        result%predecessor_block_id = context%current_block_id

        if (.not. result%terminated) then
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) return
        end if

        call set_empty(error_msg)
    end subroutine lower_if_branch

    subroutine merge_branch_symbols(context, then_result, else_result, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(in) :: then_result
        type(branch_result_t), intent(in) :: else_result
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (then_result%symbol_count /= else_result%symbol_count) then
            error_msg = 'direct LIRIC session IF cannot merge branch declarations'
            return
        end if

        context%symbol_count = then_result%symbol_count
        do i = 1, then_result%symbol_count
            if (trim(then_result%symbols(i)%name) /= &
                trim(else_result%symbols(i)%name)) then
                error_msg = 'direct LIRIC session IF branch symbols diverged'
                return
            end if

            context%symbols(i) = then_result%symbols(i)
            if (then_result%symbols(i)%value_kind /= &
                else_result%symbols(i)%value_kind) then
                error_msg = 'direct LIRIC session IF branch symbol types diverged'
                return
            end if
            if (same_operand(then_result%symbols(i)%value, &
                             else_result%symbols(i)%value)) then
                context%symbols(i)%value = then_result%symbols(i)%value
            else
                call merge_symbol_value(context, then_result, else_result, i, &
                                        error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do

        call set_empty(error_msg)
    end subroutine merge_branch_symbols

    subroutine merge_symbol_value(context, then_result, else_result, index, &
                                  error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(in) :: then_result
        type(branch_result_t), intent(in) :: else_result
        integer, intent(in) :: index
        character(len=:), allocatable, intent(out) :: error_msg

        select case (then_result%symbols(index)%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_liric_i32_phi(context%session, &
                                         then_result%symbols(index)%value, &
                                         then_result%predecessor_block_id, &
                                         else_result%symbols(index)%value, &
                                         else_result%predecessor_block_id, &
                                         context%symbols(index)%value, &
                                         error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_phi(context%session, &
                                     then_result%symbols(index)%value, &
                                     then_result%predecessor_block_id, &
                                     else_result%symbols(index)%value, &
                                     else_result%predecessor_block_id, &
                                     context%symbols(index)%value, &
                                     error_msg)) return
        case (VALUE_CHARACTER)
            ! Both deferred-length and fixed-length character symbols carry
            ! their current data pointer as the SSA %value field.  Print
            ! and concat sites read %value directly, so we need a phi on
            ! the pointer in the merge block; otherwise downstream uses
            ! pull an SSA value that is only defined in one predecessor.
            ! For deferred-length characters, the (data_ptr, length)
            ! allocas remain shared across branches, so deferred_data /
            ! deferred_length stay valid in then_result.
            if (.not. emit_liric_phi(context%session, &
                                     then_result%symbols(index)%value, &
                                     then_result%predecessor_block_id, &
                                     else_result%symbols(index)%value, &
                                     else_result%predecessor_block_id, &
                                     context%symbols(index)%value, &
                                     error_msg)) return
            context%symbols(index)%is_deferred_character = &
                then_result%symbols(index)%is_deferred_character
            context%symbols(index)%deferred_data = &
                then_result%symbols(index)%deferred_data
            context%symbols(index)%deferred_length = &
                then_result%symbols(index)%deferred_length
            context%symbols(index)%character_length = &
                then_result%symbols(index)%character_length
            context%symbols(index)%has_character_value = &
                then_result%symbols(index)%has_character_value
        case (VALUE_DERIVED)
            ! Derived-type symbols hold their storage in element_address (a
            ! stable alloca emitted at declaration time).  Both branches store
            ! into that same address, so the symbol value field is not used
            ! and no phi is needed; just adopt the then-branch snapshot.
            context%symbols(index) = then_result%symbols(index)
        case default
            error_msg = 'direct LIRIC session IF cannot merge unknown symbol kind'
            return
        end select

        call set_empty(error_msg)
    end subroutine merge_symbol_value

    logical function same_operand(lhs, rhs)
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs

        same_operand = lhs%kind == rhs%kind .and. lhs%payload == rhs%payload &
                       .and. lhs%global_offset == rhs%global_offset
    end function same_operand
