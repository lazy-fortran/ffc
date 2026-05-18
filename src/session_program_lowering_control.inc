    subroutine lower_if(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(if_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: condition
        type(symbol_t) :: saved_symbols(MAX_SYMBOLS)
        integer :: saved_symbol_count
        integer(c_int32_t) :: then_block
        integer(c_int32_t) :: else_block
        integer(c_int32_t) :: merge_block
        type(branch_result_t) :: then_result
        type(branch_result_t) :: else_result

        if (node%condition_index <= 0) then
            error_msg = 'direct LIRIC session IF requires a condition index'
            return
        end if
        if (.not. allocated(node%then_body_indices) .or. &
            .not. allocated(node%else_body_indices)) then
            error_msg = 'direct LIRIC session IF requires THEN and ELSE bodies'
            return
        end if
        ! FortFront occasionally returns size-0 body_indices for if branches
        ! whose statements its standardiser has dropped (observed for derived
        ! component assignments inside if branches, fortfront upstream).
        ! Detect this and fail loud instead of silently lowering as a no-op.
        if (size(node%then_body_indices) == 0 .and. &
            size(node%else_body_indices) == 0) then
            call unsupported_feature_error('if statement', node%line, &
                                           node%column, &
                                           'both branches empty after frontend '// &
                                           'standardisation (likely fortfront '// &
                                           'dropped derived-component or other '// &
                                           'unsupported body statements)', &
                                           error_msg)
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

        saved_symbols = context%symbols
        saved_symbol_count = context%symbol_count

        call lower_if_branch(arena, node%then_body_indices, then_block, &
                             merge_block, context, value, then_result, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols = saved_symbols
        context%symbol_count = saved_symbol_count
        call lower_if_branch(arena, node%else_body_indices, else_block, &
                             merge_block, context, value, else_result, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols = saved_symbols
        context%symbol_count = saved_symbol_count

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
            context%symbols = then_result%symbols
            context%symbol_count = then_result%symbol_count
            call set_empty(error_msg)
        else
            context%symbols = else_result%symbols
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

        if (.not. set_liric_block(context%session, block_id, error_msg)) return
        context%current_block_id = block_id

        call lower_statement_list(arena, node_indices, context, value, &
                                  result%terminated, error_msg)
        if (len_trim(error_msg) > 0) return

        result%symbols = context%symbols
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
            error_msg = 'direct LIRIC session IF cannot merge character values'
            return
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
