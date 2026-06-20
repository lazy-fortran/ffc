    subroutine lower_if(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(if_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: empty_else(0)

        if (node%condition_index <= 0) then
            error_msg = 'direct LIRIC session IF requires a condition index'
            return
        end if
        if (.not. allocated(node%then_body_indices)) then
            error_msg = 'direct LIRIC session IF requires a THEN body'
            return
        end if

        ! An ELSE IF chain is lowered as a cascade: arm 0 is the main IF, and
        ! each elseif becomes the IF nested in the previous arm's else branch.
        ! This reuses the two-way symbol merge at every level. Arithmetic IF
        ! desugars its expr==0 arm into exactly such an elseif (#280).
        if (allocated(node%else_body_indices)) then
            call lower_if_arm(arena, node, 0, node%else_body_indices, context, &
                              value, error_msg)
        else
            call lower_if_arm(arena, node, 0, empty_else, context, value, &
                              error_msg)
        end if
    end subroutine lower_if

    recursive subroutine lower_if_arm(arena, node, elseif_index, else_indices, &
                                      context, value, error_msg)
        ! Lower one arm of an IF / ELSE IF cascade. elseif_index 0 is the main
        ! IF condition and THEN body; index k>0 selects node%elseif_blocks(k).
        ! else_indices is the final ELSE body, reached only after the last arm.
        type(ast_arena_t), intent(in) :: arena
        type(if_node), intent(in) :: node
        integer, intent(in) :: elseif_index
        integer, intent(in) :: else_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: condition
        type(symbol_t), allocatable :: saved_symbols(:)
        integer :: saved_symbol_count
        integer(c_int32_t) :: then_block, else_block, merge_block
        type(branch_result_t) :: then_result, else_result
        integer :: condition_index, n_elseif, i
        logical :: has_more_arms

        n_elseif = 0
        if (allocated(node%elseif_blocks)) n_elseif = size(node%elseif_blocks)
        if (elseif_index == 0) then
            condition_index = node%condition_index
        else
            condition_index = node%elseif_blocks(elseif_index)%condition_index
        end if
        has_more_arms = elseif_index < n_elseif

        call lower_i1_condition(arena, condition_index, context, condition, &
                                error_msg)
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

        if (elseif_index == 0) then
            call lower_if_branch(arena, node%then_body_indices, then_block, &
                                 merge_block, context, value, then_result, &
                                 error_msg)
        else
            call lower_if_branch(arena, &
                                 node%elseif_blocks(elseif_index)%body_indices, &
                                 then_block, merge_block, context, value, &
                                 then_result, error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        do i = 1, saved_symbol_count
            context%symbols(i) = saved_symbols(i)
        end do
        context%symbol_count = saved_symbol_count

        ! The else branch is either the next elseif arm or the final else body.
        if (.not. set_liric_block(context%session, else_block, error_msg)) return
        context%current_block_id = else_block
        if (has_more_arms) then
            call lower_if_arm(arena, node, elseif_index + 1, else_indices, &
                              context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            call snapshot_branch_result(context, else_result)
            if (.not. else_result%terminated) then
                if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                    return
            end if
        else
            call lower_if_branch(arena, else_indices, else_block, merge_block, &
                                 context, value, else_result, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

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
    end subroutine lower_if_arm

    subroutine snapshot_branch_result(context, result)
        ! Capture the current block's terminated state, symbol table, and
        ! predecessor block id into a branch_result_t. Used for an else arm that
        ! was lowered by a nested IF cascade rather than lower_if_branch.
        type(lowering_context_t), intent(in) :: context
        type(branch_result_t), intent(out) :: result
        integer :: i

        result%terminated = context%current_block_terminated
        if (allocated(result%symbols)) deallocate(result%symbols)
        allocate(result%symbols(context%symbol_count))
        do i = 1, context%symbol_count
            result%symbols(i) = context%symbols(i)
        end do
        result%symbol_count = context%symbol_count
        result%predecessor_block_id = context%current_block_id
    end subroutine snapshot_branch_result

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
        case (VALUE_F32, VALUE_F64)
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

    subroutine lower_pause(node, context, error_msg)
        ! gfortran writes a PAUSE banner to stderr (fd 2):
        !   PAUSE <message>
        !   To resume execution, type go.  Other input will terminate the job.
        ! then waits on stdin. End-of-input terminates the job with exit 0,
        ! suppressing any later output. Emit the banner via dprintf and read one
        ! byte: on EOF (getchar() == -1) call exit(0); otherwise resume (#280).
        type(pause_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: msg_text, banner
        type(lr_operand_desc_t) :: fa(3), fmtop, msgop, ch, is_eof
        type(lr_operand_desc_t) :: eof_imm
        integer(c_int32_t) :: fmt_gid, msg_gid
        integer(c_int32_t) :: exit_block, resume_block
        character(len=64) :: gname

        call set_empty(error_msg)
        banner = 'To resume execution, type go.  '// &
                 'Other input will terminate the job.'//achar(10)
        if (allocated(node%pause_message)) then
            call strip_literal_quotes(node%pause_message, msg_text)
        else
            msg_text = ''
        end if

        ! Banner string and format. The format prints "PAUSE <msg>" plus the
        ! resume line; the message rides as a %s argument (empty when absent, so
        ! the no-message form is "PAUSE " with its trailing space, as gfortran).
        context%string_literal_count = context%string_literal_count + 1
        write (gname, '(A,I0)') '.ffc.pause.msg.', context%string_literal_count
        call create_printf_format_global(context%session, trim(gname), &
                                         msg_text, msg_gid, error_msg)
        if (len_trim(error_msg) > 0) return
        msgop = printf_format_ptr(context%session, msg_gid)
        context%string_literal_count = context%string_literal_count + 1
        write (gname, '(A,I0)') '.ffc.pause.fmt.', context%string_literal_count
        call create_printf_format_global(context%session, trim(gname), &
                                         'PAUSE %s'//achar(10)//banner, fmt_gid, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        fmtop = printf_format_ptr(context%session, fmt_gid)
        fa(1) = i32_immediate(context%session, 2_c_int64_t)
        fa(2) = fmtop
        fa(3) = msgop
        if (.not. emit_dprintf(context%session, fa, error_msg)) return

        ! getchar(); branch on EOF.
        if (.not. emit_getchar(context%session, ch, error_msg)) return
        eof_imm = i32_immediate(context%session, -1_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, ch, eof_imm, &
                                      is_eof, error_msg)) return
        exit_block = create_liric_block(context%session)
        resume_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, is_eof, exit_block, &
                                    resume_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        if (.not. emit_exit(context%session, &
                            i32_immediate(context%session, 0_c_int64_t), &
                            error_msg)) return
        ! exit() does not return; close the block with an unreachable branch back
        ! to the resume path so the function stays well formed.
        if (.not. emit_liric_br(context%session, resume_block, error_msg)) return

        if (.not. set_liric_block(context%session, resume_block, error_msg)) return
        context%current_block_id = resume_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine lower_pause
