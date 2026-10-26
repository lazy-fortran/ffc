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
        ! Lower a PAUSE statement to match gfortran behaviour:
        !   fprintf(stderr, "PAUSE <msg>\n")
        !   fprintf(stderr, "To resume execution, type go. ...")
        ! When stdin is at EOF (normal test environment), getchar() returns -1
        ! and the program terminates via exit(0), flushing stdout buffers.
        type(pause_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: msg_gid, hint_gid
        type(lr_operand_desc_t) :: msg_ptr, hint_ptr
        type(lr_operand_desc_t) :: fd2, zero
        type(lr_operand_desc_t) :: dargs(2)
        character(len=:), allocatable :: pause_line
        character(len=64) :: mgn, hgn

        ! Build "PAUSE <msg>\n" or "PAUSE \n" for bare PAUSE.
        ! pause_message is stored with surrounding quotes from the lexer;
        ! strip them before embedding in the PAUSE output line.
        if (allocated(node%pause_message) .and. &
            len_trim(node%pause_message) > 0) then
            block
                character(len=:), allocatable :: raw, inner
                raw = trim(node%pause_message)
                if (len(raw) >= 2 .and. &
                    ((raw(1:1) == '"' .and. raw(len(raw):len(raw)) == '"') .or. &
                     (raw(1:1) == "'" .and. raw(len(raw):len(raw)) == "'"))) then
                    inner = raw(2:len(raw)-1)
                else
                    inner = raw
                end if
                pause_line = 'PAUSE ' // inner // achar(10)
            end block
        else
            pause_line = 'PAUSE ' // achar(10)
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (mgn, '(A,I0)') '.ffc.pause.msg.', context%string_literal_count
        call create_printf_format_global(context%session, trim(mgn), &
                                         pause_line, msg_gid, error_msg)
        if (len_trim(error_msg) > 0) return
        msg_ptr = printf_format_ptr(context%session, msg_gid)

        context%string_literal_count = context%string_literal_count + 1
        write (hgn, '(A,I0)') '.ffc.pause.hint.', context%string_literal_count
        call create_printf_format_global(context%session, trim(hgn), &
            'To resume execution, type go.  Other input will terminate' // &
            ' the job.' // achar(10), hint_gid, error_msg)
        if (len_trim(error_msg) > 0) return
        hint_ptr = printf_format_ptr(context%session, hint_gid)

        ! dprintf(2, "PAUSE <msg>\n") — writes PAUSE line to stderr fd.
        fd2 = i32_immediate(context%session, 2_c_int64_t)
        dargs(1) = fd2
        dargs(2) = msg_ptr
        if (.not. emit_dprintf(context%session, dargs, error_msg)) return

        ! dprintf(2, "To resume...\n") — writes resume hint to stderr fd.
        dargs(2) = hint_ptr
        if (.not. emit_dprintf(context%session, dargs, error_msg)) return

        ! exit(0): terminates the program, flushing stdout buffers.
        ! This matches gfortran behaviour when stdin yields EOF at a PAUSE.
        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_void_call(context%session, 'exit', [zero], error_msg)) &
            return

        call set_empty(error_msg)
    end subroutine lower_pause
