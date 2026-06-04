    subroutine lower_adjust_deferred(arena, node, context, left, out_data, &
                                     out_length, error_msg)
        ! adjustl(s): move leading blanks to the end. adjustr(s): move trailing
        ! blanks to the front. Result length equals len(s).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: left
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, k, content_len, dst_off
        type(lr_operand_desc_t) :: len64, buf, src, dst, content64, null_pos

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'adjustl/adjustr requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_i32_to_i64(context%session, s_len, len64, &
                error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                  emit_or_add1(context, len64), buf, error_msg)) then
            return
        end if
        ! Pre-fill the whole result with spaces, then drop the content in place.
        call fill_spaces(context, buf, s_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (left) then
            call compute_leading_blanks(context, s_data, s_len, k, error_msg)
            if (len_trim(error_msg) > 0) return
            ! content = s[k:], length s_len-k, placed at offset 0
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, s_len, k, &
                    content_len, error_msg)) return
            call ptr_plus_i32(context, s_data, k, src, error_msg)
            if (len_trim(error_msg) > 0) return
            dst = buf
        else
            ! trailing blanks: content = s[0:len_trim], placed at offset s_len-LT
            call compute_len_trim(context, s_data, s_len, content_len, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, s_len, &
                    content_len, dst_off, error_msg)) return
            src = s_data
            call ptr_plus_i32(context, buf, dst_off, dst, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (.not. emit_liric_i32_to_i64(context%session, content_len, content64, &
                error_msg)) return
        if (.not. emit_memcpy(context%session, dst, src, content64, error_msg)) &
            return
        ! null-terminate at buf[s_len]
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, len64, &
                null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
        out_data = buf
        out_length = s_len
        call set_empty(error_msg)
    end subroutine lower_adjust_deferred

    function emit_or_add1(context, len64) result(size_op)
        ! len64 + 1 (buffer holds the result plus a null terminator).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: len64
        type(lr_operand_desc_t) :: size_op
        character(len=:), allocatable :: err
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, len64, &
                i64_immediate(context%session, 1_c_int64_t), size_op, err)) then
            size_op = len64
        end if
    end function emit_or_add1

    subroutine ptr_plus_i32(context, base, offset_i32, result, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base
        type(lr_operand_desc_t), intent(in) :: offset_i32
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: off64

        if (.not. emit_liric_i32_to_i64(context%session, offset_i32, off64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, base, off64, &
                result, error_msg)) return
        call set_empty(error_msg)
    end subroutine ptr_plus_i32

    subroutine char_equals(context, a_data, a_len, b_data, b_len, result, &
                           error_msg)
        ! Fortran character equality with blank padding: compare the common
        ! prefix, then require the longer value's tail to be all spaces.
        ! result (i32) is 1 when equal, else 0. Both operands are read only
        ! within their own lengths (no out-of-bounds load).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: a_data, a_len, b_data, b_len
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: flag_addr, idx_addr, i_val, cond, ab, bb, tmp
        type(lr_operand_desc_t) :: a_le_b, minlen, maxlen, tail_data, byte
        integer(c_int32_t) :: hdr, body, advance, set_zero, prefix_done
        integer(c_int32_t) :: thdr, tbody, tadvance, done

        if (.not. emit_i32_alloca(context%session, flag_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 1_c_int64_t), flag_addr, &
                error_msg)) return
        ! minlen / maxlen and the longer operand's data pointer.
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, a_len, b_len, &
                a_le_b, error_msg)) return
        call select_value(context, a_le_b, a_len, b_len, minlen, error_msg)
        if (len_trim(error_msg) > 0) return
        call select_value(context, a_le_b, b_len, a_len, maxlen, error_msg)
        if (len_trim(error_msg) > 0) return
        call select_value(context, a_le_b, b_data, a_data, tail_data, error_msg)
        if (len_trim(error_msg) > 0) return

        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        advance = create_liric_block(context%session)
        set_zero = create_liric_block(context%session)
        prefix_done = create_liric_block(context%session)
        thdr = create_liric_block(context%session)
        tbody = create_liric_block(context%session)
        tadvance = create_liric_block(context%session)
        done = create_liric_block(context%session)

        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), idx_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        ! prefix: compare [0, minlen)
        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, minlen, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, prefix_done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_liric_char_byte_zext(context%session, a_data, i_val, ab, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, b_data, i_val, bb, &
                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, ab, bb, cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, set_zero, advance, &
                error_msg)) return

        if (.not. set_liric_block(context%session, advance, error_msg)) return
        context%current_block_id = advance
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        ! tail: from minlen to maxlen, the longer value must be all spaces.
        if (.not. set_liric_block(context%session, prefix_done, error_msg)) return
        context%current_block_id = prefix_done
        if (.not. emit_i32_store(context%session, minlen, idx_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, thdr, error_msg)) return

        if (.not. set_liric_block(context%session, thdr, error_msg)) return
        context%current_block_id = thdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, maxlen, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, tbody, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, tbody, error_msg)) return
        context%current_block_id = tbody
        if (.not. emit_liric_char_byte_zext(context%session, tail_data, i_val, &
                byte, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, byte, &
                i32_immediate(context%session, 32_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, set_zero, tadvance, &
                error_msg)) return

        if (.not. set_liric_block(context%session, tadvance, error_msg)) return
        context%current_block_id = tadvance
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, thdr, error_msg)) return

        if (.not. set_liric_block(context%session, set_zero, error_msg)) return
        context%current_block_id = set_zero
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), flag_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, done, error_msg)) return

        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, flag_addr, result, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine char_equals

    subroutine lower_index_intrinsic(arena, node, context, value, error_msg)
        ! index(s, sub): 1-based position of the first occurrence of sub in s,
        ! or 0. Implemented as an outer position scan with an inner byte compare,
        ! both driven by stack counters (no SSA phi).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, sub_data, sub_len
        type(lr_operand_desc_t) :: i_addr, j_addr, res_addr, max_start
        type(lr_operand_desc_t) :: i_val, j_val, cond, sb, tb, sum_ij, tmp
        integer(c_int32_t) :: outer_hdr, outer_body, inner_hdr, inner_body
        integer(c_int32_t) :: matched, advance, done

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'index requires two character arguments'
            return
        end if
        if (size(node%arg_indices) < 2) then
            error_msg = 'index requires two character arguments'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_expr_operands(arena, node%arg_indices(2), context, sub_data, &
                                sub_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_alloca(context%session, i_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, j_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, res_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), i_addr, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), res_addr, &
                error_msg)) return
        ! max_start = s_len - sub_len (last 0-based start to try)
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, s_len, sub_len, &
                max_start, error_msg)) return

        outer_hdr = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        inner_hdr = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        matched = create_liric_block(context%session)
        advance = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! outer_hdr: if i <= max_start goto outer_body else done
        if (.not. set_liric_block(context%session, outer_hdr, error_msg)) return
        context%current_block_id = outer_hdr
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, i_val, &
                max_start, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, outer_body, done, &
                error_msg)) return

        ! outer_body: j = 0; goto inner_hdr
        if (.not. set_liric_block(context%session, outer_body, error_msg)) return
        context%current_block_id = outer_body
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), j_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return

        ! inner_hdr: if j < sub_len goto inner_body else matched
        if (.not. set_liric_block(context%session, inner_hdr, error_msg)) return
        context%current_block_id = inner_hdr
        if (.not. emit_i32_load(context%session, j_addr, j_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, j_val, &
                sub_len, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, inner_body, matched, &
                error_msg)) return

        ! inner_body: if s[i+j] == sub[j] then j=j+1; inner_hdr else advance
        if (.not. set_liric_block(context%session, inner_body, error_msg)) return
        context%current_block_id = inner_body
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, j_val, &
                sum_ij, error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, s_data, sum_ij, sb, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, sub_data, j_val, &
                tb, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, sb, tb, cond, &
                error_msg)) return
        block
            integer(c_int32_t) :: bump
            bump = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, cond, bump, advance, &
                    error_msg)) return
            if (.not. set_liric_block(context%session, bump, error_msg)) return
            context%current_block_id = bump
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, j_val, &
                    i32_immediate(context%session, 1_c_int64_t), tmp, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, tmp, j_addr, error_msg)) &
                return
            if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return
        end block

        ! matched: result = i + 1; goto done
        if (.not. set_liric_block(context%session, matched, error_msg)) return
        context%current_block_id = matched
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, res_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, done, error_msg)) return

        ! advance: i = i + 1; goto outer_hdr
        if (.not. set_liric_block(context%session, advance, error_msg)) return
        context%current_block_id = advance
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, i_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! done: value = load result
        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, res_addr, value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine lower_index_intrinsic

    subroutine lower_trim_deferred(arena, node, context, out_data, out_length, &
                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, length, n, n64, size64
        type(lr_operand_desc_t) :: buf, null_pos, null_ptr
        character(len=64) :: null_name

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'trim requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                length, error_msg)
        if (len_trim(error_msg) > 0) return
        call compute_len_trim(context, data_ptr, length, n, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Allocate n + 1 bytes, copy the trimmed prefix, and null-terminate.
        if (.not. emit_liric_i32_to_i64(context%session, n, n64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, n64, &
                i64_immediate(context%session, 1_c_int64_t), size64, &
                error_msg)) return
        if (.not. emit_alloca_bytes(context%session, size64, buf, error_msg)) &
            return
        if (.not. emit_memcpy(context%session, buf, data_ptr, n64, error_msg)) &
            return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, n64, &
                null_pos, error_msg)) return
        context%string_literal_count = context%string_literal_count + 1
        write (null_name, '(A,I0)') '.ffc.trim.term.', context%string_literal_count
        call materialize_liric_string(context%session, trim(null_name), '', &
                                      null_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_memcpy(context%session, null_pos, null_ptr, &
                i64_immediate(context%session, 1_c_int64_t), error_msg)) return

        out_data = buf
        out_length = n
        call set_empty(error_msg)
    end subroutine lower_trim_deferred

    subroutine lower_i32_len_trim_intrinsic(arena, node, context, value, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'len_trim requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                length, error_msg)
        if (len_trim(error_msg) > 0) return
        call compute_len_trim(context, data_ptr, length, value, error_msg)
    end subroutine lower_i32_len_trim_intrinsic

    subroutine compute_len_trim(context, data_ptr, length, value, error_msg)
        ! Scan data[0:length) backward for the last non-space byte (1-based).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lr_operand_desc_t), intent(in) :: length
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx_addr, res_addr
        type(lr_operand_desc_t) :: i_value, prev_index, byte_value, cond
        type(lr_operand_desc_t) :: space
        integer(c_int32_t) :: header_block, body_block, found_block
        integer(c_int32_t) :: dec_block, done_block

        ! idx = length; result = 0; scan backwards for the last non-space byte.
        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, res_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, length, idx_addr, &
                                 error_msg)) return
        if (.not. emit_i32_store(context%session, &
                                 i32_immediate(context%session, 0_c_int64_t), &
                                 res_addr, error_msg)) return

        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        found_block = create_liric_block(context%session)
        dec_block = create_liric_block(context%session)
        done_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        ! header: i = load idx; if i > 0 goto body else done
        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        if (.not. emit_i32_load(context%session, idx_addr, i_value, &
                                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGT, i_value, &
                i32_immediate(context%session, 0_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, body_block, &
                                    done_block, error_msg)) return

        ! body: byte = data[i-1]; if byte == ' ' goto dec else found
        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, i_value, &
                i32_immediate(context%session, 1_c_int64_t), prev_index, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, data_ptr, &
                prev_index, byte_value, error_msg)) return
        space = i32_immediate(context%session, 32_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, byte_value, &
                space, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, dec_block, &
                                    found_block, error_msg)) return

        ! found: result = i; goto done
        if (.not. set_liric_block(context%session, found_block, error_msg)) return
        context%current_block_id = found_block
        if (.not. emit_i32_store(context%session, i_value, res_addr, &
                                 error_msg)) return
        if (.not. emit_liric_br(context%session, done_block, error_msg)) return

        ! dec: idx = i - 1; goto header
        if (.not. set_liric_block(context%session, dec_block, error_msg)) return
        context%current_block_id = dec_block
        if (.not. emit_i32_store(context%session, prev_index, idx_addr, &
                                 error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        ! done: value = load result
        if (.not. set_liric_block(context%session, done_block, error_msg)) return
        context%current_block_id = done_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, res_addr, value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine compute_len_trim
