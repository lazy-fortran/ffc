    subroutine alloc_runtime_extent_i32(context, sym, extent_i32, error_msg)
        ! Runtime extent of a rank-1 allocatable read from its descriptor:
        ! (upper@16 - lower@8 + 1), truncated to i32. Used when the preceding
        ! allocate sized the array from a non-constant expression.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        type(lr_operand_desc_t), intent(out) :: extent_i32
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, lower1, upper1, span, ext64

        descriptor = context%symbols(sym)%allocatable_descriptor_address
        if (.not. emit_i64_load_at(context%session, descriptor, 8_c_int64_t, &
                lower1, error_msg)) return
        if (.not. emit_i64_load_at(context%session, descriptor, 16_c_int64_t, &
                upper1, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_SUB, upper1, lower1, &
                span, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, span, &
                i64_immediate(context%session, 1_c_int64_t), ext64, error_msg)) &
            return
        if (.not. emit_liric_i64_to_i32(context%session, ext64, extent_i32, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine alloc_runtime_extent_i32

    subroutine load_alloc_element_runtime(context, sym, vk, index_i32, value, &
                                          error_msg)
        ! Load one element of a rank-1 allocatable at a runtime 0-based i32 index
        ! by offsetting the descriptor's data pointer by index*stride.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym, vk
        type(lr_operand_desc_t), intent(in) :: index_i32
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, idx64, off64, addr

        descriptor = context%symbols(sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_liric_i32_to_i64(context%session, index_i32, idx64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, idx64, &
                i64_immediate(context%session, allocatable_elem_size(vk)), off64, &
                error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, off64, addr, &
                error_msg)) return
        if (.not. emit_allocatable_element_load(context, vk, addr, value, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine load_alloc_element_runtime

    subroutine lower_runtime_alloc_i32_sum(context, sym, value, error_msg)
        ! sum(a) over a rank-1 integer allocatable whose extent is only known at
        ! runtime. Walks a counted LIRIC loop reading the descriptor extent and
        ! each element, mirroring lower_runtime_i32_sum's backedge-PHI shape.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, entry_acc, entry_idx
        type(lr_operand_desc_t) :: header_acc, header_idx, backedge_acc, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_acc, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block, acc_vreg, idx_vreg

        value = i32_immediate(context%session, 0_c_int64_t)
        call alloc_runtime_extent_i32(context, sym, extent, error_msg)
        if (len_trim(error_msg) > 0) return
        entry_acc = i32_immediate(context%session, 0_c_int64_t)
        entry_idx = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.
        acc_vreg = reserve_i32_vreg(context%session)
        if (acc_vreg == 0_c_int32_t) acc_vreg = reserve_i32_vreg(context%session)
        idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg == 0_c_int32_t) idx_vreg = reserve_i32_vreg(context%session)
        if (acc_vreg <= 0_c_int32_t .or. idx_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a positive backedge vreg'
            return
        end if
        backedge_acc = i32_vreg(context%session, acc_vreg)
        backedge_idx = i32_vreg(context%session, idx_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_acc, entry_block, &
                backedge_acc, latch_block, header_acc, error_msg)) return
        if (.not. emit_liric_i32_phi(context%session, entry_idx, entry_block, &
                backedge_idx, latch_block, header_idx, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_idx, &
                extent, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        call load_alloc_element_runtime(context, sym, VALUE_I32, header_idx, &
                                        element, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, header_acc, &
                element, next_acc, error_msg)) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_idx, &
                i32_immediate(context%session, 1_c_int64_t), idx_vreg, next_idx, &
                error_msg)) return
        if (.not. emit_i32_copy_to(context%session, next_acc, acc_vreg, &
                backedge_acc, error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        value = header_acc
        call set_empty(error_msg)
    end subroutine lower_runtime_alloc_i32_sum

    subroutine emit_runtime_alloc_print(context, sym, vk, error_msg)
        ! List-directed whole-array print of a rank-1 allocatable whose extent is
        ! only known at runtime. A counted loop reads the descriptor extent and
        ! prints each element with the same leading-space spacing as the
        ! constant-extent unrolled path.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym, vk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, entry_idx, header_idx, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block, idx_vreg

        call alloc_runtime_extent_i32(context, sym, extent, error_msg)
        if (len_trim(error_msg) > 0) return
        entry_idx = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.
        idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg == 0_c_int32_t) idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a positive backedge vreg'
            return
        end if
        backedge_idx = i32_vreg(context%session, idx_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_idx, entry_block, &
                backedge_idx, latch_block, header_idx, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_idx, &
                extent, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_print_space(context%session, error_msg)) return
        call load_alloc_element_runtime(context, sym, vk, header_idx, element, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_runtime_alloc_print_one(context, vk, element, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_idx, &
                i32_immediate(context%session, 1_c_int64_t), idx_vreg, next_idx, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine emit_runtime_alloc_print

    subroutine emit_runtime_alloc_print_one(context, vk, value, error_msg)
        ! Print one already-loaded allocatable element in its own kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (vk == VALUE_F32) then
            if (.not. emit_liric_print_f32_value(context%session, value, &
                    error_msg)) return
        else if (vk == VALUE_F64) then
            if (.not. emit_liric_print_f64_value(context%session, value, &
                    error_msg)) return
        else if (vk == VALUE_LOGICAL) then
            call lower_print_logical_value(context, value, error_msg)
        else
            if (.not. emit_liric_print_i32_value(context%session, &
                    context%i32_print_format_id, value, error_msg)) return
        end if
    end subroutine emit_runtime_alloc_print_one
