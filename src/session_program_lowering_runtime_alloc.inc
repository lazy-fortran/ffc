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

    subroutine alloc_component_extent_i32(context, desc_addr, extent_i32, error_msg)
        ! i32 element extent of a rank-1 allocatable array component read from the
        ! inline descriptor byte 8.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: desc_addr
        type(lr_operand_desc_t), intent(out) :: extent_i32
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ext64

        if (.not. emit_i64_load_at(context%session, desc_addr, 8_c_int64_t, &
                ext64, error_msg)) return
        if (.not. emit_liric_i64_to_i32(context%session, ext64, extent_i32, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine alloc_component_extent_i32

    subroutine load_alloc_component_element(context, desc_addr, vk, index_i32, &
                                            value, error_msg)
        ! Load one element of a rank-1 allocatable array component at a runtime
        ! 0-based i32 index by offsetting the descriptor's data pointer.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: desc_addr
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: index_i32
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, idx64, off64, addr

        if (.not. emit_ptr_load(context%session, desc_addr, data_ptr, error_msg)) &
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
    end subroutine load_alloc_component_element

    subroutine lower_alloc_component_i32_sum(context, desc_addr, value, error_msg)
        ! sum(obj%v) over a rank-1 integer allocatable array component: a counted
        ! LIRIC loop reading the descriptor extent and each element, mirroring
        ! lower_runtime_alloc_i32_sum's backedge-PHI shape.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: desc_addr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, entry_acc, entry_idx
        type(lr_operand_desc_t) :: header_acc, header_idx, backedge_acc, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_acc, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block, acc_vreg, idx_vreg

        value = i32_immediate(context%session, 0_c_int64_t)
        call alloc_component_extent_i32(context, desc_addr, extent, error_msg)
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
        call load_alloc_component_element(context, desc_addr, VALUE_I32, &
                                          header_idx, element, error_msg)
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
    end subroutine lower_alloc_component_i32_sum

    subroutine lower_alloc_component_reduction(arena, access_index, context, &
                                               reduction_name, value, error_msg)
        ! Reduce a whole rank-1 allocatable array component. Integer sum only for
        ! now; other kinds and reductions decline so the caller reports them.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: reduction_name
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: desc_addr
        integer :: comp_kind

        comp_kind = derived_component_access_kind_at(arena, access_index, context)
        if (trim(reduction_name) /= 'sum' .or. comp_kind /= VALUE_I32) then
            call unsupported_feature_error('allocatable array component '// &
                'reduction', get_node_line(arena, access_index), &
                get_node_column(arena, access_index), trim(reduction_name)// &
                ' over an allocatable array component supports integer sum only', &
                error_msg)
            return
        end if
        call alloc_array_component_descriptor(arena, access_index, context, &
                                              desc_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_alloc_component_i32_sum(context, desc_addr, value, error_msg)
    end subroutine lower_alloc_component_reduction

    integer function derived_component_access_kind_at(arena, access_index, &
                                                      context) result(comp_kind)
        ! Component value kind for the component_access node at access_index.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(in) :: context

        comp_kind = VALUE_I32
        if (.not. node_exists(arena, access_index)) return
        select type (acc => arena%entries(access_index)%node)
        type is (component_access_node)
            comp_kind = derived_component_access_kind(arena, acc, context)
        end select
    end function derived_component_access_kind_at

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

    ! ----- Rank-1 local automatic arrays with a runtime element count -----
    ! integer :: a(n) / real :: a(0:n) inside a procedure, where n is a
    ! dummy-argument or host value only known at run time. Storage is a dynamic
    ! alloca; the compile-time lower bound stays on the symbol and the runtime
    ! element count lives in runtime_dim_size(1). Element access reuses the
    ! fixed-array GEP path (array_size 0 sentinel, harmless for rank-1); whole
    ! array print/assign and size()/sum() walk a genuine LIRIC loop.

    logical function runtime_local_array_kind_ok(value_kind) result(ok)
        integer, intent(in) :: value_kind
        ! i32/f32/f64/logical share the element load/store-at-operand helpers;
        ! i64/i8/i16 lack a runtime-index store helper here and stay declined.
        ok = value_kind == VALUE_I32 .or. value_kind == VALUE_F32 .or. &
             value_kind == VALUE_F64 .or. value_kind == VALUE_LOGICAL
    end function runtime_local_array_kind_ok

    subroutine runtime_array_bounds(context, node, lower_bound, count_i32, &
                                    error_msg)
        ! Compile-time lower bound (default 1, or a constant a(L:...)) and the
        ! runtime i32 element count of a rank-1 declaration whose upper bound is
        ! a runtime expression. count = upper - lower + 1.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: lower_bound
        type(lr_operand_desc_t), intent(out) :: count_i32
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_index
        integer(c_int64_t) :: lower_value
        type(lr_operand_desc_t) :: upper_op

        lower_bound = 1
        dim_index = node%dimension_indices(1)
        select type (dim_node => context%arena%entries(dim_index)%node)
        type is (range_expression_node)
            call eval_i32_constant(context%arena, dim_node%start_index, context, &
                                   lower_value, error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'runtime-sized array requires a compile-time '// &
                            'lower bound'
                return
            end if
            lower_bound = int(lower_value)
            call lower_i32_expression(context%arena, dim_node%end_index, context, &
                                      upper_op, error_msg)
            if (len_trim(error_msg) > 0) return
            if (lower_bound == 1) then
                count_i32 = upper_op
            else
                if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_op, &
                        i32_immediate(context%session, &
                        int(lower_bound - 1, c_int64_t)), count_i32, error_msg)) &
                    return
            end if
        class default
            call lower_i32_expression(context%arena, dim_index, context, &
                                      count_i32, error_msg)
            if (len_trim(error_msg) > 0) return
        end select
        call set_empty(error_msg)
    end subroutine runtime_array_bounds

    subroutine define_runtime_array_symbol(context, node, name, value_kind, &
                                           error_msg)
        ! Allocate dynamic storage for a rank-1 local automatic array and record
        ! its runtime element count on a fresh symbol.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index, lower_bound
        type(lr_operand_desc_t) :: count_i32, count_i64, bytes, storage

        call runtime_array_bounds(context, node, lower_bound, count_i32, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_i32_to_i64(context%session, count_i32, count_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, count_i64, &
                i64_immediate(context%session, allocatable_elem_size(value_kind)), &
                bytes, error_msg)) return
        if (.not. emit_alloca_bytes(context%session, bytes, storage, error_msg)) then
            error_msg = 'runtime_array_alloca['//trim(name)//']: '//error_msg
            return
        end if

        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_array = .true.
        context%symbols(index)%is_runtime_array = .true.
        context%symbols(index)%array_rank = 1
        context%symbols(index)%array_size = 0
        context%symbols(index)%array_lower_bound = lower_bound
        context%symbols(index)%array_dim_lowers(1) = lower_bound
        context%symbols(index)%array_dim_sizes(1) = 0
        context%symbols(index)%has_runtime_dim_size(1) = .true.
        context%symbols(index)%runtime_dim_size(1) = count_i32
        context%symbols(index)%element_address = storage
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_runtime_array_symbol

    logical function symbol_is_runtime_local_array(context, sym) result(is_rt)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: sym
        is_rt = .false.
        if (sym <= 0) return
        is_rt = context%symbols(sym)%is_runtime_array
    end function symbol_is_runtime_local_array

    subroutine emit_runtime_local_array_print(context, sym, vk, error_msg)
        ! List-directed whole-array print of a rank-1 local runtime array. A
        ! counted loop reads runtime_dim_size(1) and prints each element with the
        ! same leading-space spacing as the constant-extent unrolled path.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym, vk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, entry_idx, header_idx, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block, idx_vreg

        extent = context%symbols(sym)%runtime_dim_size(1)
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
        call load_array_element_at_operand(context, sym, header_idx, element, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_typed_array_print_value(context, vk, element, error_msg)
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
    end subroutine emit_runtime_local_array_print

    subroutine lower_runtime_local_whole_assignment(context, dst_sym, src_op, &
                                                    src_sym, error_msg)
        ! Whole-array assignment into a rank-1 local runtime array. When src_sym
        ! is 0 the loop broadcasts the already-lowered scalar src_op into every
        ! element; otherwise it copies element-for-element from another rank-1
        ! array src_sym (fixed or runtime) of the same value kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dst_sym
        type(lr_operand_desc_t), intent(in) :: src_op
        integer, intent(in) :: src_sym
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, entry_idx, header_idx, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block, idx_vreg

        extent = context%symbols(dst_sym)%runtime_dim_size(1)
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
        if (src_sym > 0) then
            call load_array_element_at_operand(context, src_sym, header_idx, &
                                               element, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            element = src_op
        end if
        call store_array_element_at_operand(context, dst_sym, header_idx, element, &
                                            error_msg)
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
    end subroutine lower_runtime_local_whole_assignment

    subroutine lower_runtime_local_array_assignment(arena, node, dst_sym, &
                                                    context, handled, error_msg)
        ! Whole-array assignment dispatch for a rank-1 local runtime array. Two
        ! forms are handled at run time: a scalar broadcast (b = expr) and a copy
        ! from another rank-1 array of the same value kind (b = c). Anything else
        ! leaves handled false so the caller reports the existing diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: dst_sym
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: vk, src_sym
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: src_name, id_err

        handled = .false.
        call set_empty(error_msg)
        vk = context%symbols(dst_sym)%value_kind
        if (.not. runtime_local_array_kind_ok(vk)) return

        ! b = c : copy from another same-kind rank-1 array identifier.
        if (is_identifier(arena, node%value_index)) then
            call get_identifier_name(arena, node%value_index, src_name, id_err)
            if (len_trim(id_err) == 0) then
                src_sym = find_symbol(context, src_name)
                if (src_sym > 0) then
                    if (context%symbols(src_sym)%is_array .and. &
                        .not. context%symbols(src_sym)%is_allocatable .and. &
                        context%symbols(src_sym)%array_rank == 1 .and. &
                        context%symbols(src_sym)%value_kind == vk) then
                        value = i32_immediate(context%session, 0_c_int64_t)
                        call lower_runtime_local_whole_assignment(context, dst_sym, &
                            value, src_sym, error_msg)
                        handled = len_trim(error_msg) == 0
                        return
                    end if
                end if
            end if
        end if

        ! b = scalar : broadcast a scalar expression across every element.
        if (.not. is_scalar_broadcast_rhs(arena, node%value_index, context)) return
        select case (vk)
        case (VALUE_F32)
            call lower_f32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node%value_index, context, value, &
                                      error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node%value_index, context, value, &
                                          error_msg)
        case (VALUE_I64)
            call lower_i64_expression(arena, node%value_index, context, value, &
                                      error_msg)
        case default
            call lower_i32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        end select
        if (len_trim(error_msg) > 0) return
        call lower_runtime_local_whole_assignment(context, dst_sym, value, 0, &
                                                  error_msg)
        handled = len_trim(error_msg) == 0
    end subroutine lower_runtime_local_array_assignment
