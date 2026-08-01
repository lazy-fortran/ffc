    subroutine alloc_runtime_extent_i32(context, sym, extent_i32, error_msg)
        ! Runtime extent of a rank-1 allocatable read straight out of its
        ! canonical descriptor, truncated to i32. Used when the preceding
        ! allocate sized the array from a non-constant expression.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        type(lr_operand_desc_t), intent(out) :: extent_i32
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, ext64

        descriptor = context%symbols(sym)%allocatable_descriptor_address
        call emit_alloc_desc_load_extent(context, descriptor, 1, ext64, error_msg)
        if (len_trim(error_msg) > 0) return
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
        ! each element, mirroring lower_runtime_sum's backedge-PHI shape.
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

    subroutine runtime_array_shape(context, node, rank, lowers, extents_i32, &
                                   error_msg)
        ! Per-dimension shape of a rank-1 or rank-2 automatic array. Each lower
        ! bound must fold at compile time; each upper bound is evaluated once,
        ! here, so the extent is fixed on entry to the scope (F2018 8.5.8.2) and
        ! never re-read if the bound expression's variables change later.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: rank
        integer, intent(out) :: lowers(:)
        type(lr_operand_desc_t), intent(out) :: extents_i32(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_index, d
        integer(c_int64_t) :: lower_value
        type(lr_operand_desc_t) :: upper_op

        lowers = 1
        rank = 0
        if (.not. allocated(node%dimension_indices)) then
            error_msg = 'runtime-sized array requires explicit dimensions'
            return
        end if
        rank = size(node%dimension_indices)
        if (rank < 1 .or. rank > 2) then
            error_msg = 'runtime-sized array supports rank 1 and rank 2 only'
            return
        end if

        do d = 1, rank
            dim_index = node%dimension_indices(d)
            select type (dim_node => context%arena%entries(dim_index)%node)
            type is (range_expression_node)
                call eval_i32_constant(context%arena, dim_node%start_index, &
                                       context, lower_value, error_msg)
                if (len_trim(error_msg) > 0) then
                    error_msg = 'runtime-sized array requires a compile-time '// &
                                'lower bound'
                    return
                end if
                lowers(d) = int(lower_value)
                call lower_i32_expression(context%arena, dim_node%end_index, &
                                          context, upper_op, error_msg)
                if (len_trim(error_msg) > 0) return
                if (lowers(d) == 1) then
                    extents_i32(d) = upper_op
                else
                    ! extent = upper - lower + 1
                    if (.not. emit_i32_binary(context%session, LR_OP_SUB, &
                            upper_op, i32_immediate(context%session, &
                            int(lowers(d) - 1, c_int64_t)), extents_i32(d), &
                            error_msg)) return
                end if
            class default
                call lower_i32_expression(context%arena, dim_index, context, &
                                          extents_i32(d), error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
        call set_empty(error_msg)
    end subroutine runtime_array_shape

    subroutine define_runtime_array_symbol(context, node, name, value_kind, &
                                           error_msg)
        ! Allocate dynamic storage for a rank-1 or rank-2 local automatic array
        ! and describe it with one canonical array descriptor (#335). The
        ! descriptor is the stored shape of record: lower bound, extent, and
        ! contiguous column-major byte stride per dimension. The symbol's cached
        ! per-dimension extents are loaded back out of that descriptor, so no
        ! second shape representation exists.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index, rank, d
        integer :: lowers(ARRAY_MAX_RANK)
        type(lr_operand_desc_t) :: extents_i32(2)
        type(lr_operand_desc_t) :: count_i32, count_i64, bytes, storage
        type(lr_operand_desc_t) :: product_i32
        type(lr_operand_desc_t) :: descriptor, extent_i64, extent_i32

        call runtime_array_shape(context, node, rank, lowers, extents_i32, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return

        ! Total element count is the product of the per-dimension extents.
        count_i32 = extents_i32(1)
        do d = 2, rank
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, count_i32, &
                    extents_i32(d), product_i32, error_msg)) return
            count_i32 = product_i32
        end do

        if (.not. emit_liric_i32_to_i64(context%session, count_i32, count_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, count_i64, &
                i64_immediate(context%session, allocatable_elem_size(value_kind)), &
                bytes, error_msg)) return
        if (.not. emit_alloca_bytes(context%session, bytes, storage, error_msg)) then
            error_msg = 'runtime_array_alloca['//trim(name)//']: '//error_msg
            return
        end if

        call emit_runtime_array_descriptor(context, storage, rank, lowers, &
                                           extents_i32, value_kind, descriptor, &
                                           error_msg)
        if (len_trim(error_msg) > 0) return

        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_array = .true.
        context%symbols(index)%is_runtime_array = .true.
        context%symbols(index)%has_runtime_descriptor = .true.
        context%symbols(index)%runtime_descriptor_address = descriptor
        context%symbols(index)%array_rank = rank
        context%symbols(index)%array_size = 0
        context%symbols(index)%array_lower_bound = lowers(1)
        context%symbols(index)%element_address = storage
        do d = 1, rank
            context%symbols(index)%array_dim_lowers(d) = lowers(d)
            context%symbols(index)%array_dim_sizes(d) = 0
            ! Read the extent back out of the descriptor so the descriptor, not
            ! the bound expression, is what every later shape query observes.
            if (.not. emit_i64_load_at(context%session, descriptor, &
                    runtime_descriptor_dim_offset(d, &
                        int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                    extent_i64, error_msg)) return
            if (.not. emit_liric_i64_to_i32(context%session, extent_i64, &
                    extent_i32, error_msg)) return
            context%symbols(index)%has_runtime_dim_size(d) = .true.
            context%symbols(index)%runtime_dim_size(d) = extent_i32
        end do
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_runtime_array_symbol

    integer(c_int64_t) function runtime_descriptor_dim_offset(dim, field) &
            result(offset)
        integer, intent(in) :: dim
        integer(c_int64_t), intent(in) :: field

        offset = int(ARRAY_DESCRIPTOR_DIM_OFFSET, c_int64_t) &
                 + int(ARRAY_DIMENSION_BYTES, c_int64_t)*int(dim - 1, c_int64_t) &
                 + field
    end function runtime_descriptor_dim_offset

    subroutine emit_runtime_array_descriptor(context, base, rank, lowers, &
                                             extents_i32, value_kind, &
                                             descriptor, error_msg)
        ! Fill one canonical array descriptor for a runtime automatic array.
        ! The array owns its storage for the lifetime of its scope, so the
        ! descriptor is contiguous and allocated but never carries the ownership
        ! bit: the storage is a stack allocation reclaimed with the frame, not a
        ! heap block any deallocator may free.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base
        integer, intent(in) :: rank
        integer, intent(in) :: lowers(:)
        type(lr_operand_desc_t), intent(in) :: extents_i32(:)
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent_i64, running, next_running, stride
        integer(c_int64_t) :: element_bytes
        integer(c_int32_t) :: flags
        integer :: d

        element_bytes = allocatable_elem_size(value_kind)
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(ARRAY_DESCRIPTOR_BYTES, c_int64_t)), descriptor, &
                error_msg)) return
        if (.not. emit_ptr_store(context%session, base, descriptor, error_msg)) &
            return
        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, element_bytes), descriptor, &
                int(ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, c_int64_t), &
                error_msg)) return
        call store_descriptor_i32(context, assumed_shape_element_type(value_kind), &
            descriptor, int(ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, c_int64_t), &
            error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, int(rank, c_int32_t), descriptor, &
            int(ARRAY_DESCRIPTOR_RANK_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        flags = ARRAY_FLAG_ALLOCATED + ARRAY_FLAG_ASSOCIATED + ARRAY_FLAG_CONTIGUOUS
        call store_descriptor_i32(context, flags, descriptor, &
            int(ARRAY_DESCRIPTOR_FLAGS_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, 0_c_int32_t, descriptor, &
            int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return

        running = i64_immediate(context%session, element_bytes)
        do d = 1, rank
            if (.not. emit_liric_i32_to_i64(context%session, extents_i32(d), &
                    extent_i64, error_msg)) return
            if (.not. emit_i64_store_at(context%session, &
                    i64_immediate(context%session, int(lowers(d), c_int64_t)), &
                    descriptor, runtime_descriptor_dim_offset(d, &
                        int(ARRAY_DIMENSION_LOWER_OFFSET, c_int64_t)), &
                    error_msg)) return
            if (.not. emit_i64_store_at(context%session, extent_i64, descriptor, &
                    runtime_descriptor_dim_offset(d, &
                        int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                    error_msg)) return
            stride = running
            if (.not. emit_i64_store_at(context%session, stride, descriptor, &
                    runtime_descriptor_dim_offset(d, &
                        int(ARRAY_DIMENSION_STRIDE_OFFSET, c_int64_t)), &
                    error_msg)) return
            if (d < rank) then
                if (.not. emit_i64_binary(context%session, LR_OP_MUL, running, &
                        extent_i64, next_running, error_msg)) return
                running = next_running
            end if
        end do
        call set_empty(error_msg)
    end subroutine emit_runtime_array_descriptor

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
        type(lr_operand_desc_t) :: extent, total_extent, entry_idx, header_idx, &
                                    backedge_idx
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
        ! Whole-array assignment into a runtime-shaped array. When src_sym is 0
        ! the loop broadcasts the already-lowered scalar src_op into every
        ! element; otherwise it copies element-for-element from another rank-1
        ! or rank-2 array src_sym of the same value kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dst_sym
        type(lr_operand_desc_t), intent(in) :: src_op
        integer, intent(in) :: src_sym
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, total_extent, entry_idx, header_idx, &
                                    backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block, idx_vreg

        extent = context%symbols(dst_sym)%runtime_dim_size(1)
        if (context%symbols(dst_sym)%array_rank == 2) then
            if (.not. context%symbols(dst_sym)%has_runtime_dim_size(2)) then
                error_msg = 'runtime rank-2 array has no second extent'
                return
            end if
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, extent, &
                    context%symbols(dst_sym)%runtime_dim_size(2), total_extent, &
                    error_msg)) return
            extent = total_extent
        else if (context%symbols(dst_sym)%array_rank /= 1) then
            error_msg = 'runtime array assignment supports rank 1 and rank 2 only'
            return
        end if
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

    subroutine lower_runtime_local_constructor(arena, ctor, dst_sym, context, &
                                               error_msg)
        ! b = [e1, e2, ...] for a rank-1 local runtime array: lower every
        ! element and store it at its linear slot in the dynamic storage.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: dst_sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, slot
        integer, allocatable :: flat(:)
        integer :: i, vk

        call set_empty(error_msg)
        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'array constructor has no elements'
            return
        end if
        allocate (flat(0))
        call flatten_constructor_elements(arena, ctor%element_indices, flat, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return

        vk = context%symbols(dst_sym)%value_kind
        do i = 1, size(flat)
            select case (vk)
            case (VALUE_F32)
                call lower_f32_expression(arena, flat(i), context, value, &
                                          error_msg)
            case (VALUE_F64)
                call lower_f64_expression(arena, flat(i), context, value, &
                                          error_msg)
            case (VALUE_LOGICAL)
                call lower_logical_expression(arena, flat(i), context, value, &
                                              error_msg)
            case default
                call lower_i32_expression(arena, flat(i), context, value, &
                                          error_msg)
            end select
            if (len_trim(error_msg) > 0) return
            slot = i32_immediate(context%session, int(i - 1, c_int64_t))
            call store_array_element_at_operand(context, dst_sym, slot, value, &
                                                error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_runtime_local_constructor

    subroutine lower_runtime_local_array_assignment(arena, node, dst_sym, &
                                                    context, handled, error_msg)
        ! Whole-array assignment dispatch for a runtime-shaped array. Two forms
        ! are handled at run time: a scalar broadcast (b = expr) and a copy
        ! from another rank-1 or rank-2 array of the same value kind. Anything else
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

        ! b = [...] : store each constructor element into the entry-sized
        ! storage. The extent is only known at run time, so no compile-time
        ! element-count check applies.
        if (node_exists(arena, node%value_index)) then
            select type (rhs => arena%entries(node%value_index)%node)
            type is (array_literal_node)
                if (.not. is_implied_do_constructor(arena, rhs)) then
                    call lower_runtime_local_constructor(arena, rhs, dst_sym, &
                                                         context, error_msg)
                    handled = len_trim(error_msg) == 0
                    return
                end if
            class default
            end select
        end if

        ! b = c : copy from another same-kind array identifier.
        if (is_identifier(arena, node%value_index)) then
            call get_identifier_name(arena, node%value_index, src_name, id_err)
            if (len_trim(id_err) == 0) then
                src_sym = find_symbol_compat(context, src_name)
                if (src_sym > 0) then
                    if ((context%symbols(src_sym)%is_array .or. &
                        context%symbols(src_sym)%is_allocatable) .and. &
                        context%symbols(src_sym)%array_rank == &
                            context%symbols(dst_sym)%array_rank .and. &
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
