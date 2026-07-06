    ! Runtime-extent ABI for a rank-1 assumed-shape dummy a(:) whose actual has
    ! no compile-time-foldable shape (an allocatable actual). A hidden i64
    ! extent argument is appended after the callee's visible pointer parameters,
    ! one per such dummy, in dummy declaration order. See docs/RUNTIME_ABI.md.
    !
    ! The existing whole-arena compile-time fold (resolve_assumed_shape_extents,
    ! session_program_lowering_arrays.inc) stays the fast path: only a dummy for
    ! which that fold fails gets a hidden parameter, so an already-working
    ! compile-time-resolved assumed-shape dummy keeps its original signature.
    ! Only subroutines are covered in this slice; a function with a
    ! non-foldable assumed-shape dummy keeps the pre-existing diagnostic.

    integer function dummy_assumed_shape_rank(arena, body_indices, param_name) &
            result(rank)
        ! Rank (1 or 2) of a rank-1/rank-2 assumed-shape dummy named param_name,
        ! or 0 when param_name is not such a dummy. Every dimension must be a
        ! bare colon (:) for the dummy to qualify as fully assumed-shape.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k, d, dim_count
        logical :: name_matches, all_colon

        rank = 0
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. decl%is_array) cycle
                if (decl%is_allocatable) cycle
                if (.not. allocated(decl%dimension_indices)) cycle
                dim_count = size(decl%dimension_indices)
                if (dim_count < 1 .or. dim_count > 2) cycle
                all_colon = .true.
                do d = 1, dim_count
                    if (.not. dim_is_assumed_shape(arena, &
                            decl%dimension_indices(d))) all_colon = .false.
                end do
                if (.not. all_colon) cycle
                name_matches = .false.
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == trim(param_name)) &
                        name_matches = .true.
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == trim(param_name)) &
                            name_matches = .true.
                    end do
                end if
                if (name_matches) then
                    rank = dim_count
                    return
                end if
            end select
        end do
    end function dummy_assumed_shape_rank

    logical function dummy_is_assumed_shape_rank1(arena, body_indices, param_name) &
            result(is_match)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name

        is_match = dummy_assumed_shape_rank(arena, body_indices, param_name) == 1
    end function dummy_is_assumed_shape_rank1

    integer function callee_dummy_assumed_shape_dims(arena, callee_name, &
                                                      param_pos) result(rank)
        ! Rank (1 or 2) of callee_name's param_pos-th dummy when it is a
        ! rank-1/rank-2 assumed-shape array, otherwise 0. Only a subroutine
        ! callee is inspected (the runtime-extent ABI slice).
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err
        character(len=:), allocatable :: node_type
        type(subroutine_def_node), pointer :: sb_node

        rank = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type /= 'subroutine_def_node' .and. &
                node_type /= 'subroutine_def') cycle
            sb_node => get_node_as_subroutine_def(arena, n)
            if (.not. associated(sb_node)) cycle
            if (.not. allocated(sb_node%name)) cycle
            if (.not. same_name(sb_node%name, callee_name)) cycle
            if (.not. allocated(sb_node%param_indices)) return
            if (param_pos < 1 .or. param_pos > size(sb_node%param_indices)) return
            call parameter_name(arena, sb_node%param_indices(param_pos), name, &
                                name_err)
            if (len_trim(name_err) > 0) return
            rank = dummy_assumed_shape_rank(arena, sb_node%body_indices, name)
            return
        end do
    end function callee_dummy_assumed_shape_dims

    logical function callee_dummy_assumed_shape_rank1(arena, callee_name, &
                                                       param_pos) result(is_match)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos

        is_match = callee_dummy_assumed_shape_dims(arena, callee_name, &
                                                   param_pos) == 1
    end function callee_dummy_assumed_shape_rank1

    logical function callee_dummy_assumed_shape_runtime(arena, callee_name, &
                                                         param_pos) result(is_match)
        ! Whether the callee's param_pos-th dummy is a rank-1 or rank-2
        ! assumed-shape array, so an allocatable actual is passed as a bare data
        ! pointer plus hidden runtime extents rather than a whole descriptor.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos

        is_match = callee_dummy_assumed_shape_dims(arena, callee_name, &
                                                   param_pos) >= 1
    end function callee_dummy_assumed_shape_runtime

    subroutine callee_param_name_at(arena, callee_name, param_pos, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n
        character(len=:), allocatable :: node_type
        type(subroutine_def_node), pointer :: sb_node

        call set_empty(name)
        error_msg = 'callee subroutine was not found: '//trim(callee_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type /= 'subroutine_def_node' .and. &
                node_type /= 'subroutine_def') cycle
            sb_node => get_node_as_subroutine_def(arena, n)
            if (.not. associated(sb_node)) cycle
            if (.not. allocated(sb_node%name)) cycle
            if (.not. same_name(sb_node%name, callee_name)) cycle
            if (.not. allocated(sb_node%param_indices)) return
            if (param_pos < 1 .or. param_pos > size(sb_node%param_indices)) return
            call parameter_name(arena, sb_node%param_indices(param_pos), name, &
                                error_msg)
            return
        end do
    end subroutine callee_param_name_at

    integer function call_arg_param_pos(arg_pos, self_position) result(param_pos)
        ! Map a 1-based call-site argument position to the callee's 1-based
        ! dummy position. self_position <= 0 means the call has no implicit
        ! passed-object receiver (the common case: identity mapping). A
        ! positive self_position is the dummy slot a type-bound subroutine
        ! call's receiver occupies (lower_method_subroutine_call): explicit
        ! arguments before that slot map 1:1, explicit arguments at or after
        ! it are shifted right by the one slot the receiver consumes.
        integer, intent(in) :: arg_pos
        integer, intent(in) :: self_position

        if (self_position <= 0 .or. arg_pos < self_position) then
            param_pos = arg_pos
        else
            param_pos = arg_pos + 1
        end if
    end function call_arg_param_pos

    integer function param_runtime_extent_rank(context, callee_name, param_pos) &
            result(rank)
        ! Number of hidden runtime extents callee_name's param_pos-th dummy
        ! needs: its assumed-shape rank (1 or 2) when the whole-arena
        ! compile-time fold fails, otherwise 0.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        character(len=:), allocatable :: name, name_err
        integer :: extents(2), dims
        logical :: ok

        rank = 0
        dims = callee_dummy_assumed_shape_dims(context%arena, callee_name, &
                                               param_pos)
        if (dims < 1) return
        call callee_param_name_at(context%arena, callee_name, param_pos, name, &
                                  name_err)
        if (len_trim(name_err) > 0) return
        call resolve_assumed_shape_extents(context, callee_name, name, dims, &
                                           extents, ok)
        if (.not. ok) rank = dims
    end function param_runtime_extent_rank

    logical function param_needs_runtime_extent(context, callee_name, param_pos) &
            result(needs_runtime)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos

        needs_runtime = param_runtime_extent_rank(context, callee_name, &
                                                  param_pos) > 0
    end function param_needs_runtime_extent

    integer function callee_hidden_extent_count(context, callee_name) result(count)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: callee_name
        integer :: n, i
        character(len=:), allocatable :: node_type
        type(subroutine_def_node), pointer :: sb_node

        count = 0
        do n = 1, context%arena%size
            if (.not. node_exists(context%arena, n)) cycle
            node_type = get_node_type_at(context%arena, n)
            if (node_type /= 'subroutine_def_node' .and. &
                node_type /= 'subroutine_def') cycle
            sb_node => get_node_as_subroutine_def(context%arena, n)
            if (.not. associated(sb_node)) cycle
            if (.not. allocated(sb_node%name)) cycle
            if (.not. same_name(sb_node%name, callee_name)) cycle
            if (.not. allocated(sb_node%param_indices)) return
            do i = 1, size(sb_node%param_indices)
                count = count + param_runtime_extent_rank(context, callee_name, i)
            end do
            return
        end do
    end function callee_hidden_extent_count

    subroutine bind_hidden_extent_params(arena, param_indices, context, &
                                         callee_name, visible_count, error_msg)
        ! Load each hidden i64 extent parameter (positions visible_count,
        ! visible_count+1, ... 0-based) and stash it on the matching dummy's
        ! symbol; the later assumed-shape declaration binding
        ! (bind_one_assumed_shape) reads it when the compile-time fold fails.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: visible_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, hidden_pos, sym_index, rank, d
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: hidden_ptr, extent_i64, extent_i32

        call set_empty(error_msg)
        if (.not. allocated(param_indices)) return
        hidden_pos = visible_count
        do i = 1, size(param_indices)
            rank = param_runtime_extent_rank(context, callee_name, i)
            if (rank < 1) cycle
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym_index = find_symbol(context, name)
            if (sym_index <= 0) then
                error_msg = 'assumed-shape dummy was not registered: '// &
                            trim(name)
                return
            end if
            do d = 1, rank
                hidden_ptr = ptr_param(context%session, hidden_pos)
                if (.not. emit_i64_load(context%session, hidden_ptr, extent_i64, &
                                        error_msg)) return
                if (.not. emit_liric_i64_to_i32(context%session, extent_i64, &
                                                extent_i32, error_msg)) return
                context%symbols(sym_index)%has_runtime_dim_size(d) = .true.
                context%symbols(sym_index)%runtime_dim_size(d) = extent_i32
                hidden_pos = hidden_pos + 1
            end do
        end do
    end subroutine bind_hidden_extent_params

    subroutine allocatable_array_actual_data_address(arena, node_index, context, &
                                                      address, error_msg)
        ! Assumed-shape dummy bound to an allocatable actual: the descriptor's
        ! runtime data pointer (offset 0), not a fixed alloca address, since the
        ! storage address is itself only known at runtime.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index
        type(lr_operand_desc_t) :: descriptor

        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, id_name)
        if (symbol_index <= 0) then
            error_msg = 'allocatable array argument was not declared: '// &
                        trim(id_name)
            return
        end if
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, address, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine allocatable_array_actual_data_address

    subroutine actual_extent_i64(arena, node_index, context, dim, extent, &
                                 error_msg)
        ! Runtime extent (i64) of dimension dim of the whole-array actual bound
        ! to a hidden extent argument. Only a rank-1 or rank-2 allocatable
        ! identifier is supported; its descriptor bounds give upper - lower + 1.
        ! Bounds live at descriptor offsets (8, 16) for dim 1 and (24, 32) for
        ! dim 2 (session_program_lowering_allocatable.inc).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dim
        type(lr_operand_desc_t), intent(out) :: extent
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index
        integer(c_int64_t) :: lower_off, upper_off
        type(lr_operand_desc_t) :: descriptor, lower, upper, span

        if (.not. is_identifier(arena, node_index)) then
            error_msg = 'assumed-shape actual extent must come from a '// &
                        'whole-array allocatable actual'
            return
        end if
        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, id_name)
        if (symbol_index <= 0) then
            error_msg = 'array argument was not declared: '//trim(id_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            error_msg = 'assumed-shape actual extent must come from a '// &
                        'whole-array allocatable actual: '//trim(id_name)
            return
        end if
        if (dim == 1) then
            lower_off = 8_c_int64_t
            upper_off = 16_c_int64_t
        else
            lower_off = 24_c_int64_t
            upper_off = 32_c_int64_t
        end if
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_i64_load_at(context%session, descriptor, lower_off, &
                                   lower, error_msg)) return
        if (.not. emit_i64_load_at(context%session, descriptor, upper_off, &
                                   upper, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_SUB, upper, lower, &
                                  span, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, span, &
                                  i64_immediate(context%session, 1_c_int64_t), &
                                  extent, error_msg)) return
        call set_empty(error_msg)
    end subroutine actual_extent_i64

    subroutine append_hidden_extent_args(arena, arg_indices, context, &
                                         callee_name, visible_count, args, &
                                         error_msg, self_position)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: visible_count
        type(lr_operand_desc_t), intent(inout) :: args(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: self_position
        integer :: i, hidden_pos, sp, rank, d
        type(lr_operand_desc_t) :: extent_i64, slot

        call set_empty(error_msg)
        sp = 0
        if (present(self_position)) sp = self_position
        hidden_pos = visible_count
        do i = 1, size(arg_indices)
            rank = param_runtime_extent_rank(context, callee_name, &
                                             call_arg_param_pos(i, sp))
            if (rank < 1) cycle
            do d = 1, rank
                hidden_pos = hidden_pos + 1
                call actual_extent_i64(arena, arg_indices(i), context, d, &
                                       extent_i64, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i64_alloca(context%session, slot, error_msg)) return
                if (.not. emit_i64_store(context%session, extent_i64, slot, &
                                         error_msg)) return
                args(hidden_pos) = slot
            end do
        end do
    end subroutine append_hidden_extent_args

    subroutine load_array_element_at_runtime_index_i32(context, symbol_index, &
                                                        index_i32, value, &
                                                        error_msg)
        ! Element load at a runtime (non-immediate) 0-based i32 index for a
        ! rank-1 integer array symbol. The GEP's declared array-size type
        ! parameter does not affect the computed address for a single-index
        ! rank-1 access, so it is harmless that a runtime-extent dummy's
        ! array_size stays the 0 sentinel.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: index_i32
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr

        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, index_i32, addr, &
                error_msg)) return
        if (.not. emit_i32_load(context%session, addr, value, error_msg)) return
        call set_empty(error_msg)
    end subroutine load_array_element_at_runtime_index_i32

    subroutine lower_runtime_i32_sum(context, source_index, value, error_msg)
        ! sum(a) over a rank-1 integer array whose extent is only known at
        ! runtime (the hidden extent argument ABI): no compile-time array_size
        ! to unroll over, so walk a genuine LIRIC loop instead, using the same
        ! reserved-backedge-vreg PHI technique as a counted DO
        ! (session_program_lowering_loops.inc).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: source_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent
        type(lr_operand_desc_t) :: entry_acc, entry_idx
        type(lr_operand_desc_t) :: header_acc, header_idx
        type(lr_operand_desc_t) :: backedge_acc, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_acc, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block
        integer(c_int32_t) :: acc_vreg, idx_vreg

        value = i32_immediate(context%session, 0_c_int64_t)
        extent = context%symbols(source_index)%runtime_dim_size(1)

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
        call load_array_element_at_runtime_index_i32(context, source_index, &
                                                      header_idx, element, &
                                                      error_msg)
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
    end subroutine lower_runtime_i32_sum
