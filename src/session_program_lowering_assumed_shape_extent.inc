    ! Shape queries and runtime helpers shared by the assumed-shape descriptor
    ! ABI (#334). The descriptor construction and binding themselves live in
    ! session_program_lowering_assumed_shape_descriptor.inc.

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

    logical function dummy_declared_optional(arena, body_indices, param_name) &
            result(is_optional)
        ! Whether the dummy named param_name is declared OPTIONAL in the body
        ! whose declarations are body_indices.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: param_name
        integer :: i, k
        logical :: name_matches

        is_optional = .false.
        if (.not. allocated(body_indices)) return
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
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
                    is_optional = decl%is_optional
                    return
                end if
            end select
        end do
    end function dummy_declared_optional

    logical function callee_dummy_assumed_shape_optional(arena, callee_name, &
                                                          param_pos) &
            result(is_optional)
        ! Whether callee_name's param_pos-th dummy is declared OPTIONAL.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer :: n
        character(len=:), allocatable :: name, name_err, node_type
        type(subroutine_def_node), pointer :: sb_node
        type(function_def_node), pointer :: fn_node

        is_optional = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1 .or. &
                    param_pos > size(sb_node%param_indices)) return
                call parameter_name(arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                is_optional = dummy_declared_optional(arena, &
                                                      sb_node%body_indices, name)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1 .or. &
                    param_pos > size(fn_node%param_indices)) return
                call parameter_name(arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                is_optional = dummy_declared_optional(arena, &
                                                      fn_node%body_indices, name)
                return
            end if
        end do
    end function callee_dummy_assumed_shape_optional

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
        type(function_def_node), pointer :: fn_node

        rank = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            node_type = get_node_type_at(arena, n)
            if (node_type == 'subroutine_def_node' .or. &
                node_type == 'subroutine_def') then
                sb_node => get_node_as_subroutine_def(arena, n)
                if (.not. associated(sb_node)) cycle
                if (.not. allocated(sb_node%name)) cycle
                if (.not. same_name(sb_node%name, callee_name)) cycle
                if (.not. allocated(sb_node%param_indices)) return
                if (param_pos < 1 .or. &
                    param_pos > size(sb_node%param_indices)) return
                call parameter_name(arena, sb_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                rank = dummy_assumed_shape_rank(arena, sb_node%body_indices, name)
                return
            else if (node_type == 'function_def_node' .or. &
                     node_type == 'function_def') then
                fn_node => get_node_as_function_def(arena, n)
                if (.not. associated(fn_node)) cycle
                if (.not. allocated(fn_node%name)) cycle
                if (.not. same_name(fn_node%name, callee_name)) cycle
                if (.not. allocated(fn_node%param_indices)) return
                if (param_pos < 1 .or. &
                    param_pos > size(fn_node%param_indices)) return
                call parameter_name(arena, fn_node%param_indices(param_pos), &
                                    name, name_err)
                if (len_trim(name_err) > 0) return
                rank = dummy_assumed_shape_rank(arena, fn_node%body_indices, name)
                return
            end if
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
        symbol_index = find_symbol_compat(context, id_name)
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
        symbol_index = find_symbol_compat(context, id_name)
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


    subroutine load_array_element_at_runtime_index(context, symbol_index, &
                                                   index_i32, value, error_msg)
        ! Element load at a runtime (non-immediate) 0-based i32 index for a
        ! rank-1 array symbol of integer, real(4), or real(8) elements. The
        ! GEP's declared array-size parameter does not affect the computed
        ! address for a single-index rank-1 access, so it is harmless that a
        ! descriptor-bound dummy's array_size stays the 0 sentinel.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: index_i32
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr

        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_F32)
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, index_i32, &
                    addr, error_msg)) return
            if (.not. emit_liric_f32_load(context%session, addr, value, &
                                          error_msg)) return
        case (VALUE_F64)
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, index_i32, &
                    addr, error_msg)) return
            if (.not. emit_liric_f64_load(context%session, addr, value, &
                                          error_msg)) return
        case default
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, index_i32, &
                    addr, error_msg)) return
            if (.not. emit_i32_load(context%session, addr, value, error_msg)) &
                return
        end select
        call set_empty(error_msg)
    end subroutine load_array_element_at_runtime_index

    subroutine runtime_sum_accumulator_slot(context, value_kind, slot, zero, &
                                            error_msg)
        ! Accumulator stack slot and typed zero for a runtime reduction loop.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: slot
        type(lr_operand_desc_t), intent(out) :: zero
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F32)
            if (.not. emit_liric_f32_alloca(context%session, slot, error_msg)) &
                return
            zero = liric_f32_immediate(context%session, 0.0_c_float)
            if (.not. emit_liric_f32_store(context%session, zero, slot, &
                                           error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_alloca(context%session, slot, error_msg)) &
                return
            zero = liric_f64_immediate(context%session, 0.0_c_double)
            if (.not. emit_liric_f64_store(context%session, zero, slot, &
                                           error_msg)) return
        case default
            if (.not. emit_i32_alloca(context%session, slot, error_msg)) return
            zero = i32_immediate(context%session, 0_c_int64_t)
            if (.not. emit_i32_store(context%session, zero, slot, error_msg)) &
                return
        end select
        call set_empty(error_msg)
    end subroutine runtime_sum_accumulator_slot

    subroutine runtime_sum_accumulate(context, value_kind, slot, element, &
                                      error_msg)
        ! One accumulation step of a runtime reduction loop: slot = slot + element.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: slot
        type(lr_operand_desc_t), intent(in) :: element
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: current, updated

        select case (value_kind)
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, slot, current, &
                                          error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, current, &
                                            element, updated, error_msg)) return
            if (.not. emit_liric_f32_store(context%session, updated, slot, &
                                           error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, slot, current, &
                                          error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, current, &
                                            element, updated, error_msg)) return
            if (.not. emit_liric_f64_store(context%session, updated, slot, &
                                           error_msg)) return
        case default
            if (.not. emit_i32_load(context%session, slot, current, error_msg)) &
                return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, current, &
                                      element, updated, error_msg)) return
            if (.not. emit_i32_store(context%session, updated, slot, error_msg)) &
                return
        end select
        call set_empty(error_msg)
    end subroutine runtime_sum_accumulate

    subroutine runtime_sum_result(context, value_kind, slot, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: slot
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, slot, value, &
                                          error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, slot, value, &
                                          error_msg)) return
        case default
            if (.not. emit_i32_load(context%session, slot, value, error_msg)) &
                return
        end select
        call set_empty(error_msg)
    end subroutine runtime_sum_result

    subroutine lower_runtime_sum(context, source_index, value, error_msg)
        ! sum(a) over a rank-1 array whose extent is known only at run time (an
        ! assumed-shape dummy bound through the caller's array descriptor):
        ! there is no compile-time array_size to unroll over, so walk a genuine
        ! LIRIC loop. The loop counter uses the reserved-backedge-vreg PHI
        ! technique of a counted DO; the accumulator lives in a stack slot so
        ! integer and real element kinds share one loop shape.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: source_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, accumulator, zero
        type(lr_operand_desc_t) :: entry_idx, header_idx, backedge_idx
        type(lr_operand_desc_t) :: condition, element, next_idx
        integer(c_int32_t) :: entry_block, header_block, body_block, &
                              latch_block, exit_block
        integer(c_int32_t) :: idx_vreg
        integer :: value_kind

        value_kind = context%symbols(source_index)%value_kind
        extent = context%symbols(source_index)%runtime_dim_size(1)

        call runtime_sum_accumulator_slot(context, value_kind, accumulator, &
                                          zero, error_msg)
        if (len_trim(error_msg) > 0) return
        value = zero

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
        call load_array_element_at_runtime_index(context, source_index, &
                                                 header_idx, element, error_msg)
        if (len_trim(error_msg) > 0) return
        call runtime_sum_accumulate(context, value_kind, accumulator, element, &
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
        call runtime_sum_result(context, value_kind, accumulator, value, error_msg)
    end subroutine lower_runtime_sum
