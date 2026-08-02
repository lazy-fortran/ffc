    subroutine lower_derived_whole_assignment(arena, node, context, &
                                              symbol_index, handled, error_msg)
        ! Whole-derived scalar assignment: a structure constructor RHS
        ! (y = t(...)) or a plain derived variable copy (y = x). Sets handled
        ! only when the form is lowered; otherwise the caller reports the
        ! existing unsupported-assignment diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index, src_index
        character(len=:), allocatable :: id_name, id_err

        handled = .false.
        call set_empty(error_msg)
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0) return
        if (context%symbols(symbol_index)%is_array) return
        if (.not. node_exists(arena, node%value_index)) return

        if (is_derived_constructor_call(arena, node%value_index, context, &
                                        type_index)) then
            if (constructor_uses_user_routine(context, &
                    context%derived_types(type_index)%name)) return
            ! y = t() re-applies the type's default initialisation, covering
            ! types with nested components the argument path cannot construct.
            if (constructor_call_is_argumentless(arena, node%value_index)) then
                call store_type_default_slots(context, symbol_index, &
                    total_component_slots(context, type_index), type_index, 0, &
                    error_msg)
                if (len_trim(error_msg) == 0) handled = .true.
                return
            end if
            call lower_derived_constructor_assignment(arena, node%value_index, &
                context, symbol_index, type_index, handled, error_msg)
            return
        end if

        if (is_identifier(arena, node%value_index)) then
            call get_identifier_name(arena, node%value_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            src_index = find_symbol_compat(context, id_name)
            if (src_index <= 0) return
            if (.not. context%symbols(src_index)%is_derived) return
            if (context%symbols(src_index)%is_array) return
            if (context%symbols(src_index)%derived_type_index /= type_index) return
            call lower_derived_scalar_copy(context, symbol_index, src_index, &
                                           error_msg)
            if (len_trim(error_msg) == 0) handled = .true.
        end if
    end subroutine lower_derived_whole_assignment

    subroutine lower_derived_array_whole_assignment(arena, node, context, &
                                                    symbol_index, handled, &
                                                    error_msg)
        ! Scalar expansion for a derived allocatable array (a = scalar).  The
        ! array's element extent is runtime descriptor state, so copy one flat
        ! derived instance per element through a counted LIRIC loop.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name, id_error
        integer :: source_index, type_index, slots
        integer(c_int64_t) :: element_bytes
        type(lr_operand_desc_t) :: extent64, extent, descriptor, data_ptr
        type(lr_operand_desc_t) :: source_ptr, entry_index, header_index
        type(lr_operand_desc_t) :: backedge_index, next_index, condition
        type(lr_operand_desc_t) :: index64, byte_offset, destination_ptr
        integer(c_int32_t) :: entry_block, header_block, body_block
        integer(c_int32_t) :: latch_block, exit_block, index_vreg

        handled = .false.
        call set_empty(error_msg)
        if (context%symbols(symbol_index)%array_rank /= 1) return
        if (.not. node_exists(arena, node%value_index)) return
        if (.not. is_identifier(arena, node%value_index)) return
        call get_identifier_name(arena, node%value_index, source_name, id_error)
        if (len_trim(id_error) > 0) return
        source_index = find_symbol_compat(context, source_name)
        if (source_index <= 0) return
        if (.not. context%symbols(source_index)%is_derived) return
        if (context%symbols(source_index)%is_array) return
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0 .or. context%symbols(source_index)%derived_type_index &
            /= type_index) return

        slots = total_component_slots(context, type_index)
        if (slots <= 0) return
        element_bytes = int(slots, c_int64_t) * 4_c_int64_t
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        call emit_alloc_desc_load_extent(context, descriptor, 1, extent64, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i64_to_i32(context%session, extent64, extent, &
                                       error_msg)) return
        source_ptr = context%symbols(source_index)%element_address

        entry_index = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.
        index_vreg = reserve_i32_vreg(context%session)
        if (index_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a derived array copy index'
            return
        end if
        backedge_index = i32_vreg(context%session, index_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_index, entry_block, &
                backedge_index, latch_block, header_index, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_index, &
                extent, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_i32_to_i64(context%session, header_index, index64, &
                                        error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, index64, &
                i64_immediate(context%session, element_bytes), byte_offset, &
                error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                                      destination_ptr, error_msg)) return
        call release_derived_array_element_components(context, destination_ptr, &
            type_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_memcpy(context%session, destination_ptr, source_ptr, &
                i64_immediate(context%session, element_bytes), error_msg)) return
        call deep_copy_derived_array_element_components(context, destination_ptr, &
            source_ptr, type_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_index, &
                i32_immediate(context%session, 1_c_int64_t), index_vreg, &
                next_index, error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        handled = .true.
        call set_empty(error_msg)
    end subroutine lower_derived_array_whole_assignment

    subroutine release_derived_array_element_components(context, element_address, &
                                                        type_index, error_msg)
        ! Release allocatable component data in one destination element before
        ! its inline descriptors are overwritten by intrinsic assignment.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: element_address
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr
        integer :: c, slot

        call set_empty(error_msg)
        do c = 1, context%derived_types(type_index)%component_count
            if (.not. component_is_allocatable(context, type_index, c) .and. &
                .not. component_is_alloc_array(context, type_index, c)) cycle
            slot = component_start_slot(context, type_index, c)
            if (.not. emit_ptr_offset(context%session, element_address, &
                    int(slot * 4, c_int64_t), descriptor, error_msg)) return
            if (.not. emit_ptr_load(context%session, descriptor, data_ptr, &
                                    error_msg)) return
            if (.not. emit_free(context%session, data_ptr, error_msg)) return
        end do
    end subroutine release_derived_array_element_components

    subroutine deep_copy_derived_array_element_components(context, dest_address, &
                                                          source_address, type_index, &
                                                          error_msg)
        ! Re-own allocatable component data after the flat slot copy. The
        ! descriptor layout is shared with allocatable arrays: pointer at byte
        ! zero and extent at byte eight. A null source remains unallocated.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: dest_address, source_address
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: source_desc, dest_desc, source_data, new_data
        type(lr_operand_desc_t) :: extent, bytes, condition
        integer(c_int32_t) :: copy_block, merge_block
        integer :: c, slot, elem_bytes, inner_bytes

        call set_empty(error_msg)
        do c = 1, context%derived_types(type_index)%component_count
            if (.not. component_is_allocatable(context, type_index, c) .and. &
                .not. component_is_alloc_array(context, type_index, c)) cycle
            if (component_is_alloc_char(context, type_index, c)) cycle
            slot = component_start_slot(context, type_index, c)
            if (.not. emit_ptr_offset(context%session, source_address, &
                    int(slot * 4, c_int64_t), source_desc, error_msg)) return
            if (.not. emit_ptr_offset(context%session, dest_address, &
                    int(slot * 4, c_int64_t), dest_desc, error_msg)) return
            if (.not. emit_ptr_load(context%session, source_desc, source_data, &
                                    error_msg)) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, source_data, &
                    null_ptr_operand(context), condition, error_msg)) return
            copy_block = create_liric_block(context%session)
            merge_block = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, condition, copy_block, &
                                        merge_block, error_msg)) return

            if (.not. set_liric_block(context%session, copy_block, error_msg)) return
            context%current_block_id = copy_block
            context%current_block_terminated = .false.
            elem_bytes = int(allocatable_elem_size( &
                component_kind(context, type_index, c)))
            if (component_is_alloc_array(context, type_index, c)) then
                inner_bytes = alloc_derived_inner_bytes(context, type_index, c)
                if (inner_bytes > 0) elem_bytes = inner_bytes
                if (.not. emit_i64_load_at(context%session, source_desc, &
                        8_c_int64_t, extent, error_msg)) return
                if (.not. emit_i64_binary(context%session, LR_OP_MUL, extent, &
                        i64_immediate(context%session, int(elem_bytes, c_int64_t)), &
                        bytes, error_msg)) return
            else
                bytes = i64_immediate(context%session, int(elem_bytes, c_int64_t))
            end if
            if (.not. emit_malloc(context%session, bytes, new_data, error_msg)) return
            if (.not. emit_memcpy(context%session, new_data, source_data, bytes, &
                    error_msg)) return
            if (.not. emit_ptr_store(context%session, new_data, dest_desc, &
                                    error_msg)) return
            if (component_is_alloc_array(context, type_index, c)) then
                if (.not. emit_i64_load_at(context%session, source_desc, &
                        8_c_int64_t, extent, error_msg)) return
                if (.not. emit_i64_store_at(context%session, extent, dest_desc, &
                        8_c_int64_t, error_msg)) return
            end if
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

            if (.not. set_liric_block(context%session, merge_block, error_msg)) return
            context%current_block_id = merge_block
            context%current_block_terminated = .false.
        end do
        call set_empty(error_msg)
    end subroutine deep_copy_derived_array_element_components

    subroutine lower_derived_scalar_copy(context, dest_index, src_index, error_msg)
        ! Copy a derived scalar slot-by-slot. Raw i32 slots preserve every
        ! component kind (int, real f32/f64 spans, logical, character bytes,
        ! nested aggregates) since the layout is identical for both operands.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_index, src_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index, slots, s, c
        type(lr_operand_desc_t) :: src_addr, dst_addr, val

        call set_empty(error_msg)
        type_index = context%symbols(dest_index)%derived_type_index
        do c = 1, context%derived_types(type_index)%component_count
            if (component_is_alloc_array(context, type_index, c) .and. &
                component_kind(context, type_index, c) == VALUE_CHARACTER) then
                error_msg = 'copying a derived value with a character '// &
                            'allocatable array component is not supported'
                return
            end if
        end do
        slots = total_component_slots(context, type_index)
        ! The destination's own allocatable character data is released before
        ! its descriptor is overwritten, so each owned buffer is freed exactly
        ! once (#402). Self-assignment keeps its data and needs no release.
        if (dest_index /= src_index) then
            call release_derived_alloc_char_components(context, dest_index, &
                                                       type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        do s = 0, slots - 1
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(src_index)%array_size, &
                    context%symbols(src_index)%element_address, &
                    i32_immediate(context%session, int(s, c_int64_t)), &
                    src_addr, error_msg)) return
            if (.not. emit_i32_load(context%session, src_addr, val, error_msg)) &
                return
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(dest_index)%array_size, &
                    context%symbols(dest_index)%element_address, &
                    i32_immediate(context%session, int(s, c_int64_t)), &
                    dst_addr, error_msg)) return
            if (.not. emit_i32_store(context%session, val, dst_addr, error_msg)) &
                return
        end do
        call deep_copy_alloc_components(context, dest_index, src_index, &
                                        type_index, error_msg)
    end subroutine lower_derived_scalar_copy

    subroutine deep_copy_alloc_components(context, dest_index, src_index, &
                                          type_index, error_msg)
        ! Give the destination independent ownership of every allocatable
        ! component the raw slot copy left aliasing the source: allocate fresh
        ! storage of the same byte span, copy the contents, and overwrite the
        ! destination's inline data pointer. An unallocated source component
        ! leaves the copied null pointer in place.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_index, src_index, type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: c, elem_bytes, inner_bytes
        logical :: is_array_comp
        type(lr_operand_desc_t) :: char_slot

        call set_empty(error_msg)
        do c = 1, context%derived_types(type_index)%component_count
            is_array_comp = component_is_alloc_array(context, type_index, c)
            if (.not. is_array_comp .and. &
                .not. component_is_allocatable(context, type_index, c)) cycle
            if (component_is_alloc_char(context, type_index, c)) then
                ! A deferred-length character component's byte span lives in its
                ! inline descriptor, not in a fixed element size, so it re-owns
                ! its data through the character descriptor path (#402).
                call alloc_char_component_slot_address(context, dest_index, &
                    type_index, c, char_slot, error_msg)
                if (len_trim(error_msg) > 0) return
                call reown_alloc_char_component(context, char_slot, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            elem_bytes = int(allocatable_elem_size( &
                             component_kind(context, type_index, c)))
            if (is_array_comp) then
                inner_bytes = alloc_derived_inner_bytes(context, type_index, c)
                if (inner_bytes > 0) elem_bytes = inner_bytes
            end if
            call deep_copy_one_alloc_component(context, dest_index, src_index, &
                type_index, c, is_array_comp, elem_bytes, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine deep_copy_alloc_components

    subroutine deep_copy_one_alloc_component(context, dest_index, src_index, &
            type_index, comp_index, is_array_comp, elem_bytes, error_msg)
        ! Reallocate and copy one allocatable component under a runtime null
        ! guard: when the source pointer is null the destination keeps the null
        ! the slot copy already stored.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_index, src_index, type_index, comp_index
        logical, intent(in) :: is_array_comp
        integer, intent(in) :: elem_bytes
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: src_desc, dst_desc, src_data, new_data
        type(lr_operand_desc_t) :: extent64, bytes64, condition
        integer(c_int32_t) :: copy_block, merge_block
        integer :: slot

        slot = component_start_slot(context, type_index, comp_index)
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(src_index)%array_size, &
                context%symbols(src_index)%element_address, &
                i32_immediate(context%session, int(slot, c_int64_t)), &
                src_desc, error_msg)) return
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(dest_index)%array_size, &
                context%symbols(dest_index)%element_address, &
                i32_immediate(context%session, int(slot, c_int64_t)), &
                dst_desc, error_msg)) return
        if (.not. emit_ptr_load(context%session, src_desc, src_data, error_msg)) &
            return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, src_data, &
                null_ptr_operand(context), condition, error_msg)) return

        copy_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, condition, copy_block, &
                merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, copy_block, error_msg)) return
        context%current_block_id = copy_block
        context%current_block_terminated = .false.
        if (is_array_comp) then
            if (.not. emit_i64_load_at(context%session, src_desc, 8_c_int64_t, &
                    extent64, error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_MUL, extent64, &
                    i64_immediate(context%session, int(elem_bytes, c_int64_t)), &
                    bytes64, error_msg)) return
        else
            bytes64 = i64_immediate(context%session, int(elem_bytes, c_int64_t))
        end if
        if (.not. emit_malloc(context%session, bytes64, new_data, error_msg)) return
        if (.not. emit_memcpy(context%session, new_data, src_data, bytes64, &
                error_msg)) return
        if (.not. emit_ptr_store(context%session, new_data, dst_desc, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine deep_copy_one_alloc_component

    subroutine alloc_char_component_slot_address(context, symbol_index, &
                                                 type_index, comp_index, addr, &
                                                 error_msg)
        ! Address of the inline character descriptor of component comp_index in
        ! the derived scalar held by symbol_index.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index, type_index, comp_index
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, &
                    int(component_start_slot(context, type_index, comp_index), &
                        c_int64_t)), addr, error_msg)) return
    end subroutine alloc_char_component_slot_address

    subroutine release_derived_alloc_char_components(context, symbol_index, &
                                                     type_index, error_msg)
        ! free() every allocatable character component of a derived scalar and
        ! clear its descriptor. free(NULL) is a no-op, so an unallocated
        ! component is handled without a branch.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index, type_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr, data_ptr
        integer :: c

        call set_empty(error_msg)
        do c = 1, context%derived_types(type_index)%component_count
            if (.not. component_is_alloc_char(context, type_index, c)) cycle
            call alloc_char_component_slot_address(context, symbol_index, &
                                                   type_index, c, addr, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_load(context%session, addr, data_ptr, error_msg)) &
                return
            if (.not. emit_free(context%session, data_ptr, error_msg)) return
            if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                                     addr, error_msg)) return
            if (.not. emit_i64_store_at(context%session, &
                    i64_immediate(context%session, 0_c_int64_t), addr, &
                    8_c_int64_t, error_msg)) return
        end do
    end subroutine release_derived_alloc_char_components

    subroutine lower_derived_constructor_assignment(arena, call_index, context, &
            symbol_index, type_index, handled, error_msg)
        ! y = t(a1, ..., an): store each provided component value into the
        ! target's slot layout; components omitted from the constructor keep
        ! their compile-time default. Only scalar integer/real/logical
        ! components are handled; any other component leaves handled false so
        ! the caller falls back to the unsupported-constructor diagnostic.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n_comp, c
        integer, allocatable :: comp_value_index(:)

        handled = .false.
        call set_empty(error_msg)
        n_comp = context%derived_types(type_index)%component_count
        if (n_comp == 1 .and. allocated(context%derived_types(type_index)% &
                component_names)) then
            if (same_name(context%derived_types(type_index)%component_names(1), &
                          '__ffc_placeholder')) then
                ! Empty derived types use one hidden storage slot so their
                ! objects still have an address.  The slot is not a Fortran
                ! component and must not participate in constructor mapping.
                handled = .true.
                return
            end if
        end if
        if (n_comp <= 0) then
            ! An empty structure constructor is the only value that can
            ! initialise an empty derived object.  There are no component
            ! stores to emit, but the DATA value was handled successfully.
            handled = .true.
            return
        end if
        do c = 1, n_comp
            if (.not. constructor_component_storeable(context, type_index, c)) return
        end do

        call map_constructor_args(arena, call_index, context, type_index, &
                                  comp_value_index)
        if (.not. allocated(comp_value_index)) return
        do c = 1, n_comp
            if (comp_value_index(c) <= 0 .and. &
                .not. context%derived_types(type_index)%component_has_default(c)) &
                return
        end do

        do c = 1, n_comp
            if (comp_value_index(c) > 0) then
                call emit_constructor_component(arena, context, symbol_index, &
                    type_index, c, comp_value_index(c), error_msg)
            else
                call emit_constructor_default(context, symbol_index, type_index, &
                    c, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do
        handled = .true.
    end subroutine lower_derived_constructor_assignment

    logical function constructor_component_storeable(context, type_index, &
            comp_index)
        ! A component the structure-constructor path can store: a scalar
        ! integer, real, logical, or fixed-length character, or a nested plain
        ! derived scalar whose leaves use the same representation.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        integer :: nested_type
        integer :: nested_count

        constructor_component_storeable = .false.
        nested_type = context%derived_types(type_index)%component_type_index(comp_index)
        if (nested_type /= 0) then
            if (component_is_alloc_array(context, type_index, comp_index)) return
            nested_count = nested_element_count(context, type_index, comp_index)
            if (nested_count /= 1) return
            constructor_component_storeable = &
                plain_constructor_type_supported(context, nested_type)
            return
        end if
        select case (component_kind(context, type_index, comp_index))
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            if (context%derived_types(type_index)%component_array_size(comp_index) &
                == 1) constructor_component_storeable = .true.
        case (VALUE_CHARACTER)
            if (component_is_alloc_array(context, type_index, comp_index)) return
            if (derived_component_char_length(context, type_index, comp_index) &
                > 0) constructor_component_storeable = .true.
        end select
    end function constructor_component_storeable

    recursive logical function plain_constructor_type_supported(context, type_index) &
            result(supported)
        ! A nested constructor is only accepted when every leaf is a scalar
        ! component already supported by the direct constructor path. Arrays,
        ! allocatables, pointers, and character components with dynamic storage
        ! remain outside this issue.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer :: c, nested_type

        supported = .false.
        if (type_index <= 0 .or. type_index > context%derived_type_count) return
        if (context%derived_types(type_index)%component_count <= 0) return
        do c = 1, context%derived_types(type_index)%component_count
            if (component_is_allocatable(context, type_index, c) .or. &
                component_is_alloc_array(context, type_index, c)) return
            if (component_is_pointer(context, type_index, c)) return
            nested_type = context%derived_types(type_index)% &
                          component_type_index(c)
            if (nested_type /= 0) then
                if (nested_element_count(context, type_index, c) /= 1) return
                if (.not. plain_constructor_type_supported(context, nested_type)) &
                    return
            else
                select case (component_kind(context, type_index, c))
                case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL, &
                      VALUE_CHARACTER)
                    continue
                case default
                    return
                end select
            end if
        end do
        supported = .true.
    end function plain_constructor_type_supported

    subroutine map_constructor_args(arena, call_index, context, type_index, &
            comp_value_index)
        ! Map constructor actual arguments to component slots. Positional
        ! arguments fill components in declaration order; keyword arguments
        ! (name=value) target the named component. On any inconsistency the
        ! result stays unallocated so the caller declines the constructor.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, allocatable, intent(out) :: comp_value_index(:)
        integer :: n_comp, a, pos, ci, value_index
        logical :: is_kw
        character(len=:), allocatable :: kw_name

        n_comp = context%derived_types(type_index)%component_count
        allocate(comp_value_index(n_comp))
        comp_value_index = 0
        pos = 0
        select type (cn => arena%entries(call_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(cn%arg_indices)) return
            do a = 1, size(cn%arg_indices)
                call classify_constructor_arg(arena, cn%arg_indices(a), is_kw, &
                                              kw_name, value_index)
                if (is_kw) then
                    ci = find_derived_component(context, type_index, kw_name)
                    if (ci <= 0) then
                        deallocate(comp_value_index); return
                    end if
                else
                    pos = pos + 1
                    ci = pos
                    if (ci > n_comp) then
                        deallocate(comp_value_index); return
                    end if
                end if
                if (comp_value_index(ci) /= 0) then
                    deallocate(comp_value_index); return
                end if
                if (.not. is_safe_constructor_value(arena, value_index)) then
                    deallocate(comp_value_index); return
                end if
                comp_value_index(ci) = value_index
            end do
        class default
            deallocate(comp_value_index)
        end select
    end subroutine map_constructor_args

    subroutine classify_constructor_arg(arena, arg_index, is_kw, kw_name, &
            value_index)
        ! A constructor actual argument is either positional (value_index is
        ! the argument itself) or keyword name=value (a binary '=' whose left
        ! identifier names the component and whose right operand is the value).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        logical, intent(out) :: is_kw
        character(len=:), allocatable, intent(out) :: kw_name
        integer, intent(out) :: value_index
        character(len=:), allocatable :: op, id_err
        integer :: left_index, right_index, ln, col

        is_kw = .false.
        call set_empty(kw_name)
        value_index = arg_index
        if (.not. is_binary_op(arena, arg_index)) return
        call get_binary_op_info(arena, arg_index, op, left_index, right_index, &
                                ln, col, id_err)
        if (len_trim(id_err) > 0) return
        if (op /= '=') return
        if (.not. is_identifier(arena, left_index)) return
        call get_identifier_name(arena, left_index, kw_name, id_err)
        if (len_trim(id_err) > 0) then
            call set_empty(kw_name)
            return
        end if
        is_kw = .true.
        value_index = right_index
    end subroutine classify_constructor_arg

    logical function is_safe_constructor_value(arena, node_index)
        ! A constructor value the scalar-component path can evaluate directly.
        ! Anything else (e.g. a keyword-argument wrapper the frontend keeps as a
        ! non-value node) makes the caller decline the constructor rather than
        ! risk mapping a value to the wrong component.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_safe_constructor_value = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            is_safe_constructor_value = .true.
            return
        end if
        if (is_literal(arena, node_index)) then
            is_safe_constructor_value = .true.
            return
        end if
        if (is_binary_op(arena, node_index)) then
            is_safe_constructor_value = .true.
            return
        end if
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            is_safe_constructor_value = .true.
        type is (component_access_node)
            is_safe_constructor_value = .true.
        end select
    end function is_safe_constructor_value

    subroutine emit_constructor_component(arena, context, symbol_index, &
            type_index, comp_index, value_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        integer, intent(in) :: value_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr, val
        integer :: comp_kind
        integer :: dest_len
        integer :: nested_type
        type(lr_operand_desc_t) :: nested_address

        comp_kind = component_kind(context, type_index, comp_index)
        nested_type = context%derived_types(type_index)%component_type_index(comp_index)
        if (component_is_alloc_array(context, type_index, comp_index) .and. &
            comp_kind == VALUE_CHARACTER) then
            error_msg = 'character allocatable array component constructors are '// &
                        'not supported'
            return
        end if
        if (nested_type /= 0) then
            call emit_component_slot_address(context, symbol_index, type_index, &
                                             comp_index, nested_address, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_nested_constructor_value(arena, context, nested_address, &
                                               nested_type, value_index, error_msg)
            return
        end if
        call emit_component_slot_address(context, symbol_index, type_index, &
                                         comp_index, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (comp_kind == VALUE_CHARACTER) then
            dest_len = derived_component_char_length(context, type_index, &
                                                      comp_index)
            if (dest_len <= 0) then
                error_msg = 'derived type character component constructor '// &
                            'argument has no fixed length'
                return
            end if
            call store_char_component_value(arena, value_index, addr, dest_len, &
                                            context, error_msg)
            return
        end if
        call lower_component_rhs_value(arena, value_index, comp_kind, context, &
                                       val, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_typed_component_store(context, comp_kind, val, addr, error_msg)
    end subroutine emit_constructor_component

    subroutine emit_nested_constructor_value(arena, context, base_address, &
                                             type_index, value_index, error_msg)
        ! Store a nested structure constructor into the already-addressed
        ! component span. A different derived type is rejected before any
        ! component store is emitted.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base_address
        integer, intent(in) :: type_index
        integer, intent(in) :: value_index
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. is_derived_constructor_call(arena, value_index, context, &
                                              type_index)) then
            error_msg = 'incompatible derived type constructor argument for '// &
                        trim(context%derived_types(type_index)%name)
            return
        end if
        if (constructor_uses_user_routine(context, &
                context%derived_types(type_index)%name)) then
            error_msg = 'derived type constructor conflicts with a user procedure: '// &
                        trim(context%derived_types(type_index)%name)
            return
        end if
        call emit_nested_constructor_at_address(arena, value_index, context, &
                                                base_address, type_index, error_msg)
    end subroutine emit_nested_constructor_value

    subroutine emit_nested_constructor_at_address(arena, call_index, context, &
                                                  base_address, type_index, &
                                                  error_msg)
        ! Recursively map and store a plain constructor at an arbitrary byte
        ! address. This is the nested counterpart of the top-level symbol path.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base_address
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: component_values(:)
        integer :: c
        logical :: supported

        call set_empty(error_msg)
        supported = plain_constructor_type_supported(context, type_index)
        if (.not. supported) then
            error_msg = 'unsupported nested derived type constructor: '// &
                        trim(context%derived_types(type_index)%name)
            return
        end if
        call map_constructor_args(arena, call_index, context, type_index, &
                                   component_values)
        if (.not. allocated(component_values)) then
            error_msg = 'incompatible derived type constructor argument for '// &
                        trim(context%derived_types(type_index)%name)
            return
        end if
        do c = 1, context%derived_types(type_index)%component_count
            if (component_values(c) <= 0 .and. &
                .not. context%derived_types(type_index)%component_has_default(c)) then
                error_msg = 'missing component in derived type constructor: '// &
                            trim(context%derived_types(type_index)%component_names(c))
                return
            end if
        end do
        do c = 1, context%derived_types(type_index)%component_count
            if (component_values(c) > 0) then
                call emit_constructor_component_at_address(arena, context, &
                    base_address, type_index, c, component_values(c), error_msg)
            else
                call emit_constructor_default_at_address(context, base_address, &
                    type_index, c, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine emit_nested_constructor_at_address

    subroutine emit_constructor_component_at_address(arena, context, base_address, &
                                                     type_index, comp_index, &
                                                     value_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base_address
        integer, intent(in) :: type_index, comp_index, value_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address
        type(lr_operand_desc_t) :: component_value
        integer :: comp_kind, nested_type
        integer :: dest_len

        call component_address_from_base(context, base_address, type_index, &
                                         comp_index, component_address, error_msg)
        if (len_trim(error_msg) > 0) return
        nested_type = context%derived_types(type_index)%component_type_index(comp_index)
        if (nested_type /= 0) then
            call emit_nested_constructor_value(arena, context, component_address, &
                                               nested_type, value_index, error_msg)
            return
        end if
        comp_kind = component_kind(context, type_index, comp_index)
        if (comp_kind == VALUE_CHARACTER) then
            if (component_is_alloc_array(context, type_index, comp_index)) then
                error_msg = 'character allocatable array component constructors are '// &
                            'not supported'
                return
            end if
            dest_len = derived_component_char_length(context, type_index, comp_index)
            if (dest_len <= 0) then
                error_msg = 'derived type character component constructor '// &
                            'argument has no fixed length'
                return
            end if
            call store_char_component_value(arena, value_index, component_address, &
                                            dest_len, context, error_msg)
            return
        end if
        call lower_component_rhs_value(arena, value_index, comp_kind, context, &
                                       component_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_typed_component_store(context, comp_kind, &
                                        component_value, component_address, &
                                        error_msg)
    end subroutine emit_constructor_component_at_address

    subroutine component_address_from_base(context, base_address, type_index, &
                                           comp_index, component_address, &
                                           error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base_address
        integer, intent(in) :: type_index, comp_index
        type(lr_operand_desc_t), intent(out) :: component_address
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: byte_offset

        byte_offset = int(component_start_slot(context, type_index, comp_index), &
                          c_int64_t) * 4_c_int64_t
        if (.not. emit_ptr_offset(context%session, base_address, byte_offset, &
                                  component_address, error_msg)) return
        call set_empty(error_msg)
    end subroutine component_address_from_base

    recursive subroutine emit_constructor_default_at_address(context, base_address, &
                                                             type_index, comp_index, &
                                                             error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base_address
        integer, intent(in) :: type_index, comp_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address, slot_address
        integer :: nested_type, w
        integer(c_int64_t), allocatable :: words(:)

        call component_address_from_base(context, base_address, type_index, &
                                         comp_index, component_address, error_msg)
        if (len_trim(error_msg) > 0) return
        nested_type = context%derived_types(type_index)%component_type_index(comp_index)
        if (nested_type /= 0) then
            call emit_nested_defaults_at_address(context, component_address, &
                                                 nested_type, error_msg)
            return
        end if
        call component_default_words(context, type_index, comp_index, words, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        do w = 1, size(words)
            slot_address = component_address
            if (w > 1) then
                if (.not. emit_i64_binary(context%session, LR_OP_ADD, &
                        component_address, &
                        i64_immediate(context%session, int(4*(w - 1), c_int64_t)), &
                        slot_address, error_msg)) return
            end if
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, words(w)), slot_address, &
                    error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine emit_constructor_default_at_address

    recursive subroutine emit_nested_defaults_at_address(context, base_address, &
                                                         type_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base_address
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: c

        call set_empty(error_msg)
        do c = 1, context%derived_types(type_index)%component_count
            if (.not. context%derived_types(type_index)% &
                    component_has_default(c)) then
                error_msg = 'missing component in nested derived type constructor: '// &
                            trim(context%derived_types(type_index)%component_names(c))
                return
            end if
            call emit_constructor_default_at_address(context, base_address, &
                                                     type_index, c, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine emit_nested_defaults_at_address

    subroutine emit_constructor_default(context, symbol_index, type_index, &
            comp_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr, slot_address
        integer(c_int64_t), allocatable :: words(:)
        integer :: w

        call emit_component_slot_address(context, symbol_index, type_index, &
                                         comp_index, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        call component_default_words(context, type_index, comp_index, words, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        do w = 1, size(words)
            slot_address = addr
            if (w > 1) then
                if (.not. emit_i64_binary(context%session, LR_OP_ADD, addr, &
                        i64_immediate(context%session, int(4*(w - 1), c_int64_t)), &
                        slot_address, error_msg)) return
            end if
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, words(w)), slot_address, &
                    error_msg)) return
        end do
    end subroutine emit_constructor_default

    subroutine emit_component_slot_address(context, symbol_index, type_index, &
            comp_index, addr, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, &
                    int(component_start_slot(context, type_index, comp_index), &
                        c_int64_t)), addr, error_msg)) then
            error_msg = 'constructor_component_gep: '//error_msg
        end if
    end subroutine emit_component_slot_address

    logical function constructor_uses_user_routine(context, name)
        ! True when name also denotes a user procedure (a generic interface or
        ! a contained function), so t(...) must dispatch to that routine rather
        ! than perform default structure construction.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        constructor_uses_user_routine = .true.
        do i = 1, context%generic_count
            if (same_name(context%generics(i)%generic_name, name)) return
        end do
        if (contained_function_kind(context, name) /= 0) return
        constructor_uses_user_routine = .false.
    end function constructor_uses_user_routine
