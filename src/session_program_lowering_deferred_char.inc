integer function resolve_concat_operand(arena, node_index, context, &
                                           out_name, operand_is_literal, error_msg, &
                                           operand_is_call, operand_is_view) &
      result(symbol_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: out_name
        logical, intent(out) :: operand_is_literal
        character(len=:), allocatable, intent(out) :: error_msg
        ! A call to a contained character-returning function is neither a
        ! literal nor a declared symbol: the caller materialises it through
        ! the descriptor return ABI before the concatenation reads it.
        logical, intent(out), optional :: operand_is_call
        ! A substring is a borrowed view: it resolves to a data pointer and a
        ! runtime length rather than to a symbol.
        logical, intent(out), optional :: operand_is_view
        character(len=:), allocatable :: name
        character(len=:), allocatable :: lit_value, lit_type
        integer :: idx

        symbol_index = 0
        operand_is_literal = .false.
        if (present(operand_is_call)) operand_is_call = .false.
        if (present(operand_is_view)) operand_is_view = .false.
        out_name = ''
        call set_empty(error_msg)

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'concat operand index does not reference an AST node'
            return
        end if

        if (present(operand_is_call)) then
            if (is_contained_char_result_call(arena, node_index, context)) then
                operand_is_call = .true.
                return
            end if
        end if

        if (present(operand_is_view)) then
            if (is_character_substring(arena, node_index, context)) then
                operand_is_view = .true.
                return
            end if
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            operand_is_literal = .true.
            out_name = lit_value
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            idx = find_symbol_compat(context, trim(name))
            if (idx > 0) then
                symbol_index = idx
                out_name = trim(name)
            else
                error_msg = 'concat operand not declared: '//trim(name)
            end if
            return
        end if

        select type (node => arena%entries(node_index)%node)
        class default
            error_msg = 'expected identifier or literal in concat operand'
        end select
    end function resolve_concat_operand

    subroutine lower_deferred_concat_assignment(arena, node, context, &
                                                dest_symbol_index, error_msg)
        ! str = <a // b // ...> for a deferred-length allocatable destination.
        !
        ! The concatenation tree is flattened to its leaf operands first, so an
        ! arbitrary operand count works and every leaf kind char_expr_operands
        ! understands - variable, literal, substring, function result,
        ! trim()/adjustl()/repeat() - is an operand here. This is the same
        ! flattening the fixed-length destination uses, so the two destinations
        ! agree on what a concatenation is instead of each carrying its own
        ! idea of one.
        !
        ! The result is heap storage the destination owns. Every operand is
        ! copied into it before the destination releases the block it currently
        ! holds, so an operand that aliases that block - `s = s // t`, or
        ! `s = s // s` - still reads valid bytes.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: data_ops(:), len_ops(:)
        integer, allocatable :: stat_len(:)
        type(lr_operand_desc_t) :: total_len, operand_len, wider_total
        type(lr_operand_desc_t) :: alloc_size, buf, null_pos
        integer :: count, i

        call set_empty(error_msg)
        ! A destination that reaches here with a declared width is a
        ! fixed-length result using the descriptor return ABI. It keeps that
        ! width: the concatenation pads with blanks or truncates to N instead
        ! of taking the operand sum, which is what an ordinary fixed-length
        ! assignment does.
        if (context%symbols(dest_symbol_index)%character_length > 0) then
            call lower_fixed_width_deferred_concat(arena, node, context, &
                                                   dest_symbol_index, error_msg)
            return
        end if

        allocate (data_ops(64), len_ops(64), stat_len(64))
        count = 0
        call collect_concat_operands(arena, node%value_index, context, data_ops, &
                                     len_ops, stat_len, count, error_msg)
        if (len_trim(error_msg) > 0) return
        if (count == 0) then
            error_msg = 'character concatenation has no operands'
            return
        end if

        ! The result length is the sum of the operand lengths. A leaf with a
        ! known declared width contributes it as a constant, including the
        ! blank padding of a fixed-length operand; the rest contribute their
        ! runtime length.
        total_len = i64_immediate(context%session, 0_c_int64_t)
        do i = 1, count
            if (stat_len(i) >= 0) then
                operand_len = i64_immediate(context%session, &
                                            int(stat_len(i), c_int64_t))
            else
                if (.not. emit_liric_i32_to_i64(context%session, len_ops(i), &
                        operand_len, error_msg)) return
            end if
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, total_len, &
                    operand_len, wider_total, error_msg)) return
            total_len = wider_total
        end do

        ! One trailing NUL so the %s printers stop at the result's end.
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, total_len, &
                i64_immediate(context%session, 1_c_int64_t), alloc_size, &
                error_msg)) return
        if (.not. emit_malloc(context%session, alloc_size, buf, error_msg)) return

        ! A deferred destination takes the whole concatenation, so nothing is
        ! clamped to a declared width here.
        call emit_concat_copies(context, data_ops, len_ops, stat_len, count, &
                                CONCAT_NO_CLAMP, buf, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, total_len, &
                null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return

        ! Every operand has been copied out, so the former block can go.
        call release_owned_character_storage(context, dest_symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, buf, &
                context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, total_len, &
                context%symbols(dest_symbol_index)%deferred_length, error_msg)) return
        call set_character_storage(context, dest_symbol_index, total_len, &
                                   LOWERING_CHARACTER_STORAGE_OWNED, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(dest_symbol_index)%value = buf
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_concat_assignment

    subroutine lower_fixed_width_deferred_concat(arena, node, context, &
                                                 dest_symbol_index, error_msg)
        ! Concatenate into a descriptor-returned destination whose declared
        ! length is fixed (character(len=N)). The buffer is exactly N content
        ! bytes plus a trailing null, blank-filled first so a short chain pads
        ! and an over-long one truncates. The descriptor records N, so LEN and
        ! the printed bytes match a fixed-length result under gfortran.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: data_ops(:), len_ops(:)
        integer, allocatable :: stat_len(:)
        type(lr_operand_desc_t) :: buf, offset, null_ptr, size_op
        character(len=64) :: null_name
        integer :: count, dest_len

        call set_empty(error_msg)
        dest_len = context%symbols(dest_symbol_index)%character_length
        allocate (data_ops(64), len_ops(64), stat_len(64))
        count = 0
        call collect_concat_operands(arena, node%value_index, context, data_ops, &
                                     len_ops, stat_len, count, error_msg)
        if (len_trim(error_msg) > 0) return
        if (count == 0) then
            error_msg = 'character concatenation has no operands'
            return
        end if

        size_op = i64_immediate(context%session, int(dest_len + 1, c_int64_t))
        if (context%in_internal_function .and. &
            dest_symbol_index == context%current_function_result_index) then
            ! The result outlives the callee frame, so its storage is owned
            ! heap memory the caller copies from and releases.
            if (.not. emit_malloc(context%session, size_op, buf, error_msg)) return
        else
            if (.not. emit_alloca_bytes(context%session, size_op, buf, &
                error_msg)) return
        end if
        call fill_spaces(context, buf, &
            i32_immediate(context%session, int(dest_len, c_int64_t)), error_msg)
        if (len_trim(error_msg) > 0) return

        call emit_concat_copies(context, data_ops, len_ops, stat_len, count, &
                                dest_len, buf, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, &
            i64_immediate(context%session, int(dest_len, c_int64_t)), offset, &
            error_msg)) return
        context%string_literal_count = context%string_literal_count + 1
        null_name = ffc_unit_global_name( &
            context, 'dfcat.term.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(null_name), '', &
                                      null_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_memcpy(context%session, offset, null_ptr, &
            i64_immediate(context%session, 1_c_int64_t), error_msg)) return

        if (.not. emit_ptr_store(context%session, buf, &
            context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, int(dest_len, c_int64_t)), &
            context%symbols(dest_symbol_index)%deferred_length, error_msg)) return

        context%symbols(dest_symbol_index)%value = buf
        context%symbols(dest_symbol_index)%has_character_value = .true.
    end subroutine lower_fixed_width_deferred_concat

    subroutine materialize_concat_literal(context, text, data_op, len_op, &
                                          error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: text
        type(lr_operand_desc_t), intent(out) :: data_op
        type(lr_operand_desc_t), intent(out) :: len_op
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: string_name

        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'cat.lit.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
                                      text, data_op, error_msg)
        if (len_trim(error_msg) > 0) return
        len_op = i64_immediate(context%session, int(len(text), c_int64_t))
        call set_empty(error_msg)
    end subroutine materialize_concat_literal

 subroutine lower_deferred_literal_assignment(literal_value, context, &
                                                  dest_symbol_index, error_msg)
        character(len=*), intent(in) :: literal_value
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text

        call strip_literal_quotes(literal_value, literal_text)
        call lower_deferred_folded_literal_assignment(literal_text, context, &
                                                      dest_symbol_index, error_msg)
    end subroutine lower_deferred_literal_assignment

    subroutine lower_deferred_folded_literal_assignment(literal_text, context, &
                                                        dest_symbol_index, &
                                                        error_msg)
        character(len=*), intent(in) :: literal_text
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_op
        type(lr_operand_desc_t) :: len_op
        integer :: fixed_length
        character(len=:), allocatable :: stored_text

        ! A fixed-length destination (character(len=N)) pads or truncates the
        ! literal to N before storing; a deferred destination keeps the literal
        ! length unchanged.
        fixed_length = context%symbols(dest_symbol_index)%character_length
        if (fixed_length > 0) then
            allocate (character(len=fixed_length) :: stored_text)
            ! stored_text is deferred-length allocatable, so a whole-variable
            ! assignment would reallocate it to the literal's own length and
            ! drop the blank padding. Assign through a substring reference so
            ! the declared width N is preserved: shorter literals pad with
            ! blanks, longer ones truncate.
            stored_text(1:fixed_length) = literal_text
            call materialize_concat_literal(context, stored_text, data_op, &
                                            len_op, error_msg)
        else
            call materialize_concat_literal(context, literal_text, data_op, &
                                            len_op, error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        ! The literal lives in a global, so the descriptor borrows it. Any heap
        ! block the descriptor still owned is released before it is overwritten.
        call release_owned_character_storage(context, dest_symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, data_op, &
            context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, len_op, &
            context%symbols(dest_symbol_index)%deferred_length, error_msg)) return
        call set_character_storage(context, dest_symbol_index, len_op, &
                                   LOWERING_CHARACTER_STORAGE_STATIC, error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(dest_symbol_index)%value = data_op
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_folded_literal_assignment

    subroutine register_char_temp(context, data_addr, storage_addr)
        ! Record a character result temporary so the statement that created it
        ! releases it. Registration is unconditional: a consumer that wants
        ! ownership takes it back with adopt_char_temp, which is a smaller and
        ! more obvious set of places than every consumer that does not.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_addr
        type(lr_operand_desc_t), intent(in) :: storage_addr
        type(lr_operand_desc_t), allocatable :: grown_data(:)
        type(lr_operand_desc_t), allocatable :: grown_storage(:)
        integer :: capacity

        if (.not. allocated(context%char_temp_data)) then
            allocate (context%char_temp_data(8))
            allocate (context%char_temp_storage(8))
            context%char_temp_count = 0
        end if
        capacity = size(context%char_temp_data)
        if (context%char_temp_count >= capacity) then
            allocate (grown_data(capacity*2))
            allocate (grown_storage(capacity*2))
            grown_data(1:capacity) = context%char_temp_data
            grown_storage(1:capacity) = context%char_temp_storage
            call move_alloc(grown_data, context%char_temp_data)
            call move_alloc(grown_storage, context%char_temp_storage)
        end if
        context%char_temp_count = context%char_temp_count + 1
        context%char_temp_data(context%char_temp_count) = data_addr
        context%char_temp_storage(context%char_temp_count) = storage_addr
    end subroutine register_char_temp

    subroutine adopt_char_temp(context, mark)
        ! A consumer took ownership of the temporaries created since mark, so
        ! the statement must not release them. Used where a deferred
        ! destination rebinds onto the return descriptor: the destination's own
        ! descriptor now owns the block and releases it at its next assignment
        ! or deallocate.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: mark

        if (mark < 0) return
        if (mark > context%char_temp_count) return
        context%char_temp_count = mark
    end subroutine adopt_char_temp

    integer function char_temp_mark(context) result(mark)
        ! How many temporaries are outstanding before lowering something.
        type(lowering_context_t), intent(in) :: context

        mark = context%char_temp_count
    end function char_temp_mark

    subroutine release_char_temps_since(context, mark, error_msg)
        ! Release every character result temporary registered since mark, in
        ! reverse order of creation, and forget them. Releasing is a no-op for
        ! a borrowed result: release_character_descriptor_storage frees only a
        ! descriptor whose storage class says it owns its bytes.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: mark
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (context%char_temp_count <= mark) return
        ! A terminated block takes no more instructions; the frame is going
        ! away with the process or the callee's return, so there is nothing to
        ! release into.
        if (context%current_block_terminated) then
            context%char_temp_count = mark
            return
        end if
        do i = context%char_temp_count, mark + 1, -1
            call release_character_descriptor_storage(context, &
                context%char_temp_data(i), context%char_temp_storage(i), &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        context%char_temp_count = mark
    end subroutine release_char_temps_since

    subroutine allocate_deferred_char_descriptor(context, descriptor_ptr, &
                                                  data_addr, len_addr, &
                                                  error_msg, capacity_addr, &
                                                  storage_addr)
        ! Stack-allocate a canonical 32-byte character descriptor
        ! (docs/CHARACTER_DESCRIPTOR_ABI.md) and hand back the addresses of its
        ! fields. Every field starts at the ABI null state, so a descriptor
        ! that is filled in by a callee, or is only ever borrowed, still has a
        ! well-defined storage class. The {data, length} prefix is unchanged,
        ! so an actual-argument descriptor still matches what a
        ! character(len=*) dummy binds against.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: descriptor_ptr
        type(lr_operand_desc_t), intent(out) :: data_addr
        type(lr_operand_desc_t), intent(out) :: len_addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), intent(out), optional :: capacity_addr
        type(lr_operand_desc_t), intent(out), optional :: storage_addr
        type(lr_operand_desc_t) :: capacity_slot, storage_slot

        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, 32_c_int64_t), descriptor_ptr, &
            error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor_ptr, 0_c_int64_t, &
            data_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor_ptr, 8_c_int64_t, &
            len_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor_ptr, 16_c_int64_t, &
            capacity_slot, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor_ptr, 24_c_int64_t, &
            storage_slot, error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, 0_c_int64_t), data_addr, &
            error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, 0_c_int64_t), len_addr, &
            error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, 0_c_int64_t), capacity_slot, &
            error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, 0_c_int64_t), storage_slot, &
            error_msg)) return
        if (present(capacity_addr)) capacity_addr = capacity_slot
        if (present(storage_addr)) storage_addr = storage_slot
        call set_empty(error_msg)
    end subroutine allocate_deferred_char_descriptor

    subroutine release_character_descriptor_storage(context, data_addr, &
                                                     storage_addr, error_msg)
        ! Release the heap block a descriptor owns, given the addresses of its
        ! data and storage_class fields. Borrowed static and stack data is left
        ! alone. The class is reset so a repeated release frees nothing.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_addr
        type(lr_operand_desc_t), intent(in) :: storage_addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: storage_class, is_owned, data_ptr, victim

        call set_empty(error_msg)
        if (.not. emit_i32_load(context%session, storage_addr, storage_class, &
                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, storage_class, &
                i32_immediate(context%session, LOWERING_CHARACTER_STORAGE_OWNED), &
                is_owned, error_msg)) return
        if (.not. emit_ptr_load(context%session, data_addr, data_ptr, &
                error_msg)) return
        ! free(NULL) is a no-op, so a borrowed descriptor selects the null
        ! pointer and the call has no effect.
        call select_value(context, is_owned, data_ptr, &
                          null_ptr_operand(context), victim, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_free(context%session, victim, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, 0_c_int64_t), storage_addr, &
                error_msg)) return
    end subroutine release_character_descriptor_storage

    subroutine prepare_deferred_char_actual(arena, node_index, context, arg, &
                                            error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: arg
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_addr
        type(lr_operand_desc_t) :: len_addr
        type(lr_operand_desc_t) :: data_value
        type(lr_operand_desc_t) :: len_value
        character(len=:), allocatable :: literal_text
        character(len=:), allocatable :: lit_value, lit_type
        integer :: symbol_index

        call allocate_deferred_char_descriptor(context, arg, data_addr, &
                                               len_addr, error_msg)
        if (len_trim(error_msg) > 0) return

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call strip_literal_quotes(lit_value, literal_text)
            call materialize_concat_literal(context, literal_text, data_value, &
                                            len_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_store(context%session, data_value, data_addr, &
                error_msg)) return
            if (.not. emit_i64_store(context%session, len_value, len_addr, &
                error_msg)) return
            call set_empty(error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, lit_value, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, lit_value)
            if (symbol_index <= 0 .or. &
                context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) then
                error_msg = 'character argument is not declared: '//trim(lit_value)
                return
            end if
            if (context%symbols(symbol_index)%is_deferred_character) then
                if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%deferred_data, data_value, &
                    error_msg)) return
                if (.not. emit_i64_load(context%session, &
                    context%symbols(symbol_index)%deferred_length, len_value, &
                    error_msg)) return
            else
                data_value = context%symbols(symbol_index)%value
                len_value = i64_immediate(context%session, &
                    int(context%symbols(symbol_index)%character_length, c_int64_t))
            end if
        else
            error_msg = 'direct LIRIC session cannot pass this character argument'
            return
        end if

        if (.not. emit_ptr_store(context%session, data_value, data_addr, &
            error_msg)) return
        if (.not. emit_i64_store(context%session, len_value, len_addr, &
            error_msg)) return
        call set_empty(error_msg)
    end subroutine prepare_deferred_char_actual

    subroutine prepare_deferred_char_call_args(arena, arg_indices, context, &
                                               args, copyback_indices, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        integer, allocatable, intent(out) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: handled
        integer :: i
        integer :: symbol_index
        character(len=:), allocatable :: id_name

        allocate (args(size(arg_indices)))
        allocate (copyback_indices(size(arg_indices)))
        copyback_indices = 0

        do i = 1, size(arg_indices)
            handled = .false.
            if (is_literal(arena, arg_indices(i))) then
                if (is_character_literal(arena, arg_indices(i))) then
                    call prepare_deferred_char_actual(arena, arg_indices(i), &
                        context, args(i), error_msg)
                    handled = .true.
                end if
            else if (is_identifier(arena, arg_indices(i))) then
                call get_identifier_name(arena, arg_indices(i), id_name, error_msg)
                if (len_trim(error_msg) > 0) return
                symbol_index = find_symbol_compat(context, id_name)
                if (symbol_index > 0 .and. &
                    context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    call prepare_deferred_char_actual(arena, arg_indices(i), &
                        context, args(i), error_msg)
                    handled = .true.
                end if
            end if
            if (len_trim(error_msg) > 0) return
            if (.not. handled) then
                call prepare_reference_arg(arena, arg_indices(i), context, &
                                           VALUE_I32, args(i), &
                                           copyback_indices(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        end do
        call set_empty(error_msg)
    end subroutine prepare_deferred_char_call_args

    subroutine lower_deferred_char_result_call(arena, node, context, &
                                                dest_symbol_index, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(call_or_subscript_node) :: call_node
        type(lr_operand_desc_t), allocatable :: user_args(:)
        type(lr_operand_desc_t), allocatable :: all_args(:)
        integer, allocatable :: copyback_indices(:)
        type(lr_operand_desc_t) :: descriptor_ptr
        type(lr_operand_desc_t) :: data_addr
        type(lr_operand_desc_t) :: len_addr
        type(lr_operand_desc_t) :: capacity_addr
        type(lr_operand_desc_t) :: storage_addr
        integer :: i
        integer :: temp_mark
        character(len=:), allocatable :: func_name

        call set_empty(error_msg)
        temp_mark = char_temp_mark(context)
        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'deferred char call assignment value does not reference an AST node'
            return
        end if
        select type (value_node => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            call_node = value_node
        class default
            error_msg = 'expected function call for deferred char result'
            return
        end select

        func_name = call_emit_name(arena, call_node%name, context)

        ! The callee writes its result through a fresh return descriptor, so
        ! the destination stops using its own. Release whatever heap block that
        ! descriptor still owned first, then rebind the destination onto the
        ! return descriptor: returning the descriptor transfers ownership, so
        ! the destination keeps its ownership slots and a later assignment or
        ! deallocate releases the adopted block.
        call release_owned_character_storage(context, dest_symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return

        temp_mark = char_temp_mark(context)
        call allocate_deferred_char_descriptor(context, descriptor_ptr, &
                                               data_addr, len_addr, error_msg, &
                                               capacity_addr, storage_addr)
        if (len_trim(error_msg) > 0) return
        context%symbols(dest_symbol_index)%deferred_data = data_addr
        context%symbols(dest_symbol_index)%deferred_length = len_addr
        context%symbols(dest_symbol_index)%deferred_capacity = capacity_addr
        context%symbols(dest_symbol_index)%deferred_storage = storage_addr
        context%symbols(dest_symbol_index)%has_character_ownership = .true.

        if (allocated(call_node%arg_indices)) then
            call prepare_deferred_char_call_args(arena, call_node%arg_indices, &
                context, user_args, copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (user_args(0))
            allocate (copyback_indices(0))
        end if

        allocate (all_args(size(user_args) + 1))
        all_args(1) = descriptor_ptr
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do

        if (.not. emit_void_call(context%session, trim(func_name), all_args, &
                                 error_msg)) return
        call copy_back_reference_args(context, user_args, copyback_indices, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_ptr_load(context%session, data_addr, &
            context%symbols(dest_symbol_index)%value, error_msg)) return
        context%symbols(dest_symbol_index)%has_character_value = .true.
        ! The destination now owns this descriptor, so the statement must not
        ! release it: that is the ownership transfer #348 established. Any
        ! temporary produced by the argument expressions stays registered.
        call adopt_char_temp(context, temp_mark)
        call set_empty(error_msg)
    end subroutine lower_deferred_char_result_call

    ! A fixed-length scalar destination receiving a character-returning
    ! contained function. The callee returns its result through the canonical
    ! character descriptor passed as the leading sret pointer. The destination
    ! takes a blank-padded or truncated copy at its own declared width, never
    ! an alias of the returned buffer, so the temporary can then be released.
    subroutine lower_fixed_char_result_call(arena, node, context, &
                                            dest_symbol_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, length

        call char_result_call_operands(arena, node%value_index, context, &
                                       data_ptr, length, error_msg)
        if (len_trim(error_msg) > 0) return
        call store_fixed_character_value(context, dest_symbol_index, data_ptr, &
                                         length, error_msg)
    end subroutine lower_fixed_char_result_call

    subroutine char_result_call_operands(arena, node_index, context, data_ptr, &
                                         length, error_msg)
        ! Emit a call to a character-returning contained function through the
        ! deferred-character descriptor ABI (hidden first descriptor argument),
        ! then load the returned {data, len} as a scalar character expression.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        type(call_or_subscript_node) :: call_node
        type(lr_operand_desc_t), allocatable :: user_args(:)
        type(lr_operand_desc_t), allocatable :: all_args(:)
        integer, allocatable :: copyback_indices(:)
        type(lr_operand_desc_t) :: descriptor_ptr
        type(lr_operand_desc_t) :: data_addr
        type(lr_operand_desc_t) :: len_addr
        type(lr_operand_desc_t) :: storage_addr
        integer :: i
        character(len=:), allocatable :: func_name

        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'character call does not reference an AST node'
            return
        end if
        select type (value_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            call_node = value_node
        class default
            error_msg = 'expected function call for character result'
            return
        end select
        func_name = call_emit_name(arena, call_node%name, context)

        call allocate_deferred_char_descriptor(context, descriptor_ptr, &
                                               data_addr, len_addr, error_msg, &
                                               storage_addr=storage_addr)
        if (len_trim(error_msg) > 0) return

        if (allocated(call_node%arg_indices)) then
            call prepare_deferred_char_call_args(arena, call_node%arg_indices, &
                context, user_args, copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (user_args(0))
            allocate (copyback_indices(0))
        end if

        allocate (all_args(size(user_args) + 1))
        all_args(1) = descriptor_ptr
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do

        if (.not. emit_void_call(context%session, trim(func_name), all_args, &
                                 error_msg)) return
        call copy_back_reference_args(context, user_args, copyback_indices, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_ptr_load(context%session, data_addr, data_ptr, &
                                error_msg)) return
        if (.not. emit_i32_load(context%session, len_addr, length, &
                                error_msg)) return
        ! The result is a temporary owned by the statement being lowered, which
        ! releases it once it has finished with it. Every consumer therefore
        ! reads it the same way and none of them has to know it is a temporary.
        call register_char_temp(context, data_addr, storage_addr)
        call set_empty(error_msg)
    end subroutine char_result_call_operands

    subroutine lower_print_deferred_char_call(arena, node, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: user_args(:)
        type(lr_operand_desc_t), allocatable :: all_args(:)
        integer, allocatable :: copyback_indices(:)
        type(lr_operand_desc_t) :: descriptor_ptr
        type(lr_operand_desc_t) :: data_addr
        type(lr_operand_desc_t) :: len_addr
        type(lr_operand_desc_t) :: storage_addr
        type(lr_operand_desc_t) :: data_value
        integer :: i
        character(len=:), allocatable :: func_name

        call set_empty(error_msg)

        func_name = call_emit_name(arena, node%name, context)

        call allocate_deferred_char_descriptor(context, descriptor_ptr, &
                                               data_addr, len_addr, error_msg, &
                                               storage_addr=storage_addr)
        if (len_trim(error_msg) > 0) return
        ! The printed result is a temporary the statement owns and releases.
        call register_char_temp(context, data_addr, storage_addr)

        if (allocated(node%arg_indices)) then
            call prepare_deferred_char_call_args(arena, node%arg_indices, &
                context, user_args, copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (user_args(0))
            allocate (copyback_indices(0))
        end if

        allocate (all_args(size(user_args) + 1))
        all_args(1) = descriptor_ptr
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do

        if (.not. emit_void_call(context%session, trim(func_name), all_args, &
                                 error_msg)) return
        call copy_back_reference_args(context, user_args, copyback_indices, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_ptr_load(context%session, data_addr, data_value, &
            error_msg)) return
        ! The list-directed leading blank is emitted by the print loop.
        if (.not. emit_liric_print_string_operand_value(context%session, &
            context%str_print_format_id, data_value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_print_deferred_char_call

    ! Flatten a // operand tree into its leaf character operands, resolving each
    ! leaf to (data pointer, i32 length) via char_expr_operands. A leaf is any
    ! node that is not itself a // binary operator: a literal, a character
    ! variable, or a character intrinsic (trim, adjustl, ...). stat_len carries
    ! each leaf's compile-time length, or -1 when only known at runtime (trim),
    ! so the caller can statically clamp to the fixed target length.
    recursive subroutine collect_concat_operands(arena, node_index, context, &
                                                 data_ops, len_ops, stat_len, &
                                                 count, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), allocatable, intent(inout) :: data_ops(:)
        type(lr_operand_desc_t), allocatable, intent(inout) :: len_ops(:)
        integer, allocatable, intent(inout) :: stat_len(:)
        integer, intent(inout) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: bin_op

        call set_empty(error_msg)
        if (is_binary_op(arena, node_index)) then
            block
                integer :: left_idx, right_idx, bin_line, bin_col
                call get_binary_op_info(arena, node_index, bin_op, left_idx, &
                                        right_idx, bin_line, bin_col, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(bin_op) == '//') then
                    call collect_concat_operands(arena, left_idx, context, &
                        data_ops, len_ops, stat_len, count, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call collect_concat_operands(arena, right_idx, context, &
                        data_ops, len_ops, stat_len, count, error_msg)
                    return
                end if
            end block
        end if
        count = count + 1
        stat_len(count) = concat_leaf_static_len(arena, node_index, context)
        call char_expr_operands(arena, node_index, context, data_ops(count), &
                                len_ops(count), error_msg)
    end subroutine collect_concat_operands

    ! Compile-time length of a // leaf, or -1 when only known at runtime. A
    ! literal contributes its text length; a fixed-length character variable its
    ! declared length. trim() and deferred-length operands are runtime-only.
    integer function concat_leaf_static_len(arena, node_index, context) &
            result(static_len)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: lit_value, lit_type, id_name
        character(len=:), allocatable :: stripped, err
        integer :: symbol_index

        static_len = -1
        if (is_literal(arena, node_index)) then
            if (.not. is_character_literal(arena, node_index)) return
            call get_literal_info(arena, node_index, lit_value, lit_type, err)
            if (len_trim(err) > 0) return
            call strip_literal_quotes(lit_value, stripped)
            static_len = len(stripped)
        else if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, err)
            if (len_trim(err) > 0) return
            symbol_index = find_symbol_compat(context, id_name)
            if (symbol_index <= 0) return
            if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER &
                .and. .not. context%symbols(symbol_index)%is_deferred_character) &
                static_len = context%symbols(symbol_index)%character_length
        end if
    end function concat_leaf_static_len

    ! Assign a // concatenation to a fixed-length character target. The result
    ! buffer is exactly the target's declared length, blank-padded then
    ! overwritten left to right, so truncation and padding match gfortran's
    ! fixed-length assignment. Each leaf contributes its full declared length;
    ! FortFront's lazy standardization usually sizes the target to the exact
    ! total. Operands of compile-time-known length are clamped statically so an
    ! over-long chain truncates without runtime min logic; trailing operands that
    ! fall past the target are skipped entirely.
    subroutine lower_fixed_concat_assignment(arena, node, context, &
                                             dest_symbol_index, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: data_ops(:), len_ops(:)
        integer, allocatable :: stat_len(:)
        type(lr_operand_desc_t) :: buf, offset, null_ptr
        character(len=64) :: null_name
        integer :: count, dest_len

        call set_empty(error_msg)
        dest_len = context%symbols(dest_symbol_index)%character_length
        allocate (data_ops(64), len_ops(64), stat_len(64))
        count = 0
        call collect_concat_operands(arena, node%value_index, context, data_ops, &
                                     len_ops, stat_len, count, error_msg)
        if (len_trim(error_msg) > 0) return
        if (count == 0) then
            error_msg = 'character concatenation has no operands'
            return
        end if

        ! Buffer holds dest_len content bytes plus a trailing null for %s prints.
        if (.not. emit_alloca_bytes(context%session, &
            i64_immediate(context%session, int(dest_len + 1, c_int64_t)), buf, &
            error_msg)) return
        call fill_spaces(context, buf, &
            i32_immediate(context%session, int(dest_len, c_int64_t)), error_msg)
        if (len_trim(error_msg) > 0) return

        call emit_concat_copies(context, data_ops, len_ops, stat_len, count, &
                                dest_len, buf, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Null-terminate at the fixed end so the print path stops after dest_len.
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, &
            i64_immediate(context%session, int(dest_len, c_int64_t)), offset, &
            error_msg)) return
        context%string_literal_count = context%string_literal_count + 1
        null_name = ffc_unit_global_name( &
            context, 'fcat.term.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(null_name), '', &
                                      null_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_memcpy(context%session, offset, null_ptr, &
            i64_immediate(context%session, 1_c_int64_t), error_msg)) return

        context%symbols(dest_symbol_index)%value = buf
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_fixed_concat_assignment

    ! Copy each leaf operand into the blank-filled buffer, left to right. With
    ! all lengths known, clamp to the target's end so an over-long chain
    ! truncates and a short one keeps the blank padding. With a runtime-length
    ! operand present (trim), the lazy target is sized to fit, so copy in full.
    subroutine emit_concat_copies(context, data_ops, len_ops, stat_len, count, &
                                  dest_len, buf, error_msg)
        ! Copy each operand into buf end to end. dest_len is the destination's
        ! declared width, which a fixed-length destination clamps to; a
        ! deferred-length destination passes CONCAT_NO_CLAMP because it takes
        ! the whole concatenation whatever its length.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ops(:), len_ops(:)
        integer, intent(in) :: stat_len(:)
        integer, intent(in) :: count, dest_len
        type(lr_operand_desc_t), intent(in) :: buf
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: offset, next_offset, copy_len
        integer :: i, written, room, copy_n
        logical :: all_static

        call set_empty(error_msg)
        all_static = all(stat_len(1:count) >= 0) .and. dest_len /= CONCAT_NO_CLAMP
        offset = buf
        written = 0
        do i = 1, count
            if (all_static) then
                room = dest_len - written
                if (room <= 0) exit
                copy_n = min(stat_len(i), room)
                if (copy_n <= 0) cycle
                copy_len = i64_immediate(context%session, int(copy_n, c_int64_t))
                written = written + copy_n
            else if (stat_len(i) >= 0) then
                copy_len = i64_immediate(context%session, &
                                         int(stat_len(i), c_int64_t))
            else
                if (.not. emit_liric_i32_to_i64(context%session, len_ops(i), &
                    copy_len, error_msg)) return
            end if
            if (.not. emit_memcpy(context%session, offset, data_ops(i), copy_len, &
                error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, offset, &
                copy_len, next_offset, error_msg)) return
            offset = next_offset
        end do
    end subroutine emit_concat_copies

    subroutine load_alloc_char_descriptor(context, slot_address, data_ptr, &
                                          length, error_msg)
        ! (data pointer, i32 length) of the canonical character descriptor a
        ! deferred-length allocatable component holds inline: pointer at +0 and
        ! i64 length at +8. An unallocated component reads as (null, 0).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: slot_address
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: length64

        if (.not. emit_ptr_load(context%session, slot_address, data_ptr, &
                                error_msg)) return
        if (.not. emit_i64_load_at(context%session, slot_address, 8_c_int64_t, &
                                   length64, error_msg)) return
        if (.not. emit_liric_i64_to_i32(context%session, length64, length, &
                                        error_msg)) return
        call set_empty(error_msg)
    end subroutine load_alloc_char_descriptor

    subroutine store_alloc_char_component(arena, value_index, slot_address, &
                                          context, error_msg)
        ! obj%s = <character expression> for a deferred-length allocatable
        ! component: allocate len+1 bytes, deep-copy the right-hand side bytes,
        ! NUL-terminate for the print/compare paths, then free whatever the
        ! component owned before and publish the new descriptor. The copy
        ! precedes the free, so a self-referential right-hand side stays valid,
        ! and each owned buffer is released exactly once.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lr_operand_desc_t), intent(in) :: slot_address
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: src_data, src_len, src_len64, total
        type(lr_operand_desc_t) :: new_ptr, term_addr, old_ptr

        call char_expr_operands(arena, value_index, context, src_data, src_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, src_len, src_len64, &
                                        error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, src_len64, &
                i64_immediate(context%session, 1_c_int64_t), total, error_msg)) &
            return
        if (.not. emit_malloc(context%session, total, new_ptr, error_msg)) return
        if (.not. emit_memcpy(context%session, new_ptr, src_data, src_len64, &
                              error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, new_ptr, src_len64, &
                                  term_addr, error_msg)) return
        if (.not. emit_i8_store(context%session, &
                i8_immediate(context%session, 0_c_int64_t), term_addr, &
                error_msg)) return
        if (.not. emit_ptr_load(context%session, slot_address, old_ptr, &
                                error_msg)) return
        if (.not. emit_free(context%session, old_ptr, error_msg)) return
        if (.not. emit_ptr_store(context%session, new_ptr, slot_address, &
                                 error_msg)) return
        if (.not. emit_i64_store_at(context%session, src_len64, slot_address, &
                                    8_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine store_alloc_char_component

    subroutine reown_alloc_char_component(context, slot_address, error_msg)
        ! After a whole-derived slot copy the destination's character descriptor
        ! still points at the source's buffer. Give the destination its own copy
        ! so the two values are independent; an unallocated descriptor (null
        ! pointer) is left alone and stays unallocated.
        use, intrinsic :: iso_c_binding, only: c_int32_t, c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: slot_address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: shared_ptr, length64, total, new_ptr, term_addr
        type(lr_operand_desc_t) :: is_allocated
        integer(c_int32_t) :: copy_block, done_block

        if (.not. emit_ptr_load(context%session, slot_address, shared_ptr, &
                                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, shared_ptr, &
                null_ptr_operand(context), is_allocated, error_msg)) return
        copy_block = create_liric_block(context%session)
        done_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, is_allocated, copy_block, &
                                    done_block, error_msg)) return

        if (.not. set_liric_block(context%session, copy_block, error_msg)) return
        context%current_block_id = copy_block
        context%current_block_terminated = .false.
        if (.not. emit_i64_load_at(context%session, slot_address, 8_c_int64_t, &
                                   length64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, length64, &
                i64_immediate(context%session, 1_c_int64_t), total, error_msg)) &
            return
        if (.not. emit_malloc(context%session, total, new_ptr, error_msg)) return
        if (.not. emit_memcpy(context%session, new_ptr, shared_ptr, length64, &
                              error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, new_ptr, length64, &
                                  term_addr, error_msg)) return
        if (.not. emit_i8_store(context%session, &
                i8_immediate(context%session, 0_c_int64_t), term_addr, &
                error_msg)) return
        if (.not. emit_ptr_store(context%session, new_ptr, slot_address, &
                                 error_msg)) return
        if (.not. emit_liric_br(context%session, done_block, error_msg)) return

        if (.not. set_liric_block(context%session, done_block, error_msg)) return
        context%current_block_id = done_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine reown_alloc_char_component
