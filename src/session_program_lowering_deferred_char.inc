integer function resolve_concat_operand(arena, node_index, context, &
                                           out_name, is_literal, error_msg) &
      result(symbol_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: out_name
        logical, intent(out) :: is_literal
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: idx

        symbol_index = 0
        is_literal = .false.
        out_name = ''
        call set_empty(error_msg)

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'concat operand index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            name = node%name
            idx = find_symbol(context, trim(name))
            if (idx > 0) then
                symbol_index = idx
                out_name = trim(name)
            else
                error_msg = 'concat operand not declared: '//trim(name)
            end if
        type is (literal_node)
            is_literal = .true.
            out_name = node%value
        class default
            error_msg = 'expected identifier or literal in concat operand'
        end select
    end function resolve_concat_operand

 subroutine lower_deferred_concat_assignment(arena, node, context, &
                                                  dest_symbol_index, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(binary_op_node) :: binop
        integer :: left_idx
        integer :: right_idx
        type(lr_operand_desc_t) :: left_data
        type(lr_operand_desc_t) :: left_len
        type(lr_operand_desc_t) :: right_data
        type(lr_operand_desc_t) :: right_len
        type(lr_operand_desc_t) :: total_len
        type(lr_operand_desc_t) :: new_ptr
        type(lr_operand_desc_t) :: zero_i64
        type(lr_operand_desc_t) :: one_i64
        type(lr_operand_desc_t) :: running_offset
        type(lr_operand_desc_t) :: old_data
        type(lr_operand_desc_t) :: old_len_val
        character(len=:), allocatable :: left_name
        character(len=:), allocatable :: right_name
        character(len=:), allocatable :: left_text
        character(len=:), allocatable :: right_text
        integer :: left_sym_idx
        integer :: right_sym_idx
        logical :: is_self_alias
        logical :: left_is_literal
        logical :: right_is_literal
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        character(len=:), allocatable :: combined_text
        integer(c_int64_t) :: combined_len

        call set_empty(error_msg)
        left_is_literal = .false.
        right_is_literal = .false.
        is_self_alias = .false.

        ! Get the binary_op_node from the assignment value
        if (.not. arena%has_node_at(node%value_index)) then
            error_msg = 'concat assignment value does not reference an AST node'
            return
        end if
        select type (binop => arena%entries(node%value_index)%node)
        type is (binary_op_node)
            if (trim(binop%operator) /= '//') then
                error_msg = 'non-concat operator on deferred character: '// &
                            trim(binop%operator)
                return
            end if
            left_idx = binop%left_index
            right_idx = binop%right_index
        class default
            error_msg = 'expected binary // operator for deferred character '// &
                        'concatenation'
            return
        end select

        ! Resolve left operand: get symbol index or mark as literal
        left_sym_idx = resolve_concat_operand(arena, left_idx, context, &
            left_name, left_is_literal, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Resolve right operand
        right_sym_idx = resolve_concat_operand(arena, right_idx, context, &
            right_name, right_is_literal, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Detect self-aliasing: dest is one of the operands
        if (left_sym_idx == dest_symbol_index .or. &
            right_sym_idx == dest_symbol_index) then
            is_self_alias = .true.
        end if

        ! For self-aliasing, we need to capture the old content
        ! We do this by loading the old data pointer and length
        if (is_self_alias) then
            if (.not. context%session%emit_i64_load( &
                context%symbols(dest_symbol_index)%deferred_data, &
                old_data, error_msg)) return
            if (.not. context%session%emit_i64_load( &
                context%symbols(dest_symbol_index)%deferred_length, &
                old_len_val, error_msg)) return
        end if

        ! Get left operand text/content
        if (left_is_literal) then
            call strip_literal_quotes(left_name, left_text)
            call materialize_concat_literal(context, left_text, left_data, &
                left_len, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            ! For self-aliasing, use captured old values
            ! For non-self-aliasing, load from symbol
            if (is_self_alias .and. left_sym_idx == dest_symbol_index) then
                left_data = old_data
                left_len = old_len_val
            else
                if (.not. context%session%emit_i64_load( &
                    context%symbols(left_sym_idx)%deferred_data, &
                    left_data, error_msg)) return
                if (.not. context%session%emit_i64_load( &
                    context%symbols(left_sym_idx)%deferred_length, &
                    left_len, error_msg)) return
            end if
        end if

        ! Get right operand text/content
        if (right_is_literal) then
            call strip_literal_quotes(right_name, right_text)
            call materialize_concat_literal(context, right_text, right_data, &
                right_len, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            if (is_self_alias .and. right_sym_idx == dest_symbol_index) then
                right_data = old_data
                right_len = old_len_val
            else
                if (.not. context%session%emit_i64_load( &
                    context%symbols(right_sym_idx)%deferred_data, &
                    right_data, error_msg)) return
                if (.not. context%session%emit_i64_load( &
                    context%symbols(right_sym_idx)%deferred_length, &
                    right_len, error_msg)) return
            end if
        end if

        ! Compute total length
        if (left_is_literal .and. right_is_literal) then
            combined_text = trim(left_text) // trim(right_text)
            combined_len = int(len(combined_text), c_int64_t)
        else
            ! Runtime concatenation: compute lengths and add
            if (left_is_literal) then
                left_len = context%session%i64_immediate( &
                    int(len(left_text), c_int64_t))
            end if
            if (right_is_literal) then
                right_len = context%session%i64_immediate( &
                    int(len(right_text), c_int64_t))
            end if
            if (.not. context%session%emit_i64_binary( &
                LR_OP_ADD, left_len, right_len, total_len, error_msg)) return
        end if

        ! For literal-only concat, create a new global string
        if (left_is_literal .and. right_is_literal) then
            context%string_literal_count = context%string_literal_count + 1
            write (string_name, '(A,I0)') '.ffc.cat.', &
                context%string_literal_count
            call materialize_liric_string(context%session, trim(string_name), &
                                          combined_text, new_ptr, error_msg)
            if (len_trim(error_msg) > 0) return

            ! Store pointer and length in descriptor
            if (.not. context%session%emit_ptr_store( &
                new_ptr, &
                context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
            if (.not. context%session%emit_i64_store( &
                context%session%i64_immediate(combined_len), &
                context%symbols(dest_symbol_index)%deferred_length, error_msg)) return

            context%symbols(dest_symbol_index)%value = new_ptr
            context%symbols(dest_symbol_index)%has_character_value = .true.
            call set_empty(error_msg)
            return
        end if

        ! Runtime concatenation: alloca (total_len + 1) for trailing null,
        ! memcpy operands, then write null byte so %s printers stay bounded.
        block
            type(lr_operand_desc_t) :: alloca_size
            type(lr_operand_desc_t) :: null_byte_ptr
            type(lr_operand_desc_t) :: null_dest
            character(len=64) :: null_name

            if (.not. context%session%emit_i64_binary( &
                LR_OP_ADD, total_len, &
                context%session%i64_immediate(1_c_int64_t), &
                alloca_size, error_msg)) return

            if (.not. context%session%emit_alloca_bytes(alloca_size, new_ptr, &
                error_msg)) return

            if (.not. context%session%emit_memcpy( &
                new_ptr, left_data, left_len, error_msg)) return

            if (.not. context%session%emit_i64_binary( &
                LR_OP_ADD, new_ptr, left_len, running_offset, error_msg)) return

            if (.not. context%session%emit_memcpy( &
                running_offset, right_data, right_len, error_msg)) return

            if (.not. context%session%emit_i64_binary( &
                LR_OP_ADD, running_offset, right_len, null_dest, &
                error_msg)) return

            context%string_literal_count = context%string_literal_count + 1
            write (null_name, '(A,I0)') '.ffc.cat.term.', &
                context%string_literal_count
            call materialize_liric_string(context%session, trim(null_name), &
                                          '', null_byte_ptr, error_msg)
            if (len_trim(error_msg) > 0) return

            if (.not. context%session%emit_memcpy( &
                null_dest, null_byte_ptr, &
                context%session%i64_immediate(1_c_int64_t), error_msg)) return
        end block

        if (.not. context%session%emit_i64_store( &
            new_ptr, context%symbols(dest_symbol_index)%deferred_data, &
            error_msg)) return

        if (.not. context%session%emit_i64_store( &
            total_len, context%symbols(dest_symbol_index)%deferred_length, &
            error_msg)) return

        context%symbols(dest_symbol_index)%value = new_ptr
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_concat_assignment

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
        write (string_name, '(A,I0)') '.ffc.cat.lit.', &
            context%string_literal_count
        call materialize_liric_string(context%session, trim(string_name), &
                                      text, data_op, error_msg)
        if (len_trim(error_msg) > 0) return
        len_op = context%session%i64_immediate(int(len(text), c_int64_t))
        call set_empty(error_msg)
    end subroutine materialize_concat_literal

    subroutine lower_deferred_literal_assignment(value_node, context, &
                                                 dest_symbol_index, error_msg)
        type(literal_node), intent(in) :: value_node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        type(lr_operand_desc_t) :: data_op
        type(lr_operand_desc_t) :: len_op

        call strip_literal_quotes(value_node%value, literal_text)
        call materialize_concat_literal(context, literal_text, data_op, &
                                        len_op, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. context%session%emit_ptr_store(data_op, &
            context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        if (.not. context%session%emit_i64_store(len_op, &
            context%symbols(dest_symbol_index)%deferred_length, error_msg)) return

        context%symbols(dest_symbol_index)%value = data_op
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_literal_assignment
