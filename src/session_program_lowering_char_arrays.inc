    ! Fixed-length character element arrays. Storage is one contiguous buffer of
    ! array_size * character_length bytes in element_address, so element i sits
    ! at base + i*character_length and the element stride is the declared length,
    ! exactly as docs/ARRAY_DESCRIPTOR_ABI.md describes a contiguous array of
    ! ARRAY_ELEMENT_CHARACTER with element_size = character_length.
    !
    ! An element is therefore a borrowed character view: character_length bytes
    ! at its address, with no terminator of its own, since the next element
    ! begins immediately after it. Reading one goes through the same view
    ! machinery as a substring, and anything that needs a C string materialises
    ! a terminated copy of exactly the element length.
    !
    ! Scope: local rank-1 and rank-2 arrays with a compile-time length and
    ! extent; dummy character arrays stay unsupported. A character parameter
    ! array may infer its element length from its folded constructor.

    recursive logical function is_character_array_constructor_expression( &
            arena, node_index, context) result(is_array)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: op, op_error
        integer :: left_index, right_index, line, column

        is_array = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (array_literal_node)
            is_array = .true.
        class default
            if (.not. is_binary_op(arena, node_index)) return
            call get_binary_op_info(arena, node_index, op, left_index, &
                                    right_index, line, column, op_error)
            if (len_trim(op_error) > 0 .or. trim(op) /= '//') return
            is_array = is_character_array_constructor_expression(arena, &
                left_index, context) .or. &
                is_character_array_constructor_expression(arena, right_index, context)
        end select
    end function is_character_array_constructor_expression

    recursive integer function character_array_constructor_size(arena, node_index, &
                                                                context) result(count)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: op, op_error
        integer :: left_index, right_index, line, column
        integer :: left_count, right_count
        integer, allocatable :: flat(:)

        count = 1
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (array_literal_node)
            count = 0
            if (.not. allocated(node%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, node%element_indices, flat, &
                                               op_error)
            if (len_trim(op_error) == 0) count = size(flat)
        class default
            if (.not. is_binary_op(arena, node_index)) return
            call get_binary_op_info(arena, node_index, op, left_index, &
                                    right_index, line, column, op_error)
            if (len_trim(op_error) > 0 .or. trim(op) /= '//') return
            left_count = character_array_constructor_size(arena, left_index, context)
            right_count = character_array_constructor_size(arena, right_index, context)
            if (is_character_array_constructor_expression(arena, left_index, context)) &
                count = left_count
            if (is_character_array_constructor_expression(arena, right_index, context)) &
                count = max(count, right_count)
        end select
    end function character_array_constructor_size

    recursive subroutine character_array_constructor_element_operands(arena, &
            node_index, element_index, context, data_ptr, length, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, element_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr, length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: op, op_error
        integer :: left_index, right_index, line, column
        integer :: left_count, right_count, left_selected, right_selected
        integer, allocatable :: flat(:)
        type(lr_operand_desc_t) :: right_data, right_length
        type(lr_operand_desc_t) :: combined_data, combined_length

        if (.not. is_character_array_constructor_expression(arena, node_index, &
                                                             context)) then
            call char_expr_operands(arena, node_index, context, data_ptr, length, &
                                    error_msg)
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (array_literal_node)
            allocate (flat(0))
            call flatten_constructor_elements(arena, node%element_indices, flat, &
                                               error_msg)
            if (len_trim(error_msg) > 0) return
            if (element_index < 1 .or. element_index > size(flat)) then
                error_msg = 'character array constructor element is out of range'
                return
            end if
            call char_expr_operands(arena, flat(element_index), context, data_ptr, &
                                    length, error_msg)
            return
        class default
        end select
        call get_binary_op_info(arena, node_index, op, left_index, right_index, &
                                line, column, error_msg)
        if (len_trim(error_msg) > 0) return
        left_count = character_array_constructor_size(arena, left_index, context)
        right_count = character_array_constructor_size(arena, right_index, context)
        left_selected = element_index
        if (left_count == 1) left_selected = 1
        call character_array_constructor_element_operands(arena, left_index, &
            left_selected, context, data_ptr, length, error_msg)
        if (len_trim(error_msg) > 0) return
        right_selected = element_index
        if (right_count == 1) right_selected = 1
        call character_array_constructor_element_operands(arena, right_index, &
            right_selected, context, right_data, right_length, error_msg)
        if (len_trim(error_msg) > 0) return
        call concat_character_operands(context, data_ptr, length, right_data, &
                                       right_length, combined_data, combined_length, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        data_ptr = combined_data
        length = combined_length
    end subroutine character_array_constructor_element_operands

    subroutine lower_character_array_declaration(node, context, error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: character_length
        integer :: array_lower_bound
        integer :: array_size
        integer :: i
        character(len=:), allocatable :: initial_texts(:)

        call set_empty(error_msg)
        if (node%is_allocatable) then
            call declaration_character_length(context, node, character_length, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            if (character_length <= 0) then
                call declare_deferred_char_allocatable_array(node, context, &
                                                             error_msg)
                return
            end if
            call lower_allocatable_declaration(node, context, error_msg, &
                                               VALUE_CHARACTER)
            return
        end if
        if (node%is_save) then
            call unsupported_feature_error('character array declaration', &
                node%line, node%column, 'direct LIRIC session does not support '// &
                'the save attribute on character arrays', error_msg)
            return
        end if
        if (declaration_is_assumed_rank(node, context) .or. &
            declaration_is_assumed_shape(node, context)) then
            call unsupported_feature_error('character array declaration', &
                node%line, node%column, 'direct LIRIC session does not support '// &
                'assumed-shape or assumed-rank character array dummies', error_msg)
            return
        end if
        if (character_array_is_dummy(context, node)) then
            call unsupported_feature_error('character array declaration', &
                node%line, node%column, 'direct LIRIC session does not support '// &
                'character array dummy arguments', error_msg)
            return
        end if
        call declaration_character_length(context, node, character_length, error_msg)
        if (len_trim(error_msg) > 0) return
        if (character_length <= 0) then
            if (.not. node%is_parameter .or. .not. node%has_initializer .or. &
                node%initializer_index <= 0) then
                call unsupported_feature_error('character array declaration', &
                    node%line, node%column, 'direct LIRIC session supports only a '// &
                    'compile-time character length for character arrays', error_msg)
                return
            end if
            call infer_character_array_initializer_length(context, node, &
                character_length, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call get_array_bounds(node, context, array_lower_bound, array_size, &
                              error_msg)
        if (len_trim(error_msg) > 0) return

        if (node%has_initializer .and. node%initializer_index > 0) then
            if (node%is_multi_declaration) then
                call unsupported_feature_error('character array declaration', &
                    node%line, node%column, 'direct LIRIC session does not '// &
                    'support character array initializers on multi-declarations', &
                    error_msg)
                return
            end if
            call fold_character_array_initializer(context, node, &
                character_length, array_size, initial_texts, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_character_array_symbol(context, node, &
                    node%var_names(i), character_length, array_lower_bound, &
                    array_size, initial_texts, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_character_array_symbol(context, node, node%var_name, &
                character_length, array_lower_bound, array_size, initial_texts, &
                error_msg)
        else
            error_msg = 'character array declaration did not expose a variable name'
        end if
    end subroutine lower_character_array_declaration

    ! Fold a compile-time character array initializer (plain array constructor,
    ! possibly nested, or a scalar literal broadcast) into per-element text
    ! normalized to the declared length. Any form whose leaves are not
    ! foldable character literals or // chains is declined the same way the
    ! prior path declined all initializers, so no previously-accepted case
    ! regresses. On decline, initial_texts stays unallocated and error_msg set.
    subroutine fold_character_array_initializer(context, node, character_length, &
                                                array_size, initial_texts, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: character_length
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(out) :: initial_texts(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: folded_text
        logical :: ok
        integer :: i

        call set_empty(error_msg)
        if (.not. node_exists(context%arena, node%initializer_index)) then
            call unsupported_feature_error('character array declaration', &
                node%line, node%column, 'direct LIRIC session does not support '// &
                'character array initializers', error_msg)
            return
        end if

        allocate (character(len=character_length) :: initial_texts(array_size))
        do i = 1, array_size
            initial_texts(i) = repeat(' ', character_length)
        end do

        select type (init => context%arena%entries(node%initializer_index)%node)
        type is (array_literal_node)
            if (allocated(init%syntax_style)) then
                if (init%syntax_style == 'implied_do') then
                    call decline_character_array_initializer(node, initial_texts, &
                                                             error_msg)
                    return
                end if
            end if
            if (.not. allocated(init%element_indices)) then
                call decline_character_array_initializer(node, initial_texts, &
                                                         error_msg)
                return
            end if
            allocate (flat(0))
            call flatten_constructor_elements(context%arena, init%element_indices, &
                                              flat, error_msg)
            if (len_trim(error_msg) > 0) then
                call decline_character_array_initializer(node, initial_texts, &
                                                         error_msg)
                return
            end if
            if (size(flat) /= array_size) then
                call decline_character_array_initializer(node, initial_texts, &
                                                         error_msg)
                return
            end if
            do i = 1, array_size
                call concat_character_literals(context%arena, flat(i), &
                                               folded_text, ok)
                if (.not. ok) then
                    call decline_character_array_initializer(node, initial_texts, &
                                                             error_msg)
                    return
                end if
                initial_texts(i) = folded_text
            end do
        class default
            call concat_character_literals(context%arena, &
                node%initializer_index, folded_text, ok)
            if (.not. ok) then
                call decline_character_array_initializer(node, initial_texts, &
                                                         error_msg)
                return
            end if
            do i = 1, array_size
                initial_texts(i) = folded_text
            end do
        end select
    end subroutine fold_character_array_initializer

    subroutine decline_character_array_initializer(node, initial_texts, error_msg)
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: initial_texts(:)
        character(len=:), allocatable, intent(out) :: error_msg

        if (allocated(initial_texts)) deallocate (initial_texts)
        call unsupported_feature_error('character array declaration', &
            node%line, node%column, 'direct LIRIC session does not support '// &
            'character array initializers', error_msg)
    end subroutine decline_character_array_initializer

    logical function character_array_is_dummy(context, node) result(is_dummy)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer :: index

        is_dummy = .false.
        if (.not. allocated(node%var_name)) return
        index = find_symbol_compat(context, node%var_name)
        if (index <= 0) return
        is_dummy = context%symbols(index)%is_dummy_argument
    end function character_array_is_dummy

    subroutine infer_character_array_initializer_length(context, node, length, &
                                                         error_msg)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: folded_text
        logical :: ok
        integer :: i

        length = 0
        call set_empty(error_msg)
        if (.not. node_exists(context%arena, node%initializer_index)) then
            error_msg = 'character parameter initializer does not exist'
            return
        end if
        select type (init => context%arena%entries(node%initializer_index)%node)
        type is (array_literal_node)
            if (.not. allocated(init%element_indices)) then
                error_msg = 'character parameter initializer has no elements'
                return
            end if
            allocate (flat(0))
            call flatten_constructor_elements(context%arena, init%element_indices, &
                                              flat, error_msg)
            if (len_trim(error_msg) > 0) return
            do i = 1, size(flat)
                call concat_character_literals(context%arena, flat(i), &
                                               folded_text, ok)
                if (.not. ok) then
                    error_msg = 'character parameter initializer is not foldable'
                    return
                end if
                length = max(length, len(folded_text))
            end do
        class default
            call concat_character_literals(context%arena, node%initializer_index, &
                                           folded_text, ok)
            if (.not. ok) then
                error_msg = 'character parameter initializer is not foldable'
                return
            end if
            length = len(folded_text)
        end select
        if (length <= 0) then
            error_msg = 'character parameter initializer has zero length'
        end if
    end subroutine infer_character_array_initializer_length

    subroutine define_character_array_symbol(context, node, name, character_length, &
                                             array_lower_bound, array_size, &
                                             initial_texts, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: character_length
        integer, intent(in) :: array_lower_bound
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(in) :: initial_texts(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        integer :: dim_count
        logical :: is_result_view

        ! VALUE_ARRAY_RESULT functions pre-register the result name against the
        ! hidden sret buffer before lowering the specification part. A
        ! character result reaches this path after its element type is known,
        ! so fill in the character-array metadata on that existing view rather
        ! than allocating a second array or reporting a duplicate declaration.
        index = find_symbol_compat(context, name)
        is_result_view = .false.
        if (index > 0) then
            if (context%symbols(index)%is_array_result) then
                is_result_view = .true.
            else
                error_msg = 'duplicate array declaration: '//trim(name)
                return
            end if
        else
            call grow_symbols(context)
            index = context%symbol_count + 1
        end if
        dim_count = 0
        if (allocated(node%dimension_indices)) dim_count = size(node%dimension_indices)

        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_CHARACTER
        context%symbols(index)%is_array = .true.
        context%symbols(index)%array_rank = dim_count
        context%symbols(index)%array_size = array_size
        context%symbols(index)%array_lower_bound = array_lower_bound
        context%symbols(index)%character_length = character_length

        call character_array_fold_dims(context, node, index, dim_count, error_msg)
        if (len_trim(error_msg) > 0) return

        if (is_result_view) then
            ! element_address already points at the caller-owned sret buffer.
            ! The result declaration supplies shape and character width only;
            ! it must not blank-fill or allocate that buffer.
            call set_empty(error_msg)
            return
        end if

        ! One contiguous block: array_size elements of character_length bytes.
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(array_size, c_int64_t)*int(character_length, c_int64_t)), &
                context%symbols(index)%element_address, error_msg)) then
            error_msg = 'decl_char_array['//trim(name)//']: '//error_msg
            return
        end if
        context%symbol_count = index

        call init_character_array_blanks(context, index, character_length, &
                                         array_size, error_msg)
        if (len_trim(error_msg) > 0) return
        if (allocated(initial_texts)) then
            call store_character_array_initials(context, index, character_length, &
                                                array_size, initial_texts, error_msg)
        end if
    end subroutine define_character_array_symbol

    ! Store folded per-element initializer text into the array's pointer slots,
    ! normalized (padded/truncated) to the declared element length.
    subroutine store_character_array_initials(context, symbol_index, &
                                              character_length, array_size, &
                                              initial_texts, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: character_length
        integer, intent(in) :: array_size
        character(len=*), intent(in) :: initial_texts(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: element_text
        character(len=64) :: string_name
        type(lr_operand_desc_t) :: index_value
        type(lr_operand_desc_t) :: slot_addr
        type(lr_operand_desc_t) :: string_ptr
        integer :: i

        call set_empty(error_msg)
        do i = 1, array_size
            element_text = initial_texts(i)
            call normalize_character_literal(element_text, character_length)
            context%string_literal_count = context%string_literal_count + 1
            string_name = ffc_unit_global_name( &
                context, 'char.', context%string_literal_count)
            call materialize_liric_string(context%session, trim(string_name), &
                                          element_text, string_ptr, error_msg)
            if (len_trim(error_msg) > 0) return
            index_value = i32_immediate(context%session, int(i - 1, c_int64_t))
            call character_array_element_address(context, symbol_index, &
                                                 index_value, slot_addr, error_msg)
            if (len_trim(error_msg) > 0) return
            ! normalize_character_literal already padded or truncated the text
            ! to the declared length, so this fills the element exactly.
            if (.not. emit_memcpy(context%session, slot_addr, string_ptr, &
                    i64_immediate(context%session, &
                        int(character_length, c_int64_t)), error_msg)) return
        end do
    end subroutine store_character_array_initials

    subroutine character_array_fold_dims(context, node, index, dim_count, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: index
        integer, intent(in) :: dim_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: dim_index
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value

        call set_empty(error_msg)
        if (.not. allocated(node%dimension_indices)) return
        do i = 1, dim_count
            dim_index = node%dimension_indices(i)
            if (.not. node_exists(context%arena, dim_index)) cycle
            select type (dim_node => context%arena%entries(dim_index)%node)
            type is (range_expression_node)
                call eval_i32_constant(context%arena, dim_node%start_index, &
                                       context, lower_value, error_msg)
                if (len_trim(error_msg) > 0) return
                call eval_i32_constant(context%arena, dim_node%end_index, &
                                       context, upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                context%symbols(index)%array_dim_lowers(i) = int(lower_value)
                context%symbols(index)%array_dim_sizes(i) = &
                    int(upper_value - lower_value + 1_c_int64_t)
            class default
                call eval_i32_constant(context%arena, dim_index, context, &
                                       upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                context%symbols(index)%array_dim_lowers(i) = 1
                context%symbols(index)%array_dim_sizes(i) = int(upper_value)
            end select
        end do
    end subroutine character_array_fold_dims

    subroutine init_character_array_blanks(context, symbol_index, character_length, &
                                           array_size, error_msg)
        ! Every element starts blank, which for a contiguous block is one fill
        ! of the whole buffer rather than a store per element.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: character_length
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (array_size <= 0 .or. character_length <= 0) return
        call fill_spaces(context, context%symbols(symbol_index)%element_address, &
                         i32_immediate(context%session, &
                             int(array_size, c_int64_t)* &
                             int(character_length, c_int64_t)), error_msg)
    end subroutine init_character_array_blanks

    subroutine character_array_element_address(context, symbol_index, &
                                               linear_index, address, error_msg)
        ! Address of element `linear_index` (0-based) in the contiguous block:
        ! base + linear_index * character_length. This is the whole of the
        ! layout contract, so every reader and writer of element storage goes
        ! through here.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: index64, offset

        if (.not. emit_liric_i32_to_i64(context%session, linear_index, index64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, index64, &
                i64_immediate(context%session, &
                    int(context%symbols(symbol_index)%character_length, &
                        c_int64_t)), offset, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, &
                context%symbols(symbol_index)%element_address, offset, address, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine character_array_element_address

    subroutine lower_character_array_element_slot(arena, node, context, &
                                                  symbol_index, slot_addr, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: slot_addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: linear_index
        integer :: rank
        integer :: arg_count

        if (.not. allocated(node%name)) then
            error_msg = 'character array access requires an array name'
            return
        end if
        symbol_index = find_symbol_compat(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'character array was not declared: '//trim(node%name)
            return
        end if
        rank = context%symbols(symbol_index)%array_rank
        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
        if (arg_count /= rank) then
            call unsupported_feature_error('character array access', node%line, &
                node%column, 'character array access requires one subscript per '// &
                'dimension', error_msg)
            return
        end if

        call character_array_linear_index(arena, node, context, symbol_index, &
                                          linear_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call character_array_element_address(context, symbol_index, linear_index, &
                                             slot_addr, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'char_arr_slot_gep: '//error_msg
            return
        end if
    end subroutine lower_character_array_element_slot

    subroutine character_array_linear_index(arena, node, context, symbol_index, &
                                            linear_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: linear_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: i_value, j_value
        type(lr_operand_desc_t) :: lower_bound_value
        type(lr_operand_desc_t) :: i_offset, j_offset
        type(lr_operand_desc_t) :: scaled_index

        call validate_i32_array_subscript(arena, node%arg_indices(1), context, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(1), context, i_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (context%symbols(symbol_index)%array_rank == 1) then
            lower_bound_value = i32_immediate(context%session, &
                int(context%symbols(symbol_index)%array_lower_bound, c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, i_value, &
                    lower_bound_value, linear_index, error_msg)) then
                error_msg = 'char_arr_sub: '//error_msg
                return
            end if
            return
        end if

        call validate_i32_array_subscript(arena, node%arg_indices(2), context, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, j_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        lower_bound_value = i32_immediate(context%session, &
            int(context%symbols(symbol_index)%array_dim_lowers(1), c_int64_t))
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, i_value, &
                lower_bound_value, i_offset, error_msg)) return
        lower_bound_value = i32_immediate(context%session, &
            int(context%symbols(symbol_index)%array_dim_lowers(2), c_int64_t))
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, j_value, &
                lower_bound_value, j_offset, error_msg)) return
        lower_bound_value = i32_immediate(context%session, &
            int(context%symbols(symbol_index)%array_dim_sizes(1), c_int64_t))
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, j_offset, &
                lower_bound_value, scaled_index, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_offset, &
                scaled_index, linear_index, error_msg)) return
    end subroutine character_array_linear_index

    subroutine lower_character_array_element_assignment(arena, node, target, &
                                                        context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        type(lr_operand_desc_t) :: slot_addr
        type(lr_operand_desc_t) :: source_data, source_length
        type(lr_operand_desc_t) :: source_copy, target_length
        type(lr_operand_desc_t) :: copy_length, copy_length64, source_fits

        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'character array assignment value does not exist'
            return
        end if
        call lower_character_array_element_slot(arena, target, context, &
                                                symbol_index, slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Evaluate the source before blank-filling the target. This preserves
        ! Fortran's evaluate-before-store rule for aliases such as a(i)=a(j).
        call char_expr_operands(arena, node%value_index, context, source_data, &
                                source_length, error_msg)
        if (len_trim(error_msg) > 0) return
        call materialize_character_view(context, source_data, source_length, &
                                        source_copy, error_msg)
        if (len_trim(error_msg) > 0) return

        target_length = i32_immediate(context%session, &
            int(context%symbols(symbol_index)%character_length, c_int64_t))
        call fill_spaces(context, slot_addr, target_length, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, source_length, &
                target_length, source_fits, error_msg)) return
        call select_value(context, source_fits, source_length, target_length, &
                          copy_length, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, copy_length, &
                copy_length64, error_msg)) return
        if (.not. emit_memcpy(context%session, slot_addr, source_copy, &
                copy_length64, error_msg)) then
            error_msg = 'char_arr_assign_store: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_character_array_element_assignment

    subroutine lower_character_array_element_print(arena, node, context, &
                                                   use_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: use_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        type(lr_operand_desc_t) :: slot_addr
        type(lr_operand_desc_t) :: string_ptr

        call lower_character_array_element_slot(arena, node, context, &
                                                symbol_index, slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        ! The element carries no terminator of its own, so print from a
        ! terminated copy of exactly the declared length.
        call materialize_character_view(context, slot_addr, &
            i32_immediate(context%session, &
                int(context%symbols(symbol_index)%character_length, c_int64_t)), &
            string_ptr, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'char_arr_print_load: '//error_msg
            return
        end if
        if (use_value) then
            if (.not. emit_liric_print_string_operand_value(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        else
            if (.not. emit_liric_print_string_operand(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine lower_character_array_element_print

    subroutine emit_character_array_print_items(context, symbol_index, &
                                                suppress_leading_space, handled, &
                                                error_msg)
        ! Whole-array print of a fixed character array: one leading list-directed
        ! space at the record start (unless the preceding print item was already a
        ! character value), then every element in storage order with no separator
        ! between the consecutive character values (matching gfortran).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        logical, intent(in) :: suppress_leading_space
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: index_value
        type(lr_operand_desc_t) :: slot_addr
        type(lr_operand_desc_t) :: string_ptr
        integer(c_int64_t) :: k, n

        handled = .false.
        call set_empty(error_msg)
        n = int(context%symbols(symbol_index)%array_size, c_int64_t)
        if (n <= 0_c_int64_t) return

        handled = .true.
        if (.not. suppress_leading_space) then
            if (.not. emit_liric_print_space(context%session, error_msg)) return
        end if
        do k = 0_c_int64_t, n - 1_c_int64_t
            index_value = i32_immediate(context%session, k)
            call character_array_element_address(context, symbol_index, &
                                                 index_value, slot_addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call materialize_character_view(context, slot_addr, &
                i32_immediate(context%session, &
                    int(context%symbols(symbol_index)%character_length, &
                        c_int64_t)), string_ptr, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_string_operand_value(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        end do
    end subroutine emit_character_array_print_items

    subroutine lower_character_array_whole_assignment(arena, node, symbol_index, &
                                                      context, handled, error_msg)
        ! c = d between two fixed character arrays. Both are contiguous blocks
        ! of array_size * character_length bytes, so a conforming assignment is
        ! a single copy of that many bytes. Declines (handled stays .false.)
        ! for anything else, so the existing paths keep their behaviour.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: element_length
        type(lr_operand_desc_t) :: source_data, source_length
        type(lr_operand_desc_t) :: source_copy, target_length
        type(lr_operand_desc_t) :: copy_length, copy_length64, source_fits
        type(lr_operand_desc_t) :: element_index, slot_addr
        integer(c_int64_t) :: k, n

        handled = .false.
        call set_empty(error_msg)
        if (context%symbols(symbol_index)%is_allocatable) return
        if (context%symbols(symbol_index)%is_runtime_array) return
        element_length = context%symbols(symbol_index)%character_length
        if (element_length <= 0) return
        if (.not. node_exists(arena, node%value_index)) return

        ! A scalar character value broadcasts to every element of a fixed
        ! character array.  The contiguous representation still needs one
        ! padded/truncated copy per element; a single block copy would only
        ! initialize the first element.  An identifier naming a character
        ! array is reserved for the array-to-array path below.
        if (.not. is_identifier(arena, node%value_index)) then
            call char_expr_operands(arena, node%value_index, context, source_data, &
                                    source_length, error_msg)
            if (len_trim(error_msg) > 0) return
            call materialize_character_view(context, source_data, source_length, &
                                            source_copy, error_msg)
            if (len_trim(error_msg) > 0) return
            target_length = i32_immediate(context%session, &
                int(element_length, c_int64_t))
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, &
                    source_length, target_length, source_fits, error_msg)) return
            call select_value(context, source_fits, source_length, target_length, &
                              copy_length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_i64(context%session, copy_length, &
                    copy_length64, error_msg)) return
            n = int(context%symbols(symbol_index)%array_size, c_int64_t)
            do k = 0_c_int64_t, n - 1_c_int64_t
                element_index = i32_immediate(context%session, k)
                call character_array_element_address(context, symbol_index, &
                                                     element_index, slot_addr, &
                                                     error_msg)
                if (len_trim(error_msg) > 0) return
                call fill_spaces(context, slot_addr, target_length, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_memcpy(context%session, slot_addr, source_copy, &
                        copy_length64, error_msg)) return
            end do
            handled = .true.
            return
        end if

        call get_identifier_name(arena, node%value_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol_compat(context, source_name)
        if (source_index <= 0) then
            ! Let the scalar expression path report the normal diagnostic for
            ! an unknown character identifier.
            call char_expr_operands(arena, node%value_index, context, source_data, &
                                    source_length, error_msg)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            call char_expr_operands(arena, node%value_index, context, source_data, &
                                    source_length, error_msg)
            if (len_trim(error_msg) > 0) return
            target_length = i32_immediate(context%session, &
                int(element_length, c_int64_t))
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, &
                    source_length, target_length, source_fits, error_msg)) return
            call select_value(context, source_fits, source_length, target_length, &
                              copy_length, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_i64(context%session, copy_length, &
                    copy_length64, error_msg)) return
            n = int(context%symbols(symbol_index)%array_size, c_int64_t)
            do k = 0_c_int64_t, n - 1_c_int64_t
                element_index = i32_immediate(context%session, k)
                call character_array_element_address(context, symbol_index, &
                                                     element_index, slot_addr, &
                                                     error_msg)
                if (len_trim(error_msg) > 0) return
                call fill_spaces(context, slot_addr, target_length, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_memcpy(context%session, slot_addr, source_data, &
                        copy_length64, error_msg)) return
            end do
            handled = .true.
            return
        end if
        if (context%symbols(source_index)%value_kind /= VALUE_CHARACTER) return
        if (context%symbols(source_index)%is_allocatable) return
        if (context%symbols(source_index)%is_runtime_array) return
        ! Differing element lengths would need per-element padding rather than
        ! a block copy; leave that to the existing element paths.
        if (context%symbols(source_index)%character_length /= element_length) return

        call ensure_array_shapes_match(context, symbol_index, source_index, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_memcpy(context%session, &
                context%symbols(symbol_index)%element_address, &
                context%symbols(source_index)%element_address, &
                i64_immediate(context%session, &
                    int(context%symbols(symbol_index)%array_size, c_int64_t)* &
                    int(element_length, c_int64_t)), error_msg)) return
        handled = .true.
    end subroutine lower_character_array_whole_assignment

    subroutine declare_deferred_char_allocatable_array(node, context, error_msg)
        ! character(len=:), allocatable :: c(:) -- a deferred-length allocatable
        ! character array. The element length is unknown until an
        ! allocate(character(len=N) :: c(M)) statement sets it, so declare the
        ! 40-byte array descriptor now (data = 0, unallocated) and leave
        ! character_length 0 as the pending marker; the allocate fills in both the
        ! element length and the storage. Only rank-1 is supported; any other
        ! form falls back to the existing allocatable path (which declines) so no
        ! previously accepted case regresses.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: rank, symbol_index, i

        call set_empty(error_msg)
        rank = 1
        if (allocated(node%dimension_indices)) rank = size(node%dimension_indices)
        if (rank /= 1) then
            call lower_allocatable_declaration(node, context, error_msg, &
                                               VALUE_CHARACTER)
            return
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call declare_allocatable(context, node%var_names(i), 1, &
                                         VALUE_CHARACTER, error_msg)
                if (len_trim(error_msg) > 0) return
                symbol_index = find_symbol_compat(context, node%var_names(i))
                if (symbol_index > 0) &
                    context%symbols(symbol_index)%character_length = 0
            end do
        else if (allocated(node%var_name)) then
            call declare_allocatable(context, node%var_name, 1, VALUE_CHARACTER, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, node%var_name)
            if (symbol_index > 0) &
                context%symbols(symbol_index)%character_length = 0
        else
            error_msg = 'character array declaration did not expose a variable name'
        end if
    end subroutine declare_deferred_char_allocatable_array

    subroutine lower_allocate_deferred_char_array(arena, node, context, handled, &
                                                  error_msg)
        ! allocate(character(len=N) :: c(M)) for a deferred-length allocatable
        ! character array: fold the compile-time element length N from the
        ! type-spec, record it on the symbol, then allocate M pointer slots and
        ! blank-fill them, exactly like the fixed-length character allocatable
        ! array path. Declines (handled stays .false.) for any target that is not
        ! a rank-1 character allocatable array, or a non-constant element length,
        ! so the existing scalar type-spec allocate paths keep their behaviour.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: spec_lc, len_expr, sub_err
        integer :: symbol_index, size_index
        integer(c_int64_t) :: length_const
        type(call_or_subscript_node) :: target

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%type_spec)) return
        spec_lc = lowercase_text(trim(adjustl(node%type_spec)))
        if (len(spec_lc) < 9) return
        if (spec_lc(1:9) /= 'character') return
        if (.not. allocated(node%var_indices)) return
        if (size(node%var_indices) /= 1) return
        if (.not. node_exists(arena, node%var_indices(1))) return

        select type (v => arena%entries(node%var_indices(1))%node)
        type is (call_or_subscript_node)
            target = v
        class default
            return
        end select
        if (.not. allocated(target%name)) return
        if (.not. allocated(target%arg_indices)) return
        if (size(target%arg_indices) /= 1) return

        symbol_index = find_symbol_compat(context, target%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_allocatable) return
        if (context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) return
        if (context%symbols(symbol_index)%array_rank /= 1) return

        call extract_type_spec_len(node%type_spec, len_expr, sub_err)
        if (len_trim(sub_err) > 0) return
        length_const = deferred_char_array_len_const(len_expr)
        if (length_const < 0_c_int64_t) return
        context%symbols(symbol_index)%character_length = int(length_const)

        size_index = target%arg_indices(1)
        call lower_allocate_i32_1d(arena, size_index, symbol_index, context, &
                                   error_msg)
        if (len_trim(error_msg) > 0) return
        call record_allocatable_static_size(arena, size_index, symbol_index, &
                                            context)
        call init_allocatable_character_blanks(context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        handled = .true.
    end subroutine lower_allocate_deferred_char_array

    integer(c_int64_t) function deferred_char_array_len_const(len_expr) &
            result(length_const)
        ! Fold a character type-spec length expression to a compile-time integer,
        ! or -1 when it is not a plain non-negative integer literal. A runtime or
        ! symbolic length is declined so the caller falls through to the existing
        ! (declining) allocate paths.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        character(len=*), intent(in) :: len_expr
        character(len=:), allocatable :: expr
        integer :: ios

        length_const = -1_c_int64_t
        expr = trim(adjustl(len_expr))
        if (len_trim(expr) == 0) return
        if (verify(trim(expr), '0123456789') /= 0) return
        read (expr, *, iostat=ios) length_const
        if (ios /= 0) length_const = -1_c_int64_t
    end function deferred_char_array_len_const
