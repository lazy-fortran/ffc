    ! Fixed-length character element arrays (#W1). Storage is a [N x ptr] buffer
    ! in element_address: each slot holds a pointer to a materialized, blank-padded
    ! LIRIC string of the declared length. Element assignment materializes the
    ! normalized literal and stores its pointer; element print loads the slot
    ! pointer and reuses the scalar string print path. Scope: local rank-1 and
    ! rank-2 arrays with a compile-time length and extent; allocatable,
    ! deferred-length, dummy, and parameter character arrays stay unsupported.

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
            call lower_allocatable_declaration(node, context, error_msg)
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
            call unsupported_feature_error('character array declaration', &
                node%line, node%column, 'direct LIRIC session supports only a '// &
                'compile-time character length for character arrays', error_msg)
            return
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
        index = find_symbol(context, node%var_name)
        if (index <= 0) return
        if (context%symbols(index)%is_parameter) is_dummy = .true.
    end function character_array_is_dummy

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

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate array declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
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

        if (.not. emit_ptr_array_alloca(context%session, array_size, &
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
            if (.not. emit_ptr_array_element_addr(context%session, array_size, &
                    context%symbols(symbol_index)%element_address, index_value, &
                    slot_addr, error_msg)) return
            if (.not. emit_ptr_store(context%session, string_ptr, slot_addr, &
                                     error_msg)) return
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
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: character_length
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: string_name
        character(len=:), allocatable :: blanks
        type(lr_operand_desc_t) :: blank_ptr
        type(lr_operand_desc_t) :: index_value
        type(lr_operand_desc_t) :: slot_addr
        integer :: i

        call set_empty(error_msg)
        allocate (character(len=character_length) :: blanks)
        blanks = repeat(' ', character_length)
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), blanks, &
                                      blank_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 0, array_size - 1
            index_value = i32_immediate(context%session, int(i, c_int64_t))
            if (.not. emit_ptr_array_element_addr(context%session, array_size, &
                    context%symbols(symbol_index)%element_address, index_value, &
                    slot_addr, error_msg)) return
            if (.not. emit_ptr_store(context%session, blank_ptr, slot_addr, &
                                     error_msg)) return
        end do
    end subroutine init_character_array_blanks

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
        symbol_index = find_symbol(context, node%name)
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
        if (.not. emit_ptr_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, linear_index, &
                slot_addr, error_msg)) then
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
        type(lr_operand_desc_t) :: string_ptr
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok

        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'character array assignment value does not exist'
            return
        end if
        call concat_character_literals(arena, node%value_index, literal_text, &
                                       fold_ok)
        if (.not. fold_ok) then
            call unsupported_feature_error('character array assignment', &
                node%line, node%column, 'direct LIRIC session supports only '// &
                'character-literal assignment into character array elements', &
                error_msg)
            return
        end if

        call lower_character_array_element_slot(arena, target, context, &
                                                symbol_index, slot_addr, error_msg)
        if (len_trim(error_msg) > 0) return

        call normalize_character_literal(literal_text, &
            context%symbols(symbol_index)%character_length)
        context%string_literal_count = context%string_literal_count + 1
        string_name = ffc_unit_global_name( &
            context, 'char.', context%string_literal_count)
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, string_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_store(context%session, string_ptr, slot_addr, &
                                 error_msg)) then
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
        if (.not. emit_ptr_load(context%session, slot_addr, string_ptr, &
                                error_msg)) then
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
            if (.not. emit_ptr_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, index_value, &
                    slot_addr, error_msg)) return
            if (.not. emit_ptr_load(context%session, slot_addr, string_ptr, &
                                    error_msg)) return
            if (.not. emit_liric_print_string_operand_value(context%session, &
                    context%str_print_format_id, string_ptr, error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine emit_character_array_print_items
