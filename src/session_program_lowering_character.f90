submodule (session_program_lowering_impl) session_program_lowering_character
    implicit none
contains
    module function is_character_substring(arena, node_index, context) &
            result(is_substring)
        ! True when node is s(l:u) on a scalar character value.  FortFront
        ! represents the nested form c(i)(l:u) as a slice whose base is the
        ! character-array element designator; retaining that base is what keeps
        ! the view on c(i) instead of silently selecting c(1).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        logical :: is_substring
        character(len=:), allocatable :: base_name, name_err
        integer :: symbol_index

        is_substring = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (array_slice_node)
            ! FortFront has already disambiguated a character substring from
            ! an array section.  This flag is especially important for the
            ! nested form c(i)(l:u), whose base expression may not retain the
            ! array-access marker after semantic analysis.
            if (n%is_character_substring) then
                is_substring = .true.
                return
            end if
            if (n%num_dimensions /= 1) return
            if (is_character_array_element(arena, n%array_index, context)) then
                is_substring = .true.
                return
            end if
            call identifier_name(arena, n%array_index, base_name, name_err)
            if (len_trim(name_err) > 0) return
            symbol_index = find_symbol_compat(context, base_name)
            if (symbol_index > 0) then
                is_substring = context%symbols(symbol_index)%value_kind == &
                    VALUE_CHARACTER .and. .not. &
                    context%symbols(symbol_index)%is_array
            end if
        end select
    end function is_character_substring

    module subroutine substring_operands(arena, node_index, context, data_ptr, &
                                         length, error_msg)
        ! Lower a scalar-character substring to a borrowed pointer/length view.
        ! The complete base expression is resolved first, so c(i)(l:u) uses
        ! c(i)'s address and remains correct for reads, writes, overlap, and
        ! character dummy arguments.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name
        integer :: symbol_index, bounds_index, lower_index, upper_index
        integer :: base_index, i
        type(lr_operand_desc_t) :: base_data, base_len, lower_i32, upper_i32
        type(lr_operand_desc_t) :: zero_based, span
        type(lr_operand_desc_t) :: view_buffer
        logical :: keep_view

        call set_empty(error_msg)
        keep_view = .false.
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            select type (assignment => arena%entries(i)%node)
            type is (assignment_node)
                if (assignment%target_index == node_index) keep_view = .true.
            end select
        end do
        bounds_index = 0
        base_index = 0
        lower_index = -1
        upper_index = -1
        select type (n => arena%entries(node_index)%node)
        type is (array_slice_node)
            base_index = n%array_index
            select type (base => arena%entries(base_index)%node)
            type is (call_or_subscript_node)
                if (.not. allocated(base%name)) then
                    error_msg = 'substring base has no designator name'
                    return
                end if
                base_name = base%name
            class default
                call identifier_name(arena, base_index, base_name, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
            bounds_index = n%bounds_indices(1)
        class default
            error_msg = 'expected a substring reference'
            return
        end select

        symbol_index = find_symbol_compat(context, base_name)
        if (symbol_index <= 0) then
            error_msg = 'substring base was not declared: '//trim(base_name)
            return
        end if
        ! Resolve the complete base designator.  Resolving only its spelling
        ! would address c's first element rather than the selected c(i).
        call char_expr_operands(arena, base_index, context, base_data, base_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. node_exists(arena, bounds_index)) then
            error_msg = 'substring bounds do not reference an AST node'
            return
        end if
        select type (b => arena%entries(bounds_index)%node)
        type is (array_bounds_node)
            if (b%stride_index > 0) then
                call unsupported_feature_error('substring', 0, 0, &
                    'a substring has no stride', error_msg)
                return
            end if
            lower_index = b%lower_bound_index
            upper_index = b%upper_bound_index
        type is (range_expression_node)
            if (b%stride_index > 0) then
                call unsupported_feature_error('substring', 0, 0, &
                    'a substring has no stride', error_msg)
                return
            end if
            lower_index = b%start_index
            upper_index = b%end_index
        class default
            error_msg = 'substring bounds are not a range'
            return
        end select

        call reject_constant_substring_overrun(arena, context, symbol_index, &
                                               lower_index, upper_index, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return

        if (lower_index > 0) then
            call lower_i32_expression(arena, lower_index, context, lower_i32, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        else
            lower_i32 = i32_immediate(context%session, 1_c_int64_t)
        end if
        if (upper_index > 0) then
            call lower_i32_expression(arena, upper_index, context, upper_i32, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        else
            upper_i32 = base_len
        end if

        if (.not. emit_i32_binary(context%session, LR_OP_SUB, lower_i32, &
                i32_immediate(context%session, 1_c_int64_t), zero_based, &
                error_msg)) return
        call ptr_plus_i32(context, base_data, zero_based, data_ptr, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_i32, &
                zero_based, span, error_msg)) return
        length = span
        if (.not. keep_view) then
            call materialize_character_view(context, data_ptr, length, &
                                            view_buffer, error_msg)
            if (len_trim(error_msg) > 0) return
            data_ptr = view_buffer
        end if
        call set_empty(error_msg)
    end subroutine substring_operands

    module function actual_is_character(arena, node_index, context) &
            result(is_character)
        ! Character actuals include nested array-element substrings; this keeps
        ! the argument path on the {data,length} descriptor ABI.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        logical :: is_character
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        is_character = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_character_literal(arena, node_index)) then
            is_character = .true.
            return
        end if
        if (is_character_substring(arena, node_index, context)) then
            is_character = .true.
            return
        end if
        if (is_char_expr_call(arena, node_index, context)) then
            is_character = .true.
            return
        end if
        select type (n => arena%entries(node_index)%node)
        type is (component_access_node)
            is_character = derived_component_access_kind(arena, n, &
                context) == VALUE_CHARACTER
            return
        end select
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = resolve_symbol_at_node(context, node_index, id_name)
            if (symbol_index > 0) is_character = &
                context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
        end if
    end function actual_is_character
end submodule session_program_lowering_character
