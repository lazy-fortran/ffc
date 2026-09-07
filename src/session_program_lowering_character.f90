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

    module subroutine capture_runtime_fixed_character_length(context, &
                                                  symbol_index, source_index, error_msg)
        ! Capture a runtime character width once at declaration.
        ! The storage layout stays the ordinary {data, length} descriptor, but
        ! the runtime-fixed flag makes later assignments retain this length.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: source_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: source_len_i32, source_len_i64
        type(lr_operand_desc_t) :: buffer_size, buffer, null_pos
        integer(c_int64_t) :: storage_class

        context%symbols(symbol_index)%is_runtime_fixed_character = .true.
        call runtime_character_source_length(context, source_index, &
                                             source_len_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, source_len_i32, &
                                        source_len_i64, error_msg)) return
        if (.not. emit_i64_store(context%session, source_len_i64, &
                            context%symbols(symbol_index)%deferred_length, error_msg)) &
            return

        ! Runtime-fixed character results and locals need storage before a
        ! substring assignment can address one of their bytes.  Contained
        ! function results outlive this frame; ordinary top-level locals do
        ! not, so retain the matching ownership class in the descriptor.
        if (.not. context%symbols(symbol_index)%has_character_value) then
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, &
                          source_len_i64, i64_immediate(context%session, 1_c_int64_t), &
                                      buffer_size, error_msg)) return
            ! The descriptor retains the buffer and later expression lowering
            ! may keep operand pointers live across the copy. Dynamic stack
            ! storage can overlap those temporaries in the direct LIRIC
            ! backend, so use owned heap storage in both scopes.
            if (.not. emit_malloc(context%session, buffer_size, buffer, &
                                  error_msg)) return
            storage_class = LOWERING_CHARACTER_STORAGE_OWNED
            call fill_spaces(context, buffer, source_len_i32, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i64_binary(context%session, LR_OP_ADD, buffer, &
                                      source_len_i64, null_pos, error_msg)) return
            if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                                          i32_immediate(context%session, 0_c_int64_t), &
                              i32_immediate(context%session, 0_c_int64_t), error_msg)) &
                return
            if (.not. emit_ptr_store(context%session, buffer, &
                              context%symbols(symbol_index)%deferred_data, error_msg)) &
                return
            call set_character_storage(context, symbol_index, source_len_i64, &
                                       storage_class, error_msg)
            if (len_trim(error_msg) > 0) return
            context%symbols(symbol_index)%value = buffer
            context%symbols(symbol_index)%has_character_value = .true.
        end if
        call set_empty(error_msg)
    end subroutine capture_runtime_fixed_character_length

    module subroutine resolve_runtime_character_length_source(context, &
            node, source_index, allow_integer)
        ! Recognize LEN(character) or a scalar integer specification variable.
        ! source_index is left at 0 when the expression does not match.
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: source_index
        logical, intent(in) :: allow_integer
        character(len=:), allocatable :: expr
        character(len=:), allocatable :: lowered
        character(len=:), allocatable :: inner

        source_index = 0
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        expr = trim(adjustl(node%character_length_expr))
        if (len(expr) == 0) return
        if (verify(expr, &
                   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            == 0) then
            if (.not. allow_integer) return
            source_index = resolve_symbol_at_node(context, &
                                                context%current_declaration_index, expr)
            if (source_index <= 0) return
            if (context%symbols(source_index)%is_array) then
                source_index = 0
                return
            end if
            select case (context%symbols(source_index)%value_kind)
            case (VALUE_I8, VALUE_I16, VALUE_I32, VALUE_I64)
            case default
                source_index = 0
            end select
            return
        end if
        if (len(expr) < 6) return
        lowered = lowercase_text(expr)
        if (lowered(1:4) /= 'len(') return
        if (lowered(len(lowered):len(lowered)) /= ')') return
        inner = trim(adjustl(expr(5:len(expr) - 1)))
        if (len(inner) == 0) return
        if (verify(inner, &
                   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return

        source_index = find_symbol_compat(context, inner)
        if (source_index <= 0) return
        if (context%symbols(source_index)%value_kind /= VALUE_CHARACTER) &
            source_index = 0
    end subroutine resolve_runtime_character_length_source

    subroutine runtime_character_source_length(context, source_index, length, &
                                               error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: source_index
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: raw, narrowed, zero, positive, source_data
        integer :: value_kind

        call set_empty(error_msg)
        value_kind = context%symbols(source_index)%value_kind
        if (value_kind == VALUE_CHARACTER) then
            call char_length_operands(context, source_index, source_data, &
                                      length, error_msg)
            return
        end if
        raw = context%symbols(source_index)%value
        if (context%symbols(source_index)%has_address) then
            call emit_array_value_load(context, value_kind, &
                                  context%symbols(source_index)%address, raw, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        select case (value_kind)
        case (VALUE_I8)
            if (.not. emit_liric_i8_to_i32(context%session, raw, narrowed, &
                                           error_msg)) return
        case (VALUE_I16)
            if (.not. emit_liric_i16_to_i32(context%session, raw, narrowed, &
                                            error_msg)) return
        case (VALUE_I64)
            call narrow_runtime_character_length(context, raw, length, error_msg)
            return
        case default
            narrowed = raw
        end select
        ! Fortran interprets a negative character length as zero.
        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGT, narrowed, &
                                      zero, positive, error_msg)) return
        call select_value(context, positive, narrowed, zero, length, error_msg)
    end subroutine runtime_character_source_length

    subroutine narrow_runtime_character_length(context, raw, length, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: raw
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: zero, positive, nonnegative, too_long
        integer(c_int32_t) :: error_block, valid_block

        ! Clamp in the declared integer kind: narrowing first could turn a
        ! negative length into a positive value or wrap an oversized length.
        zero = i64_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i64_icmp(context%session, LR_CMP_SGT, raw, zero, &
                                    positive, error_msg)) return
        call select_value(context, positive, raw, zero, nonnegative, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i64_icmp(context%session, LR_CMP_SGT, nonnegative, &
                i64_immediate(context%session, 2147483647_c_int64_t), too_long, &
                error_msg)) return
        error_block = create_liric_block(context%session)
        valid_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, too_long, error_block, &
                                   valid_block, error_msg)) return
        if (.not. set_liric_block(context%session, error_block, error_msg)) return
        context%current_block_id = error_block
        call emit_character_length_limit_error(context, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, valid_block, error_msg)) return
        if (.not. set_liric_block(context%session, valid_block, error_msg)) return
        context%current_block_id = valid_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_i64_to_i32(context%session, nonnegative, length, &
                                      error_msg)) return
        call set_empty(error_msg)
    end subroutine narrow_runtime_character_length

    subroutine emit_character_length_limit_error(context, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: global_name
        type(lr_operand_desc_t) :: args(2)
        integer(c_int32_t) :: format_id

        context%string_literal_count = context%string_literal_count + 1
        global_name = ffc_unit_global_name(context, 'char.length.error.', &
                                           context%string_literal_count)
        call create_printf_format_global(context%session, global_name, &
            'Fortran runtime error: Character length exceeds supported maximum '// &
            '2147483647'//achar(10), format_id, error_msg)
        if (len_trim(error_msg) > 0) return
        args(1) = i32_immediate(context%session, 2_c_int64_t)
        args(2) = printf_format_ptr(context%session, format_id)
        if (.not. emit_dprintf(context%session, args, error_msg)) return
        if (.not. emit_exit(context%session, &
                           i32_immediate(context%session, 2_c_int64_t), &
                           error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_character_length_limit_error
end submodule session_program_lowering_character
