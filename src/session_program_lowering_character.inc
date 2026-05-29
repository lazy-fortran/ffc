    subroutine define_declared_character_symbol(context, node, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: character_length
        integer :: existing_index

        existing_index = find_symbol(context, name)
        if (existing_index > 0) then
            if (context%in_internal_function .and. &
                existing_index == context%current_function_result_index) then
                call declaration_character_length(node, character_length, &
                                                  error_msg)
                if (len_trim(error_msg) > 0) return
                if (character_length > 0 .and. &
                    context%symbols(existing_index)%is_deferred_character) then
                    call unsupported_feature_error( &
                        'character function result declaration', node%line, &
                        node%column, 'deferred character result lowering '// &
                        'requires character(len=:)', error_msg)
                    return
                end if
                call set_empty(error_msg)
                return
            end if
            if (context%symbols(existing_index)%is_parameter) then
                call declaration_character_length(node, character_length, &
                                                  error_msg)
                if (len_trim(error_msg) > 0) return
                if (character_length > 0) then
                    call unsupported_feature_error( &
                        'character parameter declaration', node%line, &
                        node%column, 'only assumed-length character '// &
                        'parameters are supported by direct LIRIC session', &
                        error_msg)
                    return
                end if
                call bind_character_parameter_symbol(context, existing_index, &
                                                     error_msg)
                return
            end if
        end if
        call declaration_character_length(node, character_length, error_msg)
        if (len_trim(error_msg) > 0) return
        call define_character_symbol(context, name, character_length, error_msg)
    end subroutine define_declared_character_symbol

    subroutine declaration_character_length(node, character_length, error_msg)
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: parse_error
        character(len=:), allocatable :: lowered

        character_length = -1
        call set_empty(error_msg)
        if (.not. allocated(node%type_name)) then
            return
        end if
        lowered = lowercase_text(node%type_name)
        if (trim(lowered) == 'character') then
            ! Deferred character: no len=N specified
            return
        end if
        call parse_character_length(node%type_name, character_length, parse_error)
        if (len_trim(parse_error) > 0) then
            call unsupported_feature_error('character length declaration', &
                                           node%line, node%column, &
                                           trim(parse_error), error_msg)
        end if
    end subroutine declaration_character_length

    subroutine define_character_symbol(context, name, character_length, &
                                       error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate character declaration: '//trim(name)
            return
        end if
       call grow_symbols(context)
        if (character_length <= 0) then
            call define_deferred_character_symbol(context, name, error_msg)
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_CHARACTER
        context%symbols(index)%character_length = character_length
        context%symbols(index)%has_character_value = .false.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_character_symbol

    subroutine lower_character_assignment(arena, node, context, symbol_index, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name
        logical :: fold_ok
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type

        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'character assignment value does not reference an AST node'
            return
        end if

       ! Check for deferred character function call result or trim() result
        if (context%symbols(symbol_index)%is_deferred_character) then
            if (is_char_expr_call(arena, node%value_index)) then
                call lower_deferred_trim_assignment(arena, node%value_index, &
                    context, symbol_index, error_msg)
                return
            end if
            if (node_exists(arena, node%value_index)) then
                select type (val => arena%entries(node%value_index)%node)
                type is (call_or_subscript_node)
                    if (.not. val%is_array_access .and. &
                        allocated(val%name)) then
                        if (is_contained_deferred_char_function(context, &
                            val%name)) then
                            call lower_deferred_char_result_call(arena, node, &
                                context, symbol_index, error_msg)
                            return
                        end if
                    end if
                end select
            end if
        end if

        ! Check for deferred character with binary // operator or literal
        if (context%symbols(symbol_index)%is_deferred_character) then
            if (is_binary_op(arena, node%value_index)) then
                call get_binary_op_info(arena, node%value_index, bin_op, &
                                        bin_left, bin_right, bin_line, bin_col, &
                                        error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(bin_op) == '//') then
                    ! An all-literal // chain (any depth) folds to one literal.
                    call concat_character_literals(arena, node%value_index, &
                                                   literal_text, fold_ok)
                    if (fold_ok) then
                        call lower_deferred_folded_literal_assignment( &
                            literal_text, context, symbol_index, error_msg)
                        return
                    end if
                    call lower_deferred_concat_assignment(arena, node, &
                        context, symbol_index, error_msg)
                    return
                end if
            else if (is_literal(arena, node%value_index)) then
                if (is_character_literal(arena, node%value_index)) then
                    call get_literal_info(arena, node%value_index, lit_value, &
                                          lit_type, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call lower_deferred_literal_assignment(lit_value, &
                        context, symbol_index, error_msg)
                    return
                end if
            end if
        end if

        if (is_binary_op(arena, node%value_index)) then
            call concat_character_literals(arena, node%value_index, &
                                           literal_text, fold_ok)
            if (.not. fold_ok) then
                call unsupported_feature_error('character assignment', &
                                               node%line, node%column, &
                                               'only character literal '// &
                                               'assignment is supported', &
                                               error_msg)
                return
            end if
        else if (is_literal(arena, node%value_index)) then
            if (.not. is_character_literal(arena, node%value_index)) then
                call unsupported_feature_error('character assignment', &
                                               node%line, node%column, &
                                               'only character literal '// &
                                               'assignment is supported', &
                                               error_msg)
                return
            end if
            call get_literal_info(arena, node%value_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call strip_literal_quotes(lit_value, literal_text)
        else
            call unsupported_feature_error('character assignment', &
                                           node%line, node%column, &
                                           'only character literal '// &
                                           'assignment is supported', &
                                           error_msg)
            return
        end if

        call normalize_character_literal( &
            literal_text, context%symbols(symbol_index)%character_length)

        context%string_literal_count = context%string_literal_count + 1
        write (string_name, '(A,I0)') '.ffc.char.', &
            context%string_literal_count
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, &
                                      context%symbols(symbol_index)%value, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_character_assignment

    recursive subroutine concat_character_literals(arena, node_index, &
                                                   accumulated, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: accumulated
        logical, intent(out) :: ok
        character(len=:), allocatable :: left_text, right_text, piece
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type, lit_err

        ok = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            if (trim(bin_op) /= '//') return
            call concat_character_literals(arena, bin_left, left_text, ok)
            if (.not. ok) return
            call concat_character_literals(arena, bin_right, right_text, ok)
            if (.not. ok) return
            accumulated = left_text//right_text
            ok = .true.
            return
        end if

        if (is_literal(arena, node_index)) then
            if (.not. is_character_literal(arena, node_index)) return
            call get_literal_info(arena, node_index, lit_value, lit_type, lit_err)
            if (len_trim(lit_err) > 0) return
            call strip_literal_quotes(lit_value, piece)
            accumulated = piece
            ok = .true.
        end if
    end subroutine concat_character_literals

    subroutine normalize_character_literal(literal_text, character_length)
        character(len=:), allocatable, intent(inout) :: literal_text
        integer, intent(in) :: character_length
        character(len=:), allocatable :: fixed_text

        allocate (character(len=character_length) :: fixed_text)
        fixed_text(:) = literal_text
        call move_alloc(fixed_text, literal_text)
    end subroutine normalize_character_literal

    logical function is_character_type_name(name)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: lowered

        lowered = lowercase_text(name)
        is_character_type_name = lowered == 'character' .or. &
                                 index(lowered, 'character(') == 1
    end function is_character_type_name

    subroutine parse_character_length(type_name, character_length, error_msg)
        character(len=*), intent(in) :: type_name
        integer, intent(out) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: lowered
        character(len=:), allocatable :: length_text
        integer :: close_pos
        integer :: io_stat
        integer :: len_pos
        integer :: start_pos

        character_length = 1
        lowered = lowercase_text(type_name)
        if (trim(lowered) == 'character') then
            call set_empty(error_msg)
            return
        end if

        len_pos = index(lowered, 'len=')
        if (len_pos <= 0) then
            error_msg = 'only character(len=N) declarations are supported'
            return
        end if

        start_pos = len_pos + len('len=')
        close_pos = index(lowered(start_pos:), ')')
        if (close_pos <= 0) then
            error_msg = 'character length declaration is missing ")"'
            return
        end if

        close_pos = start_pos + close_pos - 2
        length_text = adjustl(lowered(start_pos:close_pos))
        if (trim(length_text) == ':' .or. trim(length_text) == '*') then
            character_length = -1
            call set_empty(error_msg)
            return
        end if
        read (length_text, *, iostat=io_stat) character_length
        if (io_stat /= 0 .or. character_length <= 0) then
            error_msg = 'character length must be a positive integer literal'
            return
        end if

        call set_empty(error_msg)
    end subroutine parse_character_length

    subroutine define_deferred_character_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        type(lr_operand_desc_t) :: data_addr
        type(lr_operand_desc_t) :: len_addr

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate character declaration: '//trim(name)
            return
        end if
     call grow_symbols(context)

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_CHARACTER
        context%symbols(index)%is_deferred_character = .true.
        context%symbols(index)%has_character_value = .false.

        ! Allocate stack space for the 16-byte descriptor (data + length) and
        ! zero both fields so the unallocated state is well-defined per the ABI.
        if (.not. emit_i64_alloca(context%session, data_addr, error_msg)) return
        if (.not. emit_i64_alloca(context%session, len_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                                 i64_immediate(context%session, 0_c_int64_t), &
                                 data_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                                 i64_immediate(context%session, 0_c_int64_t), &
                                 len_addr, error_msg)) return

        context%symbols(index)%deferred_data = data_addr
        context%symbols(index)%deferred_length = len_addr
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_deferred_character_symbol

    subroutine bind_character_parameter_symbol(context, symbol_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg

        context%symbols(symbol_index)%value_kind = VALUE_CHARACTER
        context%symbols(symbol_index)%is_deferred_character = .true.
        context%symbols(symbol_index)%has_character_value = .true.
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(symbol_index)%address, 0_c_int64_t, &
            context%symbols(symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
            context%symbols(symbol_index)%address, 8_c_int64_t, &
            context%symbols(symbol_index)%deferred_length, error_msg)) return
        call set_empty(error_msg)
    end subroutine bind_character_parameter_symbol

    subroutine char_length_arg_symbol(arena, node, context, symbol_index, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name

        symbol_index = 0
        if (.not. allocated(node%arg_indices)) then
            error_msg = 'len/len_trim require one character argument'
            return
        end if
        if (size(node%arg_indices) /= 1) then
            error_msg = 'len/len_trim require exactly one argument'
            return
        end if
        if (.not. node_exists(arena, node%arg_indices(1))) then
            error_msg = 'len/len_trim argument does not reference an AST node'
            return
        end if
        if (is_identifier(arena, node%arg_indices(1))) then
            call get_identifier_name(arena, node%arg_indices(1), id_name, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
        else
            call unsupported_feature_error('len/len_trim argument', node%line, &
                node%column, 'direct LIRIC session only supports len/len_trim '// &
                'of a character variable', error_msg)
            return
        end if
        if (symbol_index <= 0) then
            error_msg = 'len/len_trim argument is not a declared character: '// &
                        trim(node%name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) then
            call unsupported_feature_error('len/len_trim argument', node%line, &
                node%column, 'direct LIRIC session only supports len/len_trim '// &
                'of a character variable', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine char_length_arg_symbol

    subroutine char_length_operands(context, symbol_index, data_ptr, length, &
                                     error_msg)
        ! Produce the data pointer and i32 length for a character symbol.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg

        if (context%symbols(symbol_index)%is_deferred_character) then
            if (.not. emit_ptr_load(context%session, &
                context%symbols(symbol_index)%deferred_data, data_ptr, &
                error_msg)) return
            ! Length lives in an i64 slot; load its low word as i32 (lengths
            ! never approach 2**31 here).
            if (.not. emit_i32_load(context%session, &
                context%symbols(symbol_index)%deferred_length, length, &
                error_msg)) return
        else
            data_ptr = context%symbols(symbol_index)%value
            length = i32_immediate(context%session, &
                int(context%symbols(symbol_index)%character_length, c_int64_t))
            call set_empty(error_msg)
        end if
    end subroutine char_length_operands

    subroutine lower_deferred_trim_assignment(arena, trim_index, context, &
                                              dest_symbol_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: trim_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length
        type(lr_operand_desc_t) :: length64

        call char_expr_operands(arena, trim_index, context, data_ptr, length, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, length, length64, &
                                        error_msg)) return
        if (.not. emit_ptr_store(context%session, data_ptr, &
            context%symbols(dest_symbol_index)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, length64, &
            context%symbols(dest_symbol_index)%deferred_length, error_msg)) return
        context%symbols(dest_symbol_index)%value = data_ptr
        context%symbols(dest_symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_deferred_trim_assignment

    subroutine lower_i32_len_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'len requires one character argument'
            return
        end if
        ! len of a character expression (variable or trim()): the descriptor
        ! length. char_expr_operands materialises trim() as a trimmed buffer.
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                value, error_msg)
    end subroutine lower_i32_len_intrinsic

    logical function is_trim_call(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_trim_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. n%is_array_access .and. allocated(n%name)) &
                is_trim_call = same_name(n%name, 'trim')
        end select
    end function is_trim_call

    subroutine char_expr_operands(arena, node_index, context, data_ptr, length, &
                                  error_msg)
        ! Resolve a scalar character expression (a variable or a trim() call) to
        ! its data pointer and i32 length. trim() materialises a fresh
        ! null-terminated buffer holding the trimmed text.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'character expression does not reference an AST node'
            return
        end if
        if (is_trim_call(arena, node_index)) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_trim_deferred(arena, n, context, data_ptr, length, &
                                         error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'achar')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_achar_deferred(arena, n, context, data_ptr, length, &
                                          error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'adjustl')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_adjust_deferred(arena, n, context, .true., data_ptr, &
                                           length, error_msg)
            end select
            return
        end if
        if (is_named_char_call(arena, node_index, 'adjustr')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_adjust_deferred(arena, n, context, .false., data_ptr, &
                                           length, error_msg)
            end select
            return
        end if
        if (is_literal(arena, node_index)) then
            if (.not. is_character_literal(arena, node_index)) then
                call unsupported_feature_error('character expression', 0, 0, &
                    'direct LIRIC session only supports a character variable, '// &
                    'literal, or trim() here', error_msg)
                return
            end if
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call char_literal_operands(context, lit_value, data_ptr, length, &
                                       error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0 .or. &
                context%symbols(max(symbol_index, 1))%value_kind /= &
                VALUE_CHARACTER) then
                call unsupported_feature_error('character expression', 0, 0, &
                    'direct LIRIC session only supports a character variable '// &
                    'or trim() here', error_msg)
                return
            end if
            call char_length_operands(context, symbol_index, data_ptr, length, &
                                      error_msg)
            return
        end if

        call unsupported_feature_error('character expression', 0, 0, &
            'direct LIRIC session only supports a character variable or '// &
            'trim() here', error_msg)
    end subroutine char_expr_operands

    logical function is_char_expr_call(arena, node_index)
        ! A call that produces a character value usable in print/assign/len.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_char_expr_call = is_trim_call(arena, node_index) .or. &
                            is_named_char_call(arena, node_index, 'achar') .or. &
                            is_named_char_call(arena, node_index, 'adjustl') .or. &
                            is_named_char_call(arena, node_index, 'adjustr')
    end function is_char_expr_call

    logical function is_named_char_call(arena, node_index, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name

        is_named_char_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. n%is_array_access .and. allocated(n%name)) &
                is_named_char_call = same_name(n%name, name)
        end select
    end function is_named_char_call

    subroutine lower_achar_deferred(arena, node, context, out_data, out_length, &
                                    error_msg)
        ! achar(i): a length-1 character whose byte is the low 8 bits of i.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: code
        type(lr_operand_desc_t) :: buf

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'achar requires one integer argument'
            return
        end if
        call lower_i32_expression(arena, node%arg_indices(1), context, code, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        ! 2 bytes: the character plus a null terminator.
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 2_c_int64_t), buf, error_msg)) &
            return
        if (.not. emit_liric_store_char_byte(context%session, buf, &
                i32_immediate(context%session, 0_c_int64_t), code, error_msg)) &
            return
        if (.not. emit_liric_store_char_byte(context%session, buf, &
                i32_immediate(context%session, 1_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
        out_data = buf
        out_length = i32_immediate(context%session, 1_c_int64_t)
        call set_empty(error_msg)
    end subroutine lower_achar_deferred

    subroutine lower_iachar_intrinsic(arena, node, context, value, error_msg)
        ! iachar(c): the ASCII code of the first byte of c.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'iachar requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                length, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_char_byte_zext(context%session, data_ptr, &
                i32_immediate(context%session, 0_c_int64_t), value, error_msg)) &
            return
    end subroutine lower_iachar_intrinsic

    subroutine char_literal_operands(context, literal_value, data_ptr, length, &
                                      error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: literal_value
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: literal_text
        character(len=64) :: string_name

        call strip_literal_quotes(literal_value, literal_text)
        context%string_literal_count = context%string_literal_count + 1
        write (string_name, '(A,I0)') '.ffc.cexpr.', context%string_literal_count
        call materialize_liric_string(context%session, trim(string_name), &
                                      literal_text, data_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        length = i32_immediate(context%session, int(len(literal_text), c_int64_t))
    end subroutine char_literal_operands

    subroutine compute_leading_blanks(context, data_ptr, length, value, error_msg)
        ! Count leading space bytes in data[0:length).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lr_operand_desc_t), intent(in) :: length
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx_addr, i_val, byte_value, cond, tmp
        integer(c_int32_t) :: hdr, body, inc, done

        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), idx_addr, &
                error_msg)) return
        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        inc = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, &
                length, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_liric_char_byte_zext(context%session, data_ptr, i_val, &
                byte_value, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, byte_value, &
                i32_immediate(context%session, 32_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, inc, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, inc, error_msg)) return
        context%current_block_id = inc
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, idx_addr, value, error_msg)) return
        call set_empty(error_msg)
    end subroutine compute_leading_blanks

    subroutine fill_spaces(context, buf, count, error_msg)
        ! Store `count` space bytes into buf[0:count).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: buf
        type(lr_operand_desc_t), intent(in) :: count
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx_addr, i_val, cond, tmp
        integer(c_int32_t) :: hdr, body, done

        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), idx_addr, &
                error_msg)) return
        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, count, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_liric_store_char_byte(context%session, buf, i_val, &
                i32_immediate(context%session, 32_c_int64_t), error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine fill_spaces

    subroutine lower_adjust_deferred(arena, node, context, left, out_data, &
                                     out_length, error_msg)
        ! adjustl(s): move leading blanks to the end. adjustr(s): move trailing
        ! blanks to the front. Result length equals len(s).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: left
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, k, content_len, dst_off
        type(lr_operand_desc_t) :: len64, buf, src, dst, content64, null_pos

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'adjustl/adjustr requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_i32_to_i64(context%session, s_len, len64, &
                error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                  emit_or_add1(context, len64), buf, error_msg)) then
            return
        end if
        ! Pre-fill the whole result with spaces, then drop the content in place.
        call fill_spaces(context, buf, s_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (left) then
            call compute_leading_blanks(context, s_data, s_len, k, error_msg)
            if (len_trim(error_msg) > 0) return
            ! content = s[k:], length s_len-k, placed at offset 0
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, s_len, k, &
                    content_len, error_msg)) return
            call ptr_plus_i32(context, s_data, k, src, error_msg)
            if (len_trim(error_msg) > 0) return
            dst = buf
        else
            ! trailing blanks: content = s[0:len_trim], placed at offset s_len-LT
            call compute_len_trim(context, s_data, s_len, content_len, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, s_len, &
                    content_len, dst_off, error_msg)) return
            src = s_data
            call ptr_plus_i32(context, buf, dst_off, dst, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (.not. emit_liric_i32_to_i64(context%session, content_len, content64, &
                error_msg)) return
        if (.not. emit_memcpy(context%session, dst, src, content64, error_msg)) &
            return
        ! null-terminate at buf[s_len]
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, len64, &
                null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return
        out_data = buf
        out_length = s_len
        call set_empty(error_msg)
    end subroutine lower_adjust_deferred

    function emit_or_add1(context, len64) result(size_op)
        ! len64 + 1 (buffer holds the result plus a null terminator).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: len64
        type(lr_operand_desc_t) :: size_op
        character(len=:), allocatable :: err
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, len64, &
                i64_immediate(context%session, 1_c_int64_t), size_op, err)) then
            size_op = len64
        end if
    end function emit_or_add1

    subroutine ptr_plus_i32(context, base, offset_i32, result, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: base
        type(lr_operand_desc_t), intent(in) :: offset_i32
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: off64

        if (.not. emit_liric_i32_to_i64(context%session, offset_i32, off64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, base, off64, &
                result, error_msg)) return
        call set_empty(error_msg)
    end subroutine ptr_plus_i32

    subroutine char_equals(context, a_data, a_len, b_data, b_len, result, &
                           error_msg)
        ! Fortran character equality with blank padding: compare the common
        ! prefix, then require the longer value's tail to be all spaces.
        ! result (i32) is 1 when equal, else 0. Both operands are read only
        ! within their own lengths (no out-of-bounds load).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: a_data, a_len, b_data, b_len
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: flag_addr, idx_addr, i_val, cond, ab, bb, tmp
        type(lr_operand_desc_t) :: a_le_b, minlen, maxlen, tail_data, byte
        integer(c_int32_t) :: hdr, body, advance, set_zero, prefix_done
        integer(c_int32_t) :: thdr, tbody, tadvance, done

        if (.not. emit_i32_alloca(context%session, flag_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 1_c_int64_t), flag_addr, &
                error_msg)) return
        ! minlen / maxlen and the longer operand's data pointer.
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, a_len, b_len, &
                a_le_b, error_msg)) return
        call select_value(context, a_le_b, a_len, b_len, minlen, error_msg)
        if (len_trim(error_msg) > 0) return
        call select_value(context, a_le_b, b_len, a_len, maxlen, error_msg)
        if (len_trim(error_msg) > 0) return
        call select_value(context, a_le_b, b_data, a_data, tail_data, error_msg)
        if (len_trim(error_msg) > 0) return

        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        advance = create_liric_block(context%session)
        set_zero = create_liric_block(context%session)
        prefix_done = create_liric_block(context%session)
        thdr = create_liric_block(context%session)
        tbody = create_liric_block(context%session)
        tadvance = create_liric_block(context%session)
        done = create_liric_block(context%session)

        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), idx_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        ! prefix: compare [0, minlen)
        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, minlen, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, prefix_done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_liric_char_byte_zext(context%session, a_data, i_val, ab, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, b_data, i_val, bb, &
                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, ab, bb, cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, set_zero, advance, &
                error_msg)) return

        if (.not. set_liric_block(context%session, advance, error_msg)) return
        context%current_block_id = advance
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        ! tail: from minlen to maxlen, the longer value must be all spaces.
        if (.not. set_liric_block(context%session, prefix_done, error_msg)) return
        context%current_block_id = prefix_done
        if (.not. emit_i32_store(context%session, minlen, idx_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, thdr, error_msg)) return

        if (.not. set_liric_block(context%session, thdr, error_msg)) return
        context%current_block_id = thdr
        if (.not. emit_i32_load(context%session, idx_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, maxlen, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, tbody, done, &
                error_msg)) return

        if (.not. set_liric_block(context%session, tbody, error_msg)) return
        context%current_block_id = tbody
        if (.not. emit_liric_char_byte_zext(context%session, tail_data, i_val, &
                byte, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, byte, &
                i32_immediate(context%session, 32_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, set_zero, tadvance, &
                error_msg)) return

        if (.not. set_liric_block(context%session, tadvance, error_msg)) return
        context%current_block_id = tadvance
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, idx_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, thdr, error_msg)) return

        if (.not. set_liric_block(context%session, set_zero, error_msg)) return
        context%current_block_id = set_zero
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), flag_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, done, error_msg)) return

        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, flag_addr, result, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine char_equals

    subroutine lower_index_intrinsic(arena, node, context, value, error_msg)
        ! index(s, sub): 1-based position of the first occurrence of sub in s,
        ! or 0. Implemented as an outer position scan with an inner byte compare,
        ! both driven by stack counters (no SSA phi).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, sub_data, sub_len
        type(lr_operand_desc_t) :: i_addr, j_addr, res_addr, max_start
        type(lr_operand_desc_t) :: i_val, j_val, cond, sb, tb, sum_ij, tmp
        integer(c_int32_t) :: outer_hdr, outer_body, inner_hdr, inner_body
        integer(c_int32_t) :: matched, advance, done

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'index requires two character arguments'
            return
        end if
        if (size(node%arg_indices) < 2) then
            error_msg = 'index requires two character arguments'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_expr_operands(arena, node%arg_indices(2), context, sub_data, &
                                sub_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_alloca(context%session, i_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, j_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, res_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), i_addr, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), res_addr, &
                error_msg)) return
        ! max_start = s_len - sub_len (last 0-based start to try)
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, s_len, sub_len, &
                max_start, error_msg)) return

        outer_hdr = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        inner_hdr = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        matched = create_liric_block(context%session)
        advance = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! outer_hdr: if i <= max_start goto outer_body else done
        if (.not. set_liric_block(context%session, outer_hdr, error_msg)) return
        context%current_block_id = outer_hdr
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, i_val, &
                max_start, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, outer_body, done, &
                error_msg)) return

        ! outer_body: j = 0; goto inner_hdr
        if (.not. set_liric_block(context%session, outer_body, error_msg)) return
        context%current_block_id = outer_body
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), j_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return

        ! inner_hdr: if j < sub_len goto inner_body else matched
        if (.not. set_liric_block(context%session, inner_hdr, error_msg)) return
        context%current_block_id = inner_hdr
        if (.not. emit_i32_load(context%session, j_addr, j_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, j_val, &
                sub_len, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, inner_body, matched, &
                error_msg)) return

        ! inner_body: if s[i+j] == sub[j] then j=j+1; inner_hdr else advance
        if (.not. set_liric_block(context%session, inner_body, error_msg)) return
        context%current_block_id = inner_body
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, j_val, &
                sum_ij, error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, s_data, sum_ij, sb, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, sub_data, j_val, &
                tb, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, sb, tb, cond, &
                error_msg)) return
        block
            integer(c_int32_t) :: bump
            bump = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, cond, bump, advance, &
                    error_msg)) return
            if (.not. set_liric_block(context%session, bump, error_msg)) return
            context%current_block_id = bump
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, j_val, &
                    i32_immediate(context%session, 1_c_int64_t), tmp, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, tmp, j_addr, error_msg)) &
                return
            if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return
        end block

        ! matched: result = i + 1; goto done
        if (.not. set_liric_block(context%session, matched, error_msg)) return
        context%current_block_id = matched
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, res_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, done, error_msg)) return

        ! advance: i = i + 1; goto outer_hdr
        if (.not. set_liric_block(context%session, advance, error_msg)) return
        context%current_block_id = advance
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, i_addr, error_msg)) &
            return
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! done: value = load result
        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, res_addr, value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine lower_index_intrinsic

    subroutine lower_trim_deferred(arena, node, context, out_data, out_length, &
                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr, length, n, n64, size64
        type(lr_operand_desc_t) :: buf, null_pos, null_ptr
        character(len=64) :: null_name

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'trim requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                length, error_msg)
        if (len_trim(error_msg) > 0) return
        call compute_len_trim(context, data_ptr, length, n, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Allocate n + 1 bytes, copy the trimmed prefix, and null-terminate.
        if (.not. emit_liric_i32_to_i64(context%session, n, n64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, n64, &
                i64_immediate(context%session, 1_c_int64_t), size64, &
                error_msg)) return
        if (.not. emit_alloca_bytes(context%session, size64, buf, error_msg)) &
            return
        if (.not. emit_memcpy(context%session, buf, data_ptr, n64, error_msg)) &
            return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, n64, &
                null_pos, error_msg)) return
        context%string_literal_count = context%string_literal_count + 1
        write (null_name, '(A,I0)') '.ffc.trim.term.', context%string_literal_count
        call materialize_liric_string(context%session, trim(null_name), '', &
                                      null_ptr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_memcpy(context%session, null_pos, null_ptr, &
                i64_immediate(context%session, 1_c_int64_t), error_msg)) return

        out_data = buf
        out_length = n
        call set_empty(error_msg)
    end subroutine lower_trim_deferred

    subroutine lower_i32_len_trim_intrinsic(arena, node, context, value, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length

        if (.not. allocated(node%arg_indices)) then
            error_msg = 'len_trim requires one character argument'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, data_ptr, &
                                length, error_msg)
        if (len_trim(error_msg) > 0) return
        call compute_len_trim(context, data_ptr, length, value, error_msg)
    end subroutine lower_i32_len_trim_intrinsic

    subroutine compute_len_trim(context, data_ptr, length, value, error_msg)
        ! Scan data[0:length) backward for the last non-space byte (1-based).
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lr_operand_desc_t), intent(in) :: length
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx_addr, res_addr
        type(lr_operand_desc_t) :: i_value, prev_index, byte_value, cond
        type(lr_operand_desc_t) :: space
        integer(c_int32_t) :: header_block, body_block, found_block
        integer(c_int32_t) :: dec_block, done_block

        ! idx = length; result = 0; scan backwards for the last non-space byte.
        if (.not. emit_i32_alloca(context%session, idx_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, res_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, length, idx_addr, &
                                 error_msg)) return
        if (.not. emit_i32_store(context%session, &
                                 i32_immediate(context%session, 0_c_int64_t), &
                                 res_addr, error_msg)) return

        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        found_block = create_liric_block(context%session)
        dec_block = create_liric_block(context%session)
        done_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        ! header: i = load idx; if i > 0 goto body else done
        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        if (.not. emit_i32_load(context%session, idx_addr, i_value, &
                                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGT, i_value, &
                i32_immediate(context%session, 0_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, body_block, &
                                    done_block, error_msg)) return

        ! body: byte = data[i-1]; if byte == ' ' goto dec else found
        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, i_value, &
                i32_immediate(context%session, 1_c_int64_t), prev_index, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, data_ptr, &
                prev_index, byte_value, error_msg)) return
        space = i32_immediate(context%session, 32_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, byte_value, &
                space, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, dec_block, &
                                    found_block, error_msg)) return

        ! found: result = i; goto done
        if (.not. set_liric_block(context%session, found_block, error_msg)) return
        context%current_block_id = found_block
        if (.not. emit_i32_store(context%session, i_value, res_addr, &
                                 error_msg)) return
        if (.not. emit_liric_br(context%session, done_block, error_msg)) return

        ! dec: idx = i - 1; goto header
        if (.not. set_liric_block(context%session, dec_block, error_msg)) return
        context%current_block_id = dec_block
        if (.not. emit_i32_store(context%session, prev_index, idx_addr, &
                                 error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        ! done: value = load result
        if (.not. set_liric_block(context%session, done_block, error_msg)) return
        context%current_block_id = done_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, res_addr, value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine compute_len_trim
