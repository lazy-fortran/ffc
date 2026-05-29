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

        if (.not. node_exists(arena, node%value_index)) then
            error_msg = 'character assignment value does not reference an AST node'
            return
        end if

       ! Check for deferred character function call result
        if (context%symbols(symbol_index)%is_deferred_character) then
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
            select type (value_node => arena%entries(node%value_index)%node)
            type is (binary_op_node)
                if (trim(value_node%operator) == '//') then
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
            type is (literal_node)
                if (is_character_literal(value_node)) then
                    call lower_deferred_literal_assignment(value_node, &
                        context, symbol_index, error_msg)
                    return
                end if
            end select
        end if

        select type (value_node => arena%entries(node%value_index)%node)
        type is (literal_node)
            if (.not. is_character_literal(value_node)) then
                call unsupported_feature_error('character assignment', &
                                               node%line, node%column, &
                                               'only character literal '// &
                                               'assignment is supported', &
                                               error_msg)
                return
            end if
            call strip_literal_quotes(value_node%value, literal_text)
        type is (binary_op_node)
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
        class default
            call unsupported_feature_error('character assignment', &
                                           node%line, node%column, &
                                           'only character literal assignment '// &
                                           'is supported', error_msg)
            return
        end select

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

        ok = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            if (.not. is_character_literal(node)) return
            call strip_literal_quotes(node%value, piece)
            accumulated = piece
            ok = .true.
        type is (binary_op_node)
            if (trim(node%operator) /= '//') return
            call concat_character_literals(arena, node%left_index, left_text, ok)
            if (.not. ok) return
            call concat_character_literals(arena, node%right_index, right_text, ok)
            if (.not. ok) return
            accumulated = left_text//right_text
            ok = .true.
        end select
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
        select type (arg => arena%entries(node%arg_indices(1))%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, arg%name)
        class default
            call unsupported_feature_error('len/len_trim argument', node%line, &
                node%column, 'direct LIRIC session only supports len/len_trim '// &
                'of a character variable', error_msg)
            return
        end select
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

    subroutine lower_i32_len_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        type(lr_operand_desc_t) :: data_ptr

        call char_length_arg_symbol(arena, node, context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_length_operands(context, symbol_index, data_ptr, value, &
                                  error_msg)
    end subroutine lower_i32_len_intrinsic

    subroutine lower_i32_len_trim_intrinsic(arena, node, context, value, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        type(lr_operand_desc_t) :: data_ptr
        type(lr_operand_desc_t) :: length
        type(lr_operand_desc_t) :: idx_addr, res_addr
        type(lr_operand_desc_t) :: i_value, prev_index, byte_value, cond
        type(lr_operand_desc_t) :: space
        integer(c_int32_t) :: header_block, body_block, found_block
        integer(c_int32_t) :: dec_block, done_block

        call char_length_arg_symbol(arena, node, context, symbol_index, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_length_operands(context, symbol_index, data_ptr, length, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

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
    end subroutine lower_i32_len_trim_intrinsic
