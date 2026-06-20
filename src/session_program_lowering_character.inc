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
                call declaration_character_length(context, node, &
                                                  character_length, error_msg)
                ! A runtime length (character(len=n) where n is a dummy arg or
                ! other non-constant) keeps the deferred descriptor ABI: the
                ! length is resolved when the body assigns the result value.
                if (len_trim(error_msg) > 0) then
                    if (runtime_character_length(context, node)) then
                        call set_empty(error_msg)
                    end if
                    return
                end if
                ! A fixed-length result (character(len=N) :: s) reuses the
                ! descriptor-return ABI of the deferred result but pads its
                ! assigned value to N; record the length so the assignment
                ! lowering knows the fixed width.
                if (character_length > 0 .and. &
                    context%symbols(existing_index)%is_deferred_character) then
                    context%symbols(existing_index)%character_length = &
                        character_length
                end if
                call set_empty(error_msg)
                return
            end if
            if (context%symbols(existing_index)%is_parameter) then
                call declaration_character_length(context, node, &
                                                  character_length, error_msg)
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
        call declaration_character_length(context, node, character_length, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        call define_character_symbol(context, name, character_length, error_msg)
    end subroutine define_declared_character_symbol

    subroutine declaration_character_length(context, node, character_length, &
                                            error_msg)
        type(lowering_context_t), intent(in) :: context
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
        if (len_trim(parse_error) == 0) return

        ! A non-literal length such as character(max_len) keeps the length
        ! text in character_length_expr. Resolve a named integer parameter
        ! against the symbol table before reporting it unsupported.
        call resolve_named_character_length(context, node, character_length, &
                                            parse_error)
        if (len_trim(parse_error) > 0) then
            call unsupported_feature_error('character length declaration', &
                                           node%line, node%column, &
                                           trim(parse_error), error_msg)
        end if
    end subroutine declaration_character_length

    ! Resolve character(<name>) where <name> is a compile-time integer
    ! parameter already in the symbol table. Leaves error_msg set when the
    ! length text is not a single resolvable parameter name.
    subroutine resolve_named_character_length(context, node, character_length, &
                                              error_msg)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(out) :: character_length
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index

        character_length = -1
        error_msg = 'character length must be a positive integer literal'
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        name = trim(adjustl(node%character_length_expr))
        if (len(name) == 0) return
        if (verify(name, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%has_i32_constant) return
        if (context%symbols(symbol_index)%i32_constant <= 0) return
        character_length = int(context%symbols(symbol_index)%i32_constant)
        call set_empty(error_msg)
    end subroutine resolve_named_character_length

    ! True when the character length is a simple runtime identifier (such as a
    ! dummy argument) rather than a literal or compile-time parameter. Such a
    ! length is only valid for a function result here, lowered through the
    ! deferred descriptor ABI.
    logical function runtime_character_length(context, node)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable :: name
        integer :: symbol_index

        runtime_character_length = .false.
        if (.not. node%has_character_length) return
        if (.not. allocated(node%character_length_expr)) return
        name = trim(adjustl(node%character_length_expr))
        if (len(name) == 0) return
        if (verify(name, &
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') &
            /= 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) return
        ! A declared identifier that is not a compile-time integer parameter is
        ! a runtime length.
        runtime_character_length = &
            .not. context%symbols(symbol_index)%has_i32_constant
    end function runtime_character_length

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

        ! A fixed-length scalar may receive the result of a character-returning
        ! contained function (e.g. character(len=5) :: r; r = make(...)). The
        ! callee returns a {data,len} descriptor; bind the destination's value
        ! to the returned data so later uses (print) see the result text.
        if (.not. context%symbols(symbol_index)%is_deferred_character) then
            if (node_exists(arena, node%value_index)) then
                select type (val => arena%entries(node%value_index)%node)
                type is (call_or_subscript_node)
                    if (.not. val%is_array_access .and. allocated(val%name)) then
                        if (is_contained_deferred_char_function(context, &
                            val%name)) then
                            call lower_fixed_char_result_call(arena, node, &
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
                ! A // chain with variable or intrinsic operands concatenates at
                ! runtime into a fresh buffer bound to the target's value.
                call lower_fixed_concat_assignment(arena, node, context, &
                                                   symbol_index, error_msg)
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
        type(lr_operand_desc_t) :: descriptor
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

        ! One contiguous 16-byte descriptor {i8* data; i64 length}, so the
        ! data slot is at offset 0 and the length at offset 8. This matches the
        ! parameter ABI (bind_character_parameter_symbol), so the descriptor's
        ! address can be passed for an intent(out)/(inout) dummy. Both fields
        ! start zeroed for a well-defined unallocated state.
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 16_c_int64_t), descriptor, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, 0_c_int64_t, &
                                  data_addr, error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, 8_c_int64_t, &
                                  len_addr, error_msg)) return
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

    recursive subroutine char_expr_operands(arena, node_index, context, &
                                            data_ptr, length, error_msg)
        ! Resolve a scalar character expression (a variable, a // concatenation,
        ! or a character intrinsic such as trim()) to its data pointer and i32
        ! length. Concatenations and trim() materialise a fresh null-terminated
        ! buffer holding the result text.
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
        if (is_binary_op(arena, node_index)) then
            call concat_char_operands(arena, node_index, context, data_ptr, &
                                      length, error_msg)
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
        if (is_named_char_call(arena, node_index, 'repeat')) then
            select type (n => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_repeat_deferred(arena, n, context, data_ptr, length, &
                                           error_msg)
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
                            is_named_char_call(arena, node_index, 'adjustr') .or. &
                            is_named_char_call(arena, node_index, 'repeat')
    end function is_char_expr_call

    recursive logical function is_character_concat(arena, node_index, context) &
            result(is_concat)
        ! True when node is a // binary op whose operands resolve to character
        ! data (a literal, a character variable, a character intrinsic, or a
        ! nested // chain). Used to route print/assign through the runtime
        ! concatenation path instead of the integer fallback.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: bin_op, bin_err
        integer :: left_idx, right_idx, bin_line, bin_col

        is_concat = .false.
        if (.not. node_exists(arena, node_index)) return
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, left_idx, right_idx, &
                                bin_line, bin_col, bin_err)
        if (len_trim(bin_err) > 0) return
        if (trim(bin_op) /= '//') return
        is_concat = is_character_operand(arena, left_idx, context) .or. &
                    is_character_operand(arena, right_idx, context)
    end function is_character_concat

    recursive logical function is_character_operand(arena, node_index, context) &
            result(is_char)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: symbol_index

        is_char = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_literal(arena, node_index)) then
            is_char = is_character_literal(arena, node_index)
            return
        end if
        if (is_char_expr_call(arena, node_index)) then
            is_char = .true.
            return
        end if
        if (is_character_concat(arena, node_index, context)) then
            is_char = .true.
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) &
                is_char = context%symbols(symbol_index)%value_kind == &
                          VALUE_CHARACTER
        end if
    end function is_character_operand

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

    include 'session_program_lowering_character_tail.inc'
