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
