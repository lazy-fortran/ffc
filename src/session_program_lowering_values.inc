    subroutine lower_print(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(print_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: expr_count

        if (.not. allocated(node%expression_indices)) then
            call set_empty(error_msg)
            return
        end if

        expr_count = size(node%expression_indices)
        do i = 1, expr_count
            call lower_print_expression_value(arena, node%expression_indices(i), &
                                              context, error_msg)
            if (len_trim(error_msg) > 0) return
            if (i < expr_count) then
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
            else
                if (.not. emit_liric_print_newline(context%session, error_msg)) &
                    return
            end if
        end do

        call set_empty(error_msg)
    end subroutine lower_print

    subroutine lower_print_expression(arena, node_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        real(c_double) :: real_value
        integer :: symbol_index

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
                    call lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f64(context%session, &
                                                   context%f64_print_format_id, &
                                                   value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i32(context%session, &
                                                   context%i32_print_format_id, &
                                                   value, error_msg)) return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    if (.not. context%symbols(symbol_index)%has_character_value) then
                        call unsupported_feature_error( &
                            'character variable print', node%line, node%column, &
                            'direct LIRIC session requires character variables '// &
                            'to be assigned from a literal before printing', &
                            error_msg)
                        return
                    end if
                    if (.not. emit_liric_print_string_operand( &
                        context%session, context%str_print_format_id, &
                        context%symbols(symbol_index)%value, error_msg)) return
                    return
                end if
            end if
        type is (binary_op_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               context%f64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
        type is (call_or_subscript_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64(context%session, &
                                               context%f64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
        type is (literal_node)
            if (is_character_literal(node)) then
                call strip_literal_quotes(node%value, character_value)
                context%string_literal_count = context%string_literal_count + 1
                write (string_name, '(A,I0)') '.ffc.str.', &
                    context%string_literal_count
                if (.not. emit_liric_print_string(context%session, &
                                                  context%str_print_format_id, &
                                                  trim(string_name), &
                                                  character_value, &
                                                  error_msg)) return
                return
            end if
            if (is_logical_literal(node)) then
                value = i32_immediate(context%session, logical_i32_value(node%value))
                if (.not. emit_liric_print_i32(context%session, &
                                               context%i32_print_format_id, &
                                               value, error_msg)) return
                return
            end if
            if (is_real_literal(node)) then
                call parse_f64_literal(node%value, real_value, error_msg)
                if (len_trim(error_msg) > 0) return
                value = liric_f64_immediate(context%session, real_value)
                if (.not. emit_liric_print_f64(context%session, &
                                               context%f64_print_format_id, &
                                               value, error_msg)) return
                return
            end if
        end select

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32(context%session, &
                                       context%i32_print_format_id, value, &
                                       error_msg)) return
    end subroutine lower_print_expression

    subroutine lower_print_expression_value(arena, node_index, context, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: character_value
        character(len=64) :: string_name
        real(c_double) :: real_value
        integer :: symbol_index

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'print expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%value_kind == VALUE_F64) then
                    call lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_f64_value(context%session, &
                                                         context%f64_print_format_id, &
                                                         value, error_msg)) &
                        return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_LOGICAL) then
                    call lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_print_i32_value(context%session, &
                                                         context%i32_print_format_id, &
                                                         value, error_msg)) &
                        return
                    return
                end if
                if (context%symbols(symbol_index)%value_kind == VALUE_CHARACTER) then
                    if (.not. context%symbols(symbol_index)%has_character_value) then
                        call unsupported_feature_error( &
                            'character variable print', node%line, node%column, &
                            'direct LIRIC session requires character variables '// &
                            'to be assigned from a literal before printing', &
                            error_msg)
                        return
                    end if
                    if (.not. emit_liric_print_string_operand_value( &
                        context%session, context%str_print_format_id, &
                        context%symbols(symbol_index)%value, error_msg)) return
                    return
                end if
            end if
        type is (binary_op_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     context%f64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
        type is (call_or_subscript_node)
            if (is_f64_expression(arena, node_index, context)) then
                call lower_f64_expression(arena, node_index, context, value, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     context%f64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
        type is (literal_node)
            if (is_character_literal(node)) then
                call strip_literal_quotes(node%value, character_value)
                context%string_literal_count = context%string_literal_count + 1
                write (string_name, '(A,I0)') '.ffc.str.', &
                    context%string_literal_count
                if (.not. emit_liric_print_string_value(context%session, &
                                                        context%str_print_format_id, &
                                                        trim(string_name), &
                                                        character_value, &
                                                        error_msg)) return
                return
            end if
            if (is_logical_literal(node)) then
                value = i32_immediate(context%session, logical_i32_value(node%value))
                if (.not. emit_liric_print_i32_value(context%session, &
                                                     context%i32_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
            if (is_real_literal(node)) then
                call parse_f64_literal(node%value, real_value, error_msg)
                if (len_trim(error_msg) > 0) return
                value = liric_f64_immediate(context%session, real_value)
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     context%f64_print_format_id, &
                                                     value, error_msg)) return
                return
            end if
        end select

        call lower_i32_expression(arena, node_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_i32_value(context%session, &
                                             context%i32_print_format_id, value, &
                                             error_msg)) return
    end subroutine lower_print_expression_value

    recursive subroutine lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'logical expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            if (.not. is_logical_literal(node)) then
                error_msg = 'direct LIRIC session expected a logical literal'
                return
            end if
            value = i32_immediate(context%session, logical_i32_value(node%value))
            call set_empty(error_msg)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index <= 0) then
                error_msg = 'logical identifier was not declared: '// &
                            trim(node%name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_LOGICAL) then
                error_msg = 'logical expression used non-logical identifier: '// &
                            trim(node%name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_i32_load(context%session,  &
                    context%symbols(symbol_index)%address, value, &
                    error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
        type is (call_or_subscript_node)
            call lower_logical_call(arena, node, context, value, error_msg)
        class default
            error_msg = 'ffc direct-session lowering only supports logical '// &
                        'literals, identifiers, and contained calls'
        end select
    end subroutine lower_logical_expression

    subroutine lower_logical_call(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        if (node%is_array_access) then
            call unsupported_feature_error('array expression', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support reading array elements', &
                                           error_msg)
            return
        end if
        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session logical function call requires a name'
            return
        end if
        if (.not. is_contained_logical_function(context, node%name)) then
            call unsupported_feature_error('scalar logical function call', &
                                           node%line, node%column, &
                                           'direct LIRIC session only '// &
                                           'supports contained logical '// &
                                           'functions', error_msg)
            return
        end if

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_LOGICAL, args, copyback_indices, &
                                        error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_i32_call(context%session, node%name, args, value, &
                                                error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_logical_call

    recursive subroutine lower_f64_expression(arena, node_index, context, &
                                              value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        real(c_double) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'real expression index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            call parse_f64_literal(node%value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = liric_f64_immediate(context%session, literal_value)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index <= 0) then
                error_msg = 'real identifier was not declared: '//trim(node%name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_F64) then
                error_msg = 'real expression used non-real identifier: '// &
                            trim(node%name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_liric_f64_load( &
                    context%session, context%symbols(symbol_index)%address, &
                    value, error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
        type is (binary_op_node)
            call lower_f64_expression(arena, node%left_index, context, lhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, node%right_index, context, rhs, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call real_opcode(node%operator, node%line, node%column, opcode, &
                             error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_binary(context%session, opcode, lhs, rhs, &
                                            value, error_msg)) return
        type is (call_or_subscript_node)
            call lower_f64_call(arena, node, context, value, error_msg)
        class default
            error_msg = 'ffc direct-session lowering only supports real expressions'
        end select
    end subroutine lower_f64_expression

    subroutine lower_f64_call(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: intrinsic_id

        if (node%is_array_access) then
            call unsupported_feature_error('array expression', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support reading array elements', &
                                           error_msg)
            return
        end if
        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session real function call requires a name'
            return
        end if

        intrinsic_id = f64_intrinsic_id(node%name)
        if (intrinsic_id /= F64_INTRINSIC_NONE) then
            call lower_f64_intrinsic_call(arena, node, intrinsic_id, context, &
                                          value, error_msg)
            return
        end if
        if (is_contained_f64_function(context, node%name)) then
            call lower_contained_f64_call(arena, node, context, value, error_msg)
            return
        end if
        if (node%is_intrinsic) then
            call unsupported_intrinsic_error(node, error_msg)
        else
            call unsupported_feature_error('scalar real function call', &
                                           node%line, node%column, &
                                           'direct LIRIC session only '// &
                                           'supports contained real '// &
                                           'functions and supported '// &
                                           'intrinsics', error_msg)
        end if
    end subroutine lower_f64_call

    subroutine lower_contained_f64_call(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F64, args, copyback_indices, &
                                        error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_liric_f64_call(context%session, node%name, args, value, &
                                      error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_contained_f64_call

    recursive logical function is_f64_expression(arena, node_index, context) &
        result(is_f64)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_f64 = .false.
        if (.not. node_exists(arena, node_index)) return

        select type (node => arena%entries(node_index)%node)
        type is (literal_node)
            is_f64 = is_real_literal(node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                is_f64 = context%symbols(symbol_index)%value_kind == VALUE_F64
            end if
        type is (binary_op_node)
            is_f64 = is_f64_expression(arena, node%left_index, context) .or. &
                     is_f64_expression(arena, node%right_index, context)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                is_f64 = is_contained_f64_function(context, node%name)
            end if
            is_f64 = is_f64 .or. is_f64_intrinsic_expression(arena, node, context)
        end select
    end function is_f64_expression

    logical function is_f64_intrinsic_expression(arena, node, context) &
        result(is_f64)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: i
        integer :: intrinsic_id

        is_f64 = .false.
        if (.not. allocated(node%name)) return

        intrinsic_id = f64_intrinsic_id(node%name)
        if (intrinsic_id == F64_INTRINSIC_REAL) then
            is_f64 = .true.
            return
        end if
        if (intrinsic_id == F64_INTRINSIC_NONE) return
        if (.not. allocated(node%arg_indices)) return

        do i = 1, size(node%arg_indices)
            if (is_f64_expression(arena, node%arg_indices(i), context)) then
                is_f64 = .true.
                return
            end if
        end do
    end function is_f64_intrinsic_expression

    subroutine real_opcode(source_op, line, column, opcode, error_msg)
        character(len=*), intent(in) :: source_op
        integer, intent(in) :: line
        integer, intent(in) :: column
        integer, intent(out) :: opcode
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('+')
            opcode = 18
        case ('-')
            opcode = 19
        case ('*')
            opcode = 20
        case ('/')
            opcode = 21
        case default
            call unsupported_feature_error('real operator', line, column, &
                                           'direct LIRIC session supports '// &
                                           '+, -, *, and / for real expressions', &
                                           error_msg)
        end select
    end subroutine real_opcode

    logical function is_real_literal(node)
        type(literal_node), intent(in) :: node

        is_real_literal = .false.
        if (allocated(node%literal_type)) then
            is_real_literal = trim(node%literal_type) == 'real'
        end if
        is_real_literal = is_real_literal .or. index(node%value, '.') > 0 .or. &
                          index(node%value, 'e') > 0 .or. &
                          index(node%value, 'E') > 0
    end function is_real_literal

    logical function is_character_literal(node)
        type(literal_node), intent(in) :: node

        is_character_literal = .false.
        if (allocated(node%literal_type)) then
            is_character_literal = trim(node%literal_type) == 'character'
        end if
        is_character_literal = is_character_literal .or. &
                               starts_with_quote(node%value)
    end function is_character_literal

    logical function is_logical_literal(node)
        type(literal_node), intent(in) :: node
        character(len=:), allocatable :: value

        value = trim(node%value)
        is_logical_literal = value == '.true.' .or. value == '.false.'
        if (allocated(node%literal_type)) then
            is_logical_literal = is_logical_literal .or. &
                                 trim(node%literal_type) == 'logical'
        end if
    end function is_logical_literal

    logical function starts_with_quote(text)
        character(len=*), intent(in) :: text

        starts_with_quote = len_trim(text) >= 2 .and. &
                            (text(1:1) == '"' .or. text(1:1) == "'")
    end function starts_with_quote

    subroutine strip_literal_quotes(text, value)
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: value
        integer :: text_len

        text_len = len_trim(text)
        if (text_len >= 2 .and. &
            ((text(1:1) == '"' .and. text(text_len:text_len) == '"') .or. &
             (text(1:1) == "'" .and. text(text_len:text_len) == "'"))) then
            value = text(2:text_len - 1)
        else
            value = trim(text)
        end if
    end subroutine strip_literal_quotes

    integer(c_int64_t) function logical_i32_value(text) result(value)
        character(len=*), intent(in) :: text

        if (trim(text) == '.true.') then
            value = 1_c_int64_t
        else
            value = 0_c_int64_t
        end if
    end function logical_i32_value

    subroutine parse_f64_literal(text, value, error_msg)
        character(len=*), intent(in) :: text
        real(c_double), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: io_stat

        read (text, *, iostat=io_stat) value
        if (io_stat == 0) then
            call set_empty(error_msg)
        else
            error_msg = 'invalid real literal for direct LIRIC session: '// &
                        trim(text)
        end if
    end subroutine parse_f64_literal
