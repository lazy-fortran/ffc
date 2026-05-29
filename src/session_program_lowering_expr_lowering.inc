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
            if (trim(node%operator) == '**') then
                ! Real exponentiation lowers to a libm pow(f64, f64) call.
                if (.not. emit_liric_f64_call(context%session, 'pow', &
                                              [lhs, rhs], value, error_msg)) &
                    return
                return
            end if
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
