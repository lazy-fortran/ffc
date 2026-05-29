    logical function is_present_call(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_present_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) is_present_call = same_name(node%name, &
                                                                  'present')
        end select
    end function is_present_call

    subroutine lower_present_condition(arena, node_index, context, value, &
                                       error_msg)
        ! present(arg) (i1) = the dummy's reference pointer is non-null.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg_name
        type(lr_operand_desc_t) :: nullptr
        integer :: symbol_index

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%arg_indices)) then
                error_msg = 'present requires one argument'
                return
            end if
            if (size(node%arg_indices) /= 1) then
                error_msg = 'present takes exactly one argument'
                return
            end if
            call identifier_name(arena, node%arg_indices(1), arg_name, error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            error_msg = 'present requires a dummy argument'
            return
        end select

        symbol_index = find_symbol(context, arg_name)
        if (symbol_index <= 0) then
            error_msg = 'present argument is not a dummy: '//trim(arg_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%has_address) then
            error_msg = 'present requires an optional reference dummy: '// &
                        trim(arg_name)
            return
        end if
        nullptr = context%symbols(symbol_index)%address
        nullptr%kind = LR_OP_KIND_IMM_I64
        nullptr%payload = 0_c_int64_t
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, &
                context%symbols(symbol_index)%address, nullptr, value, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_present_condition

    recursive subroutine lower_logical_expression(arena, node_index, context, &
                                                  value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        character(len=:), allocatable :: lit_value, lit_type

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'logical expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            if (.not. is_logical_literal(arena, node_index)) then
                error_msg = 'direct LIRIC session expected a logical literal'
                return
            end if
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            value = i32_immediate(context%session, logical_i32_value(lit_value))
            call set_empty(error_msg)
            return
        end if

        select type (node => arena%entries(node_index)%node)
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
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'real expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_f64_literal(lit_value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = liric_f64_immediate(context%session, literal_value)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(bin_op) == '**') then
                ! Real exponentiation lowers to a libm pow(f64, f64) call.
                if (.not. emit_liric_f64_call(context%session, 'pow', &
                                              [lhs, rhs], value, error_msg)) &
                    return
                return
            end if
            call real_opcode(bin_op, bin_line, bin_col, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_binary(context%session, opcode, lhs, rhs, &
                                            value, error_msg)) return
            return
        end if

        select type (node => arena%entries(node_index)%node)
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
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col

        is_f64 = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            is_f64 = is_f64_expression(arena, bin_left, context) .or. &
                     is_f64_expression(arena, bin_right, context)
            return
        end if

        if (is_literal(arena, node_index)) then
            is_f64 = is_real_literal(arena, node_index)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                is_f64 = context%symbols(symbol_index)%value_kind == VALUE_F64
            end if
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                is_f64 = is_contained_f64_function(context, node%name)
            end if
            is_f64 = is_f64 .or. is_f64_intrinsic_expression(arena, node, context)
        end select
    end function is_f64_expression
