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
        character(len=:), allocatable :: id_name

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

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'logical identifier was not declared: '// &
                            trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_LOGICAL) then
                error_msg = 'logical expression used non-logical identifier: '// &
                            trim(id_name)
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
            return
        end if

        select type (node => arena%entries(node_index)%node)
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
        character(len=:), allocatable :: call_name

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
        ! Resolve generic -> specific (#249 B7c).
        call_name = degeneric_call_name(context, node%name, &
            first_call_arg_kind(arena, node, context, VALUE_LOGICAL))
        if (same_name(call_name, 'c_associated') .and. .not. &
            is_contained_logical_function(context, call_name)) then
            call lower_c_associated(arena, node%arg_indices, context, value, &
                                    error_msg)
            return
        end if
        if (same_name(call_name, 'associated') .and. .not. &
            is_contained_logical_function(context, call_name)) then
            call lower_associated(arena, node%arg_indices, context, value, &
                                  error_msg)
            return
        end if
        if (.not. is_contained_logical_function(context, call_name)) then
            call unsupported_feature_error('scalar logical function call', &
                                           node%line, node%column, &
                                           'direct LIRIC session only '// &
                                           'supports contained logical '// &
                                           'functions', error_msg)
            return
        end if

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_LOGICAL, call_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_i32_call(context%session, call_emit_name(arena, call_name), &
                                args, value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_logical_call

    recursive subroutine lower_f32_expression(arena, node_index, context, &
                                              value, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        real(c_float) :: literal_value_f32
        real(c_double) :: literal_value_f64
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'real expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_f64_literal(lit_value, literal_value_f64, error_msg)
            if (len_trim(error_msg) > 0) return
            literal_value_f32 = real(literal_value_f64, c_float)
            value = liric_f32_immediate(context%session, literal_value_f32)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(bin_op) == '**') then
                ! f32 exponentiation: call libm powf(f32, f32).
                if (.not. emit_liric_f32_call(context%session, 'powf', &
                                              [lhs, rhs], value, error_msg)) &
                    return
                return
            end if
            call real_opcode(bin_op, bin_line, bin_col, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_binary(context%session, opcode, lhs, rhs, &
                                            value, error_msg)) return
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'real identifier was not declared: '//trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_F32) then
                error_msg = 'real(4) expression used non-f32 identifier: '// &
                            trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_liric_f32_load( &
                    context%session, context%symbols(symbol_index)%address, &
                    value, error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            call lower_f32_call(arena, node, context, value, error_msg)
        class default
            error_msg = 'ffc direct-session lowering only supports real(4) expressions'
        end select
    end subroutine lower_f32_expression

    subroutine lower_f32_call(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        type(lr_operand_desc_t) :: element_addr
        integer, allocatable :: copyback_indices(:)
        integer :: intrinsic_id
        character(len=:), allocatable :: call_name

        if (node%is_array_access) then
            call lower_real_array_element_address(arena, node, context, VALUE_F32, &
                                                  element_addr, error_msg, &
                                                  'array expression')
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_load(context%session, element_addr, value, &
                                          error_msg)) then
                error_msg = 'f32_arr_read: '//error_msg
            end if
            return
        end if
        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session real(4) function call requires a name'
            return
        end if

        ! Resolve generic -> specific (#249 B7c).
        call_name = degeneric_call_name(context, node%name, &
            first_call_arg_kind(arena, node, context, VALUE_F32))

        intrinsic_id = f64_intrinsic_id(call_name)
        if (intrinsic_id /= F64_INTRINSIC_NONE) then
            call lower_f32_intrinsic_call(arena, node, intrinsic_id, context, &
                                          value, error_msg)
            return
        end if

        if (is_real_array_reduction(arena, node, context, VALUE_F32)) then
            call lower_array_reduction_intrinsic(arena, node, context, value, &
                                                 error_msg, trim(call_name))
            return
        end if

        if (is_real_dot_product(arena, node, context, VALUE_F32)) then
            call lower_f32_dot_product_intrinsic(arena, node, context, value, &
                                                 error_msg)
            return
        end if

        if (.not. is_contained_f32_function(context, call_name)) then
            call unsupported_feature_error('scalar real(4) function call', &
                                           node%line, node%column, &
                                           'direct LIRIC session only '// &
                                           'supports contained real(4) '// &
                                           'functions and supported intrinsics', &
                                           error_msg)
            return
        end if

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F32, call_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_liric_f32_call(context%session, &
                call_emit_name(arena, call_name), args, value, &
                                      error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_f32_call

    recursive logical function is_f32_expression(arena, node_index, context) &
        result(is_f32)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: id_name, id_err

        is_f32 = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_literal(arena, node_index)) then
            is_f32 = is_real_literal(arena, node_index)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            is_f32 = is_f32_expression(arena, bin_left, context) .or. &
                     is_f32_expression(arena, bin_right, context)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                is_f32 = context%symbols(symbol_index)%value_kind == VALUE_F32
            end if
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                if (node%is_array_access) then
                    symbol_index = find_symbol(context, node%name)
                    if (symbol_index > 0) then
                        is_f32 = context%symbols(symbol_index)%value_kind == VALUE_F32
                    end if
                else
                    is_f32 = is_contained_f32_function(context, node%name) .or. &
                        generic_call_return_kind(context, node%name, &
                            first_call_arg_kind(arena, node, context, &
                                                VALUE_I32)) == VALUE_F32
                    if (.not. is_f32) &
                        is_f32 = is_real_array_reduction(arena, node, context, VALUE_F32)
                    if (.not. is_f32) &
                        is_f32 = is_real_dot_product(arena, node, context, VALUE_F32)
                end if
            end if
        end select
    end function is_f32_expression

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
        character(len=:), allocatable :: id_name

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

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'real identifier was not declared: '//trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_F64) then
                error_msg = 'real expression used non-real identifier: '// &
                            trim(id_name)
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
            return
        end if

        select type (node => arena%entries(node_index)%node)
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
        type(lr_operand_desc_t) :: element_addr
        integer :: intrinsic_id
        character(len=:), allocatable :: call_name

        if (node%is_array_access) then
            call lower_real_array_element_address(arena, node, context, VALUE_F64, &
                                                  element_addr, error_msg, &
                                                  'array expression')
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_load(context%session, element_addr, value, &
                                          error_msg)) then
                error_msg = 'f64_arr_read: '//error_msg
            end if
            return
        end if
        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session real function call requires a name'
            return
        end if

        ! Resolve generic -> specific (#249 B7c).
        call_name = degeneric_call_name(context, node%name, &
            first_call_arg_kind(arena, node, context, VALUE_F64))

        intrinsic_id = f64_intrinsic_id(call_name)
        if (intrinsic_id /= F64_INTRINSIC_NONE) then
            call lower_f64_intrinsic_call(arena, node, intrinsic_id, context, &
                                          value, error_msg)
            return
        end if
        ! dble(x) is identical to real(x, kind=8): promote/convert any type to f64.
        if (same_name(call_name, 'dble') .and. .not. &
            is_contained_f64_function(context, call_name)) then
            call lower_f64_real_intrinsic(arena, node, context, value, error_msg)
            return
        end if
        if (is_real_array_reduction(arena, node, context, VALUE_F64)) then
            call lower_array_reduction_intrinsic(arena, node, context, value, &
                                                 error_msg, trim(call_name))
            return
        end if

        if (is_real_dot_product(arena, node, context, VALUE_F64)) then
            call lower_f64_dot_product_intrinsic(arena, node, context, value, &
                                                 error_msg)
            return
        end if

        if (is_contained_f64_function(context, call_name)) then
            call lower_contained_f64_call_named(arena, node, context, call_name, &
                                                value, error_msg)
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

        call lower_contained_f64_call_named(arena, node, context, node%name, &
                                            value, error_msg)
    end subroutine lower_contained_f64_call

    subroutine lower_contained_f64_call_named(arena, node, context, call_name, &
                                              value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: call_name
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_F64, call_name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_liric_f64_call(context%session, &
                call_emit_name(arena, call_name), args, value, &
                                      error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_contained_f64_call_named

    recursive logical function is_f64_expression(arena, node_index, context) &
        result(is_f64)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: id_name, id_err

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
            ! Only an explicitly-typed f64 literal (d exponent or _8/_real64/…
            ! suffix) is f64.  A bare literal like "2.5" is default real (f32).
            if (is_real_literal(arena, node_index)) then
                block
                    character(len=:), allocatable :: lv, lt, le
                    call get_literal_info(arena, node_index, lv, lt, le)
                    is_f64 = literal_is_f64(lv)
                end block
            end if
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                is_f64 = context%symbols(symbol_index)%value_kind == VALUE_F64
            end if
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                if (node%is_array_access) then
                    symbol_index = find_symbol(context, node%name)
                    if (symbol_index > 0) then
                        is_f64 = context%symbols(symbol_index)%value_kind == VALUE_F64
                    end if
                else
                    is_f64 = is_contained_f64_function(context, node%name) .or. &
                        generic_call_return_kind(context, node%name, &
                            first_call_arg_kind(arena, node, context, &
                                                VALUE_I32)) == VALUE_F64
                    is_f64 = is_f64 .or. is_f64_intrinsic_expression(arena, node, context)
                    if (.not. is_f64) &
                        is_f64 = is_real_array_reduction(arena, node, context, VALUE_F64)
                    if (.not. is_f64) &
                        is_f64 = is_real_dot_product(arena, node, context, VALUE_F64)
                end if
            else
                is_f64 = is_f64_intrinsic_expression(arena, node, context)
            end if
        end select
    end function is_f64_expression

    logical function is_i64_binary_op(arena, node_index, context)
        ! Returns true when node_index is a binary op with at least one i64 operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op, err, id_name
        integer :: sym

        is_i64_binary_op = .false.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        if (is_identifier(arena, bin_left)) then
            call get_identifier_name(arena, bin_left, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_I64) then
                is_i64_binary_op = .true.
                return
            end if
        end if
        if (is_identifier(arena, bin_right)) then
            call get_identifier_name(arena, bin_right, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_I64) then
                is_i64_binary_op = .true.
                return
            end if
        end if
    end function is_i64_binary_op

    recursive subroutine lower_i64_expression(arena, node_index, context, &
                                               value, error_msg)
        ! Lower an expression expected to produce an integer(8)/i64 result.
        ! Literals are parsed as i32 then zero-extended; binary ops use
        ! emit_i64_binary. i32 identifiers are widened via emit_liric_i32_to_i64.
        use liric_session_bindings, only: LR_OP_ADD
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name
        integer :: uscore_pos

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
            if (len_trim(error_msg) > 0) return
            ! Strip Fortran kind suffix (e.g. 42_8, 100_int64) before parsing.
            uscore_pos = index(lit_value, '_')
            if (uscore_pos > 0) lit_value = lit_value(1:uscore_pos - 1)
            call parse_i32_literal(lit_value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = i64_immediate(context%session, literal_value)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i64_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i64_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_opcode(bin_op, opcode, error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('integer(8) operator', &
                                               bin_line, bin_col, &
                                               'direct LIRIC session supports '// &
                                               '+, -, *, and / for integer(8) '// &
                                               'expressions', error_msg)
                return
            end if
            if (.not. emit_i64_binary(context%session, opcode, lhs, rhs, value, &
                                      error_msg)) return
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'integer(8) identifier was not declared: '//trim(id_name)
                return
            end if
            ! Widen an i32 symbol to i64 (e.g. do-variable used in i64 context).
            if (context%symbols(symbol_index)%value_kind == VALUE_I32) then
                call lower_i32_expression(arena, node_index, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_to_i64(context%session, lhs, value, &
                                                 error_msg)) return
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_I64) then
                error_msg = 'integer(8) expression used non-i64 identifier: '// &
                            trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_i64_load(context%session, &
                    context%symbols(symbol_index)%address, value, &
                    error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
            return
        end if

        error_msg = 'ffc direct-session lowering only supports integer(8) expressions'
    end subroutine lower_i64_expression

    logical function is_i8_binary_op(arena, node_index, context)
        ! Returns true when node_index is a binary op with at least one i8 operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op, err, id_name
        integer :: sym

        is_i8_binary_op = .false.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        if (is_identifier(arena, bin_left)) then
            call get_identifier_name(arena, bin_left, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_I8) then
                is_i8_binary_op = .true.
                return
            end if
        end if
        if (is_identifier(arena, bin_right)) then
            call get_identifier_name(arena, bin_right, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_I8) then
                is_i8_binary_op = .true.
                return
            end if
        end if
    end function is_i8_binary_op

    logical function is_i16_binary_op(arena, node_index, context)
        ! Returns true when node_index is a binary op with at least one i16 operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op, err, id_name
        integer :: sym

        is_i16_binary_op = .false.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        if (is_identifier(arena, bin_left)) then
            call get_identifier_name(arena, bin_left, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_I16) then
                is_i16_binary_op = .true.
                return
            end if
        end if
        if (is_identifier(arena, bin_right)) then
            call get_identifier_name(arena, bin_right, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym > 0 .and. context%symbols(sym)%value_kind == VALUE_I16) then
                is_i16_binary_op = .true.
                return
            end if
        end if
    end function is_i16_binary_op

    recursive subroutine lower_i8_expression(arena, node_index, context, &
                                              value, error_msg)
        ! Lower an expression expected to produce an integer(1)/i8 result.
        use liric_session_bindings, only: LR_OP_ADD
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name
        integer :: uscore_pos

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
            if (len_trim(error_msg) > 0) return
            uscore_pos = index(lit_value, '_')
            if (uscore_pos > 0) lit_value = lit_value(1:uscore_pos - 1)
            call parse_i32_literal(lit_value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = i8_immediate(context%session, literal_value)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i8_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i8_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_opcode(bin_op, opcode, error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('integer(1) operator', &
                                               bin_line, bin_col, &
                                               'direct LIRIC session supports '// &
                                               '+, -, *, and / for integer(1) '// &
                                               'expressions', error_msg)
                return
            end if
            if (.not. emit_i8_binary(context%session, opcode, lhs, rhs, value, &
                                     error_msg)) return
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'integer(1) identifier was not declared: '//trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_I8) then
                error_msg = 'integer(1) expression used non-i8 identifier: '// &
                            trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_i8_load(context%session, &
                    context%symbols(symbol_index)%address, value, &
                    error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
            return
        end if

        error_msg = 'ffc direct-session lowering only supports integer(1) expressions'
    end subroutine lower_i8_expression

    recursive subroutine lower_i16_expression(arena, node_index, context, &
                                               value, error_msg)
        ! Lower an expression expected to produce an integer(2)/i16 result.
        use liric_session_bindings, only: LR_OP_ADD
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: literal_value
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs
        integer :: opcode
        integer :: symbol_index
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name
        integer :: uscore_pos

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'expression index does not reference an AST node'
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, error_msg)
            if (len_trim(error_msg) > 0) return
            uscore_pos = index(lit_value, '_')
            if (uscore_pos > 0) lit_value = lit_value(1:uscore_pos - 1)
            call parse_i32_literal(lit_value, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = i16_immediate(context%session, literal_value)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i16_expression(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i16_expression(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_opcode(bin_op, opcode, error_msg)
            if (len_trim(error_msg) > 0) then
                call unsupported_feature_error('integer(2) operator', &
                                               bin_line, bin_col, &
                                               'direct LIRIC session supports '// &
                                               '+, -, *, and / for integer(2) '// &
                                               'expressions', error_msg)
                return
            end if
            if (.not. emit_i16_binary(context%session, opcode, lhs, rhs, value, &
                                      error_msg)) return
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'integer(2) identifier was not declared: '//trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_I16) then
                error_msg = 'integer(2) expression used non-i16 identifier: '// &
                            trim(id_name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_i16_load(context%session, &
                    context%symbols(symbol_index)%address, value, &
                    error_msg)) return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
            return
        end if

        error_msg = 'ffc direct-session lowering only supports integer(2) expressions'
    end subroutine lower_i16_expression
