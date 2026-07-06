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

        ! A logical-valued binary op (comparison such as x > 3, a chained
        ! comparison rewritten to 1 < x .and. x < 10, or a logical connective)
        ! reuses the IF-condition lowering, which yields the i1 truth value.
        if (is_binary_op(arena, node_index)) then
            call lower_i1_condition(arena, node_index, context, value, error_msg)
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
            if (node%base_expr_index > 0) then
                call lower_derived_component_element_load(arena, node, context, &
                                                          value, error_msg)
            else if ((node%is_array_access .and. &
                 array_access_value_kind(node, context) == VALUE_LOGICAL) .or. &
                is_allocatable_element_ref(node, context) .or. &
                is_declared_array_element_ref(node, context)) then
                ! Logical array element occupies an i32 slot; the load yields the
                ! 0/1 value the logical-context callers expect. Allocatable
                ! logical refs reuse the same i32-slot load.
                call lower_i32_array_element(arena, node, context, value, &
                                             error_msg)
            else
                call lower_logical_call(arena, node, context, value, error_msg)
            end if
        type is (component_access_node)
            call lower_derived_component_load(arena, node, context, value, error_msg)
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
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)

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
        call call_argument_kinds(arena, node, context, VALUE_LOGICAL, &
                                call_arg_count, call_arg_kinds)
        call call_argument_ranks(arena, node, context, call_arg_count, &
                                 call_arg_ranks)
        call_name = degeneric_call_name(context, node%name, &
            call_arg_count, call_arg_kinds, call_arg_ranks)
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
        if (same_name(call_name, 'allocated') .and. .not. &
            is_contained_logical_function(context, call_name)) then
            call lower_allocated(arena, node%arg_indices, context, value, &
                                 error_msg)
            return
        end if
        if (same_name(call_name, 'btest') .and. .not. &
            is_contained_logical_function(context, call_name)) then
            call lower_btest_intrinsic(arena, node, context, value, error_msg)
            return
        end if
        if ((same_name(call_name, 'all') .or. same_name(call_name, 'any')) &
            .and. .not. is_contained_logical_function(context, call_name)) then
            block
                logical :: handled
                call try_alloc_component_mask_reduction(arena, node, context, &
                    same_name(call_name, 'all'), handled, value, error_msg)
                if (handled .or. len_trim(error_msg) > 0) return
                call try_array_mask_reduction(arena, node, context, &
                    same_name(call_name, 'all'), handled, value, error_msg)
                if (handled .or. len_trim(error_msg) > 0) return
            end block
            if (same_name(call_name, 'all')) then
                call lower_logical_array_all_intrinsic(arena, node, context, &
                                                       value, error_msg)
            else
                call lower_logical_array_any_intrinsic(arena, node, context, &
                                                       value, error_msg)
            end if
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

    ! Mixed int/real arithmetic: an integer operand in a real expression
    ! coerces to the expression's real kind (Fortran type promotion). Detect
    ! integer-typed subexpressions and emit SITOFP rather than rejecting them.
    recursive subroutine lower_f32_operand(arena, node_index, context, &
                                           value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: int_value

        if (is_integer_operand(arena, node_index, context)) then
            call lower_i32_expression(arena, node_index, context, int_value, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_f32(context%session, int_value, value, &
                                            error_msg)) return
            return
        end if
        call lower_f32_expression(arena, node_index, context, value, error_msg)
    end subroutine lower_f32_operand

    recursive subroutine lower_f64_operand(arena, node_index, context, &
                                           value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: int_value

        if (is_integer_operand(arena, node_index, context)) then
            call lower_i32_expression(arena, node_index, context, int_value, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_f64(context%session, int_value, value, &
                                            error_msg)) return
            return
        end if
        ! A default-real (f32) sub-expression used in an f64 context promotes to
        ! f64, matching Fortran's mixed-kind arithmetic (e.g. an f64 result body
        ! that adds an f32 dummy argument).
        if (is_f32_expression(arena, node_index, context)) then
            call lower_f32_expression(arena, node_index, context, int_value, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_to_f64(context%session, int_value, value, &
                                            error_msg)) return
            return
        end if
        call lower_f64_expression(arena, node_index, context, value, error_msg)
    end subroutine lower_f64_operand

    ! True when the node is an integer-typed scalar subexpression: an integer
    ! literal, an i32 identifier, or a binary op whose operands are both
    ! integer. Used to decide whether a real expression's operand needs
    ! int->real coercion. A mixed op (one real operand) is not integer, so the
    ! coercion happens one level down on the integer sub-operand.
    recursive function is_integer_operand(arena, node_index, context) &
        result(is_int)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        logical :: is_int
        integer :: symbol_index, bl, br, blin, bcol
        character(len=:), allocatable :: id_name, id_err, bin_op
        character(len=:), allocatable :: lit_value, lit_type

        is_int = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_literal(arena, node_index)) then
            if (is_real_literal(arena, node_index)) return
            if (is_logical_literal(arena, node_index)) return
            if (is_character_literal(arena, node_index)) return
            call get_literal_info(arena, node_index, lit_value, lit_type, id_err)
            if (len_trim(id_err) > 0) return
            is_int = trim(lit_type) == 'integer'
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) return
            if (context%symbols(symbol_index)%is_array) return
            is_int = context%symbols(symbol_index)%value_kind == VALUE_I32
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bl, br, &
                                    blin, bcol, id_err)
            if (len_trim(id_err) > 0) return
            is_int = is_integer_operand(arena, bl, context) .and. &
                     is_integer_operand(arena, br, context)
        end if
    end function is_integer_operand

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

        if (is_statement_function_call(arena, node_index, context)) then
            call lower_statement_function_call(arena, node_index, VALUE_F32, &
                                               context, value, error_msg)
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_f64_literal(lit_value, context, literal_value_f64, error_msg)
            if (len_trim(error_msg) > 0) return
            literal_value_f32 = real(literal_value_f64, c_float)
            value = liric_f32_immediate(context%session, literal_value_f32)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            block
                integer :: op_slot
                if (overloaded_operator_slot(arena, node_index, context, op_slot)) &
                    then
                    call lower_overloaded_operator(arena, node_index, op_slot, &
                                                   context, value, error_msg)
                    return
                end if
            end block
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_operand(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_operand(arena, bin_right, context, rhs, error_msg)
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
            if (node%base_expr_index > 0) then
                call lower_derived_component_element_load(arena, node, context, &
                                                          value, error_msg)
            else
                call lower_f32_call(arena, node, context, value, error_msg)
            end if
        type is (component_access_node)
            block
                logical :: matched
                call lower_complex_component_access_load(arena, node, context, &
                    .true., value, matched, error_msg)
                if (matched .or. len_trim(error_msg) > 0) return
            end block
            call lower_derived_component_load(arena, node, context, value, error_msg)
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
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)

        if (node%is_array_access .or. is_allocatable_element_ref(node, context) &
            .or. is_declared_array_element_ref(node, context)) then
            call lower_array_element_address_for_kind(arena, node, context, &
                VALUE_F32, element_addr, error_msg, 'array expression')
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
        if (is_complex_component_extract(arena, node, context, VALUE_C4)) then
            call lower_complex_component_extract(arena, node, context, &
                                                 .true., value, error_msg)
            return
        end if

        ! Resolve generic -> specific (#249 B7c).
        call call_argument_kinds(arena, node, context, VALUE_F32, &
                                call_arg_count, call_arg_kinds)
        call call_argument_ranks(arena, node, context, call_arg_count, &
                                 call_arg_ranks)
        call_name = degeneric_call_name(context, node%name, &
            call_arg_count, call_arg_kinds, call_arg_ranks)

        if (same_name(call_name, 'transfer') .and. .not. &
            is_contained_f32_function(context, call_name)) then
            block
                logical :: handled
                call lower_transfer_intrinsic(arena, node, context, VALUE_F32, &
                                              value, handled, error_msg)
                if (handled) return
                if (len_trim(error_msg) > 0) return
            end block
        end if

        intrinsic_id = f64_intrinsic_id(call_name)
        if (intrinsic_id /= F64_INTRINSIC_NONE) then
            call lower_f32_intrinsic_call(arena, node, intrinsic_id, context, &
                                          value, error_msg)
            return
        end if

        if (is_real_inquiry_intrinsic(call_name)) then
            call lower_real_inquiry_intrinsic(arena, node, context, VALUE_F32, &
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
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: id_name, id_err

        is_f32 = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_literal(arena, node_index)) then
            ! A bare real literal ("2.5", "3.0e0") is default real (f32), but an
            ! explicitly-typed f64 literal (d exponent or _8/_real64/… suffix) is
            ! f64 and must not be lowered through the f32 path, which would round
            ! it to single precision.
            if (is_real_literal(arena, node_index)) then
                block
                    character(len=:), allocatable :: lv, lt, le
                    call get_literal_info(arena, node_index, lv, lt, le)
                    is_f32 = .not. literal_is_f64(lv, context)
                end block
            end if
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
            if (node%base_expr_index > 0) then
                ! obj%comp(i): element of an array component.
                is_f32 = component_element_access_kind(arena, node, context) == &
                         VALUE_F32
                return
            end if
            if (is_complex_component_extract(arena, node, context, VALUE_C4)) then
                ! real(z)/aimag(z) on a complex(4) yields a real(4) component.
                is_f32 = .true.
                return
            end if
            if (allocated(node%name)) then
                if (same_name(node%name, 'transfer') .and. &
                    allocated(node%arg_indices)) then
                    if (size(node%arg_indices) == 2) then
                        is_f32 = transfer_operand_kind(arena, &
                            node%arg_indices(2), context) == VALUE_F32
                    end if
                    return
                end if
                if (node%is_array_access .or. &
                    is_declared_array_element_ref(node, context)) then
                    symbol_index = find_symbol(context, node%name)
                    if (symbol_index > 0) then
                        is_f32 = context%symbols(symbol_index)%value_kind == VALUE_F32
                    end if
                else if (f64_intrinsic_id(node%name) == F64_INTRINSIC_REAL) then
                    ! real(A [,KIND]) is the one non-kind-preserving conversion:
                    ! it is f32 whenever its result kind is not f64.
                    is_f32 = .not. real_intrinsic_is_f64(arena, node, context)
                else if (f64_intrinsic_id(node%name) /= F64_INTRINSIC_NONE) then
                    ! Elemental real intrinsics (sqrt, sin, abs, ...) return the
                    ! kind of their argument, so a nested sqrt(sqrt(x)) is f32
                    ! exactly when its argument is. Recurse on the first argument.
                    ! abs() of a complex(4) yields a real(4) magnitude.
                    if (allocated(node%arg_indices)) then
                        if (size(node%arg_indices) >= 1) then
                            if (same_name(node%name, 'abs') .and. &
                                is_complex_valued(arena, node%arg_indices(1), &
                                                  context, VALUE_C4)) then
                                is_f32 = .true.
                            else
                                is_f32 = is_f32_expression(arena, &
                                    node%arg_indices(1), context)
                            end if
                        end if
                    end if
                else if (is_real_inquiry_intrinsic(node%name)) then
                    is_f32 = inquiry_arg_real_kind(arena, node, context) == 4
                else
                    call call_argument_kinds(arena, node, context, VALUE_I32, &
                        call_arg_count, call_arg_kinds)
                    call call_argument_ranks(arena, node, context, call_arg_count, &
                        call_arg_ranks)
                    is_f32 = is_contained_f32_function(context, node%name) .or. &
                        generic_call_return_kind(context, node%name, &
                            call_arg_count, call_arg_kinds, call_arg_ranks) == VALUE_F32
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

        if (is_statement_function_call(arena, node_index, context)) then
            call lower_statement_function_call(arena, node_index, VALUE_F64, &
                                               context, value, error_msg)
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_f64_literal(lit_value, context, literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            value = liric_f64_immediate(context%session, literal_value)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            block
                integer :: op_slot
                if (overloaded_operator_slot(arena, node_index, context, op_slot)) &
                    then
                    call lower_overloaded_operator(arena, node_index, op_slot, &
                                                   context, value, error_msg)
                    return
                end if
            end block
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_operand(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_operand(arena, bin_right, context, rhs, error_msg)
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
            if (node%base_expr_index > 0) then
                call lower_derived_component_element_load(arena, node, context, &
                                                          value, error_msg)
            else
                call lower_f64_call(arena, node, context, value, error_msg)
            end if
        type is (component_access_node)
            block
                logical :: matched
                call lower_complex_component_access_load(arena, node, context, &
                    .false., value, matched, error_msg)
                if (matched .or. len_trim(error_msg) > 0) return
            end block
            call lower_derived_component_load(arena, node, context, value, error_msg)
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
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)

        if (node%is_array_access .or. is_allocatable_element_ref(node, context) &
            .or. is_declared_array_element_ref(node, context)) then
            call lower_array_element_address_for_kind(arena, node, context, &
                VALUE_F64, element_addr, error_msg, 'array expression')
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
        if (is_complex_component_extract(arena, node, context, VALUE_C8)) then
            call lower_complex_component_extract(arena, node, context, &
                                                 .false., value, error_msg)
            return
        end if

        ! Resolve generic -> specific (#249 B7c).
        call call_argument_kinds(arena, node, context, VALUE_F64, &
                                call_arg_count, call_arg_kinds)
        call call_argument_ranks(arena, node, context, call_arg_count, &
                                 call_arg_ranks)
        call_name = degeneric_call_name(context, node%name, &
            call_arg_count, call_arg_kinds, call_arg_ranks)

        if (same_name(call_name, 'transfer') .and. .not. &
            is_contained_f64_function(context, call_name)) then
            block
                logical :: handled
                call lower_transfer_intrinsic(arena, node, context, VALUE_F64, &
                                              value, handled, error_msg)
                if (handled) return
                if (len_trim(error_msg) > 0) return
            end block
        end if

        intrinsic_id = f64_intrinsic_id(call_name)
        if (intrinsic_id /= F64_INTRINSIC_NONE) then
            call lower_f64_intrinsic_call(arena, node, intrinsic_id, context, &
                                          value, error_msg)
            return
        end if

        if (is_real_inquiry_intrinsic(call_name)) then
            call lower_real_inquiry_intrinsic(arena, node, context, VALUE_F64, &
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

    logical function overloaded_operator_slot(arena, node_index, context, slot)
        ! When node_index is a binary op whose operator token is registered as a
        ! user-defined operator overload for the operand kinds, return .true. and
        ! the matching specific slot. Array operands fall through to the array
        ! path (the overload would need an elemental specific, out of scope).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: slot
        character(len=:), allocatable :: bin_op, bin_err
        integer :: bin_left, bin_right, bin_line, bin_col
        integer :: left_kind, right_kind

        slot = 0
        overloaded_operator_slot = .false.
        if (context%operator_count == 0) return
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                bin_right, bin_line, bin_col, bin_err)
        if (len_trim(bin_err) > 0) return
        if (.not. allocated(bin_op)) return
        left_kind = operand_overload_kind(arena, bin_left, context)
        right_kind = operand_overload_kind(arena, bin_right, context)
        slot = find_operator(context, bin_op, left_kind, right_kind, .false.)
        overloaded_operator_slot = slot > 0
    end function overloaded_operator_slot

    integer function operand_overload_kind(arena, node_index, context) &
        result(kind)
        ! Value kind of an operator operand for overload dispatch; a derived
        ! operand reports VALUE_DERIVED so it pairs with a derived dummy.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        kind = VALUE_I32
        if (actual_is_derived(arena, node_index, context)) then
            kind = VALUE_DERIVED
            return
        end if
        kind = expression_value_kind(arena, node_index, context, VALUE_I32)
    end function operand_overload_kind

    subroutine lower_overloaded_operator(arena, node_index, slot, context, &
                                         value, error_msg)
        ! Lower a user-defined binary operator as a call to its specific:
        ! a .op. b -> specific(a, b). Operand ABI reuses prepare_reference_args
        ! (so derived operands pass by reference like any 2-arg call), and the
        ! result is emitted with the specific's return kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, slot
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: bin_op, specific
        integer :: bin_left, bin_right, bin_line, bin_col, ret_kind
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                bin_right, bin_line, bin_col, error_msg)
        if (len_trim(error_msg) > 0) return
        specific = operator_specific_name(context, slot)
        ret_kind = operator_return_kind(context, slot)
        call prepare_reference_args(arena, [bin_left, bin_right], context, &
                                    VALUE_I32, specific, args, copyback_indices, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        if (ret_kind == VALUE_F64) then
            if (.not. emit_liric_f64_call(context%session, &
                    call_emit_name(arena, specific), args, value, error_msg)) return
        else if (ret_kind == VALUE_F32) then
            if (.not. emit_liric_f32_call(context%session, &
                    call_emit_name(arena, specific), args, value, error_msg)) return
        else
            if (.not. emit_i32_call(context%session, &
                    call_emit_name(arena, specific), args, value, error_msg)) return
        end if
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_overloaded_operator

    recursive logical function is_f64_expression(arena, node_index, context) &
        result(is_f64)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        integer :: call_arg_count
        integer :: call_arg_kinds(MAX_PROC_ARGS)
        integer :: call_arg_ranks(MAX_PROC_ARGS)
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
                    is_f64 = literal_is_f64(lv, context)
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
            if (node%base_expr_index > 0) then
                ! obj%comp(i): element of an array component.
                is_f64 = component_element_access_kind(arena, node, context) == &
                         VALUE_F64
                return
            end if
            if (is_complex_component_extract(arena, node, context, VALUE_C8)) then
                ! real(z)/aimag(z) on a complex(8) yields a real(8) component.
                is_f64 = .true.
                return
            end if
            if (allocated(node%name)) then
                if (same_name(node%name, 'transfer') .and. &
                    allocated(node%arg_indices)) then
                    if (size(node%arg_indices) == 2) then
                        is_f64 = transfer_operand_kind(arena, &
                            node%arg_indices(2), context) == VALUE_F64
                    end if
                    return
                end if
                if (node%is_array_access .or. &
                    is_declared_array_element_ref(node, context)) then
                    symbol_index = find_symbol(context, node%name)
                    if (symbol_index > 0) then
                        is_f64 = context%symbols(symbol_index)%value_kind == VALUE_F64
                    end if
                else if (is_real_inquiry_intrinsic(node%name)) then
                    is_f64 = inquiry_arg_real_kind(arena, node, context) == 8
                else
                    call call_argument_kinds(arena, node, context, VALUE_I32, &
                        call_arg_count, call_arg_kinds)
                    call call_argument_ranks(arena, node, context, call_arg_count, &
                        call_arg_ranks)
                    is_f64 = is_contained_f64_function(context, node%name) .or. &
                        generic_call_return_kind(context, node%name, &
                            call_arg_count, call_arg_kinds, call_arg_ranks) == VALUE_F64
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

    logical function operand_has_value_kind(arena, node_index, context, want_kind)
        ! True when node_index is a scalar identifier or an array-element
        ! access whose declared value kind is want_kind. Used to route a
        ! comparison or arithmetic operand of a non-default integer kind
        ! (integer(1)/(2)/(8)) through its matching width path even when the
        ! operand is a subscripted array reference rather than a bare name.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: want_kind
        character(len=:), allocatable :: id_name, err
        integer :: sym

        operand_has_value_kind = .false.
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol(context, id_name)
            operand_has_value_kind = sym > 0 .and. &
                context%symbols(sym)%value_kind == want_kind
            return
        end if
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%base_expr_index > 0) return
            if (.not. (node%is_array_access .or. &
                       is_declared_array_element_ref(node, context))) return
            operand_has_value_kind = &
                array_access_value_kind(node, context) == want_kind
        end select
    end function operand_has_value_kind

    logical function is_i64_binary_op(arena, node_index, context)
        ! Returns true when node_index is a binary op with at least one i64 operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op, err

        is_i64_binary_op = .false.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        is_i64_binary_op = &
            operand_has_value_kind(arena, bin_left, context, VALUE_I64) .or. &
            operand_has_value_kind(arena, bin_right, context, VALUE_I64)
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

        if (is_statement_function_call(arena, node_index, context)) then
            call lower_statement_function_call(arena, node_index, VALUE_I64, &
                                               context, value, error_msg)
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

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%base_expr_index <= 0 .and. &
                (node%is_array_access .or. &
                 is_declared_array_element_ref(node, context))) then
                call lower_i64_array_element(arena, node, context, value, error_msg)
                return
            end if
            if (node%base_expr_index <= 0 .and. allocated(node%name)) then
                if (contained_function_kind(context, node%name) == VALUE_I64) then
                    call lower_i64_call(arena, node, context, value, error_msg)
                    return
                end if
                if (same_name(node%name, 'transfer') .and. &
                    contained_function_kind(context, node%name) == 0) then
                    block
                        logical :: handled
                        call lower_transfer_intrinsic(arena, node, context, &
                            VALUE_I64, value, handled, error_msg)
                        if (handled) return
                        if (len_trim(error_msg) > 0) return
                    end block
                end if
            end if
        end select

        error_msg = 'ffc direct-session lowering only supports integer(8) expressions'
    end subroutine lower_i64_expression

    subroutine lower_i64_call(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_I64, node%name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_i64_call(context%session, &
                                call_emit_name(arena, node%name), &
                                args, value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_i64_call

    subroutine lower_i64_array_element(arena, node, context, value, error_msg)
        ! Read one element of an integer(8) array through an i64-width GEP.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_address
        integer :: symbol_index

        if (.not. allocated(node%name)) then
            error_msg = 'integer(8) array access requires an array name'
            return
        end if
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'integer(8) array identifier was not declared: '// &
                        trim(node%name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_I64) then
            error_msg = 'integer(8) expression used a non-i64 array element: '// &
                        trim(node%name)
            return
        end if
        call lower_i32_array_element_address(arena, node, context, &
                                             element_address, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i64_load(context%session, element_address, value, &
                                error_msg)) then
            error_msg = 'i64_arr_read_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_i64_array_element

    logical function is_i8_binary_op(arena, node_index, context)
        ! Returns true when node_index is a binary op with at least one i8 operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op, err

        is_i8_binary_op = .false.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        is_i8_binary_op = &
            operand_has_value_kind(arena, bin_left, context, VALUE_I8) .or. &
            operand_has_value_kind(arena, bin_right, context, VALUE_I8)
    end function is_i8_binary_op

    logical function is_i16_binary_op(arena, node_index, context)
        ! Returns true when node_index is a binary op with at least one i16 operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op, err

        is_i16_binary_op = .false.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, err)
        if (len_trim(err) > 0) return
        is_i16_binary_op = &
            operand_has_value_kind(arena, bin_left, context, VALUE_I16) .or. &
            operand_has_value_kind(arena, bin_right, context, VALUE_I16)
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

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%base_expr_index <= 0 .and. &
                (node%is_array_access .or. &
                 is_declared_array_element_ref(node, context))) then
                call lower_i8_array_element(arena, node, context, value, error_msg)
                return
            end if
        end select

        error_msg = 'ffc direct-session lowering only supports integer(1) expressions'
    end subroutine lower_i8_expression

    subroutine lower_i8_array_element(arena, node, context, value, error_msg)
        ! Read one element of an integer(1) array through an i8-width GEP.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_address
        integer :: symbol_index

        if (.not. allocated(node%name)) then
            error_msg = 'integer(1) array access requires an array name'
            return
        end if
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'integer(1) array identifier was not declared: '// &
                        trim(node%name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_I8) then
            error_msg = 'integer(1) expression used a non-i8 array element: '// &
                        trim(node%name)
            return
        end if
        call lower_i32_array_element_address(arena, node, context, &
                                             element_address, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i8_load(context%session, element_address, value, &
                               error_msg)) then
            error_msg = 'i8_arr_read_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_i8_array_element

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

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (node%base_expr_index <= 0 .and. &
                (node%is_array_access .or. &
                 is_declared_array_element_ref(node, context))) then
                call lower_i16_array_element(arena, node, context, value, error_msg)
                return
            end if
        end select

        error_msg = 'ffc direct-session lowering only supports integer(2) expressions'
    end subroutine lower_i16_expression

    subroutine lower_i16_array_element(arena, node, context, value, error_msg)
        ! Read one element of an integer(2) array through an i16-width GEP.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_address
        integer :: symbol_index

        if (.not. allocated(node%name)) then
            error_msg = 'integer(2) array access requires an array name'
            return
        end if
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'integer(2) array identifier was not declared: '// &
                        trim(node%name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_I16) then
            error_msg = 'integer(2) expression used a non-i16 array element: '// &
                        trim(node%name)
            return
        end if
        call lower_i32_array_element_address(arena, node, context, &
                                             element_address, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i16_load(context%session, element_address, value, &
                                error_msg)) then
            error_msg = 'i16_arr_read_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_i16_array_element
