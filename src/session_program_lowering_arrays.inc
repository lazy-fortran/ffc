    subroutine get_array_bounds(node, context, array_lower_bound, array_size, &
                                error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: array_lower_bound
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim_count
        integer :: dim_index
        integer(c_int64_t) :: extent
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value
        integer :: i

        array_lower_bound = 1
        array_size = 0
        call set_empty(error_msg)
        if (.not. allocated(node%dimension_indices)) return

        dim_count = size(node%dimension_indices)
        if (dim_count < 1 .or. dim_count > 2) then
            call unsupported_feature_error('array declaration', node%line, &
                                           node%column, &
                                           'direct LIRIC session supports '// &
                                           'at most rank-2 arrays', error_msg)
            return
        end if

        do i = 1, dim_count
            dim_index = node%dimension_indices(i)
            if (.not. node_exists(context%arena, dim_index)) then
                error_msg = 'array dimension index does not reference an AST node'
                return
            end if

            select type (dim_node => context%arena%entries(dim_index)%node)
            type is (range_expression_node)
                if (dim_node%stride_index > 0) then
                    call unsupported_feature_error('array declaration', node%line, &
                                                   node%column, &
                                                   'array declaration bounds do not '// &
                                                   'support strides', error_msg)
                    return
                end if
                if (dim_node%start_index <= 0 .or. dim_node%end_index <= 0) then
                    call unsupported_feature_error('array declaration', node%line, &
                                                   node%column, &
                                                   'array declaration requires '// &
                                                   'explicit lower and upper bounds', &
                                                   error_msg)
                    return
                end if
                call eval_i32_constant(context%arena, dim_node%start_index, context, &
                                       lower_value, error_msg)
                if (len_trim(error_msg) > 0) return
                call eval_i32_constant(context%arena, dim_node%end_index, context, &
                                       upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                extent = upper_value - lower_value + 1_c_int64_t
            class default
                call eval_i32_constant(context%arena, dim_index, context, &
                                       upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                lower_value = 1_c_int64_t
                extent = upper_value
            end select

            if (extent <= 0_c_int64_t) then
                call unsupported_feature_error('array declaration', node%line, &
                                               node%column, &
                                               'array bounds produce a '// &
                                               'non-positive extent', error_msg)
                return
            end if

            if (i == 1) then
                array_lower_bound = int(lower_value)
                array_size = int(extent)
            else
                array_size = array_size * int(extent)
            end if
        end do
        call set_empty(error_msg)
    end subroutine get_array_bounds

    recursive subroutine eval_i32_constant(arena, node_index, context, &
                                           constant_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: lhs
        integer(c_int64_t) :: rhs
        integer :: symbol_index
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

        constant_value = 0_c_int64_t
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'constant expression index does not reference an AST node'
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, bin_left, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_i32_constant(arena, bin_right, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (trim(bin_op))
            case ('+')
                constant_value = lhs + rhs
            case ('-')
                constant_value = lhs - rhs
            case ('*')
                constant_value = lhs*rhs
            case ('/')
                if (rhs == 0_c_int64_t) then
                    error_msg = 'integer constant division by zero'
                    return
                end if
                constant_value = lhs/rhs
            case default
                error_msg = 'unsupported integer constant operator: '// &
                            trim(bin_op)
                return
            end select
            call set_empty(error_msg)
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_i32_literal(lit_value, constant_value, error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'compile-time integer parameter was not declared: '// &
                            trim(id_name)
                return
            end if
            if (.not. context%symbols(symbol_index)%has_i32_constant) then
                error_msg = 'integer identifier is not a compile-time parameter: '// &
                            trim(id_name)
                return
            end if
            constant_value = context%symbols(symbol_index)%i32_constant
            call set_empty(error_msg)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            call eval_kind_inquiry_constant(arena, node, context, &
                                            constant_value, error_msg)
        class default
            error_msg = 'array bound is not a compile-time integer expression'
        end select
    end subroutine eval_i32_constant

    ! Constant-fold selected_int_kind / selected_real_kind so they may appear
    ! in a compile-time integer parameter initializer (kind selection).
    subroutine eval_kind_inquiry_constant(arena, node, context, constant_value, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: constant_value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: p, r

        constant_value = 0_c_int64_t
        if (.not. allocated(node%name)) then
            error_msg = 'compile-time call has no callee name'
            return
        end if
        if (same_name(node%name, 'selected_int_kind')) then
            if (.not. allocated(node%arg_indices) .or. &
                size(node%arg_indices) /= 1) then
                error_msg = 'selected_int_kind requires one argument'
                return
            end if
            call eval_i32_constant(arena, node%arg_indices(1), context, r, error_msg)
            if (len_trim(error_msg) > 0) return
            constant_value = selected_int_kind_value(r)
            call set_empty(error_msg)
        else if (same_name(node%name, 'selected_real_kind')) then
            if (.not. allocated(node%arg_indices) .or. &
                size(node%arg_indices) < 1 .or. size(node%arg_indices) > 2) then
                error_msg = 'selected_real_kind requires one or two arguments'
                return
            end if
            p = 0_c_int64_t
            r = 0_c_int64_t
            call eval_i32_constant(arena, node%arg_indices(1), context, p, error_msg)
            if (len_trim(error_msg) > 0) return
            if (size(node%arg_indices) >= 2) then
                call eval_i32_constant(arena, node%arg_indices(2), context, r, &
                                       error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            constant_value = selected_real_kind_value(p, r)
            call set_empty(error_msg)
        else
            error_msg = 'compile-time integer expression uses unsupported call: '// &
                        trim(node%name)
        end if
    end subroutine eval_kind_inquiry_constant

    subroutine define_declared_array_symbol(context, node, name, array_lower_bound, &
                                            array_size, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: array_lower_bound
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index
        integer :: dim_count
        integer :: i
        integer :: dim_index
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value
        integer(c_int64_t) :: extent

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate array declaration: '//trim(name)
            return
        end if

        dim_count = 0
        if (allocated(node%dimension_indices)) then
            dim_count = size(node%dimension_indices)
        end if

        call grow_symbols(context)

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%is_array = .true.
        context%symbols(index)%array_rank = dim_count
        context%symbols(index)%array_size = array_size
        context%symbols(index)%array_lower_bound = array_lower_bound

        if (dim_count >= 1 .and. allocated(node%dimension_indices)) then
            do i = 1, dim_count
                dim_index = node%dimension_indices(i)
                if (.not. node_exists(context%arena, dim_index)) cycle
                select type (dim_node => context%arena%entries(dim_index)%node)
                type is (range_expression_node)
                    call eval_i32_constant(context%arena, dim_node%start_index, context, &
                                           lower_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call eval_i32_constant(context%arena, dim_node%end_index, context, &
                                           upper_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    context%symbols(index)%array_dim_lowers(i) = int(lower_value)
                    extent = upper_value - lower_value + 1_c_int64_t
                    context%symbols(index)%array_dim_sizes(i) = int(extent)
                class default
                    context%symbols(index)%array_dim_lowers(i) = 1
                    call eval_i32_constant(context%arena, dim_index, context, &
                                           upper_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    context%symbols(index)%array_dim_sizes(i) = int(upper_value)
                end select
            end do
        end if

        if (.not. emit_i32_array_alloca(context%session,  &
            array_size, context%symbols(index)%element_address, error_msg)) then
            error_msg = 'decl_array['//trim(name)//']: '//error_msg
            return
        end if

        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_declared_array_symbol

    subroutine lower_array_constructor_assignment(arena, ctor, symbol_index, &
                                                  context, error_msg)
        ! arr = [a, b, c] for any fixed-size integer array: lower each element
        ! and store it at the matching slot in linear storage order.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: addr
        integer :: i
        integer :: n

        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'array constructor has no elements'
            return
        end if
        n = size(ctor%element_indices)
        if (n /= context%symbols(symbol_index)%array_size) then
            error_msg = 'array constructor element count does not match the '// &
                'declared size of '//trim(context%symbols(symbol_index)%name)
            return
        end if
        do i = 1, n
            call lower_i32_expression(arena, ctor%element_indices(i), context, &
                                      value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, int(i - 1, c_int64_t)), addr, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) &
                return
        end do
        call set_empty(error_msg)
    end subroutine lower_array_constructor_assignment

    subroutine lower_array_whole_assignment(arena, node, symbol_index, context, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: addr
        integer :: i
        integer :: array_size

        array_size = context%symbols(symbol_index)%array_size
        if (array_size <= 0) then
            error_msg = 'whole-array assignment requires a positive array size'
            return
        end if

        if (node_exists(arena, node%value_index)) then
            select type (rhs => arena%entries(node%value_index)%node)
            type is (call_or_subscript_node)
                if (allocated(rhs%name)) then
                    if (same_name(rhs%name, 'shape')) then
                        call lower_shape_intrinsic_assignment(arena, node, &
                                                              symbol_index, context, &
                                                              error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'matmul')) then
                        call lower_matmul_assignment(arena, node, symbol_index, &
                                                     context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'transpose')) then
                        call lower_transpose_assignment(arena, node, symbol_index, &
                                                        context, error_msg)
                        return
                    end if
                    if (same_name(rhs%name, 'reshape')) then
                        call lower_reshape_assignment(arena, node, symbol_index, &
                                                      context, error_msg)
                        return
                    end if
                end if
            end select
        end if

        do i = 1, array_size
            call lower_array_elementwise_value(arena, node%value_index, &
                                               symbol_index, i - 1, context, &
                                               value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_array_element_addr(context%session, array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, int(i - 1, c_int64_t)), addr, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) &
                return
        end do
        call set_empty(error_msg)
    end subroutine lower_array_whole_assignment

    subroutine lower_shape_intrinsic_assignment(arena, node, symbol_index, &
                                                context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: source_rank
        integer :: i

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 1) then
                error_msg = 'shape requires exactly one array argument'
                return
            end if
            call get_identifier_name(arena, rhs%arg_indices(1), source_name, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            source_index = find_symbol(context, source_name)
            if (source_index <= 0) then
                error_msg = 'shape argument was not declared: '//trim(source_name)
                return
            end if
            if (.not. context%symbols(source_index)%is_array) then
                error_msg = 'shape requires an array argument: '//trim(source_name)
                return
            end if
            if (context%symbols(source_index)%value_kind /= VALUE_I32) then
                error_msg = 'shape only supports integer arrays'
                return
            end if
        class default
            error_msg = 'shape requires an array argument'
            return
        end select

        source_rank = context%symbols(source_index)%array_rank
        if (context%symbols(symbol_index)%array_rank /= 1) then
            error_msg = 'shape result must be assigned to a rank-1 integer array'
            return
        end if
        if (context%symbols(symbol_index)%array_size /= source_rank) then
            error_msg = 'shape result length does not match the target array size'
            return
        end if

        do i = 1, source_rank
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, int(i - 1, c_int64_t)), addr, &
                    error_msg)) return
            value = i32_immediate(context%session, &
                                  int(context%symbols(source_index)%array_dim_sizes(i), &
                                      c_int64_t))
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) &
                return
        end do
        call set_empty(error_msg)
    end subroutine lower_shape_intrinsic_assignment

    subroutine lower_shape_intrinsic_element(arena, node, target_symbol_index, &
                                             linear_index, context, value, &
                                             error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: source_rank

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'shape requires exactly one array argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), source_name, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, source_name)
        if (source_index <= 0) then
            error_msg = 'shape argument was not declared: '//trim(source_name)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = 'shape requires an array argument: '//trim(source_name)
            return
        end if
        if (context%symbols(source_index)%value_kind /= VALUE_I32) then
            error_msg = 'shape only supports integer arrays'
            return
        end if

        source_rank = context%symbols(source_index)%array_rank
        if (context%symbols(target_symbol_index)%array_rank /= 1) then
            error_msg = 'shape result must be assigned to a rank-1 integer array'
            return
        end if
        if (context%symbols(target_symbol_index)%array_size /= source_rank) then
            error_msg = 'shape result length does not match the target array size'
            return
        end if
        if (linear_index < 0 .or. linear_index >= source_rank) then
            error_msg = 'shape result index is out of range'
            return
        end if

        value = i32_immediate(context%session, &
                              int(context%symbols(source_index)%array_dim_sizes( &
                                  linear_index + 1), c_int64_t))
        call set_empty(error_msg)
    end subroutine lower_shape_intrinsic_element

    subroutine lower_array_size_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: array_name
        integer :: source_index

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = 'size requires exactly one array argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, array_name)
        if (source_index <= 0) then
            error_msg = 'size argument was not declared: '//trim(array_name)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = 'size requires an array argument: '//trim(array_name)
            return
        end if
        if (context%symbols(source_index)%value_kind /= VALUE_I32) then
            error_msg = 'size only supports integer arrays'
            return
        end if
        value = i32_immediate(context%session, &
                              int(context%symbols(source_index)%array_size, &
                                  c_int64_t))
        call set_empty(error_msg)
    end subroutine lower_array_size_intrinsic

    subroutine lower_array_reduction_intrinsic(arena, node, context, value, &
                                               error_msg, reduction_name)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: reduction_name
        character(len=:), allocatable :: array_name
        type(lr_operand_desc_t) :: candidate
        type(lr_operand_desc_t) :: next_value
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: selected
        integer :: source_index
        integer :: array_size
        integer :: i
        integer(c_int) :: predicate

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) then
            error_msg = trim(reduction_name)//' requires exactly one array argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, array_name)
        if (source_index <= 0) then
            error_msg = trim(reduction_name)//' argument was not declared: '// &
                        trim(array_name)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = trim(reduction_name)//' requires an array argument: '// &
                        trim(array_name)
            return
        end if
        if (context%symbols(source_index)%value_kind /= VALUE_I32) then
            error_msg = trim(reduction_name)//' only supports integer arrays'
            return
        end if

        array_size = context%symbols(source_index)%array_size
        if (array_size <= 0) then
            error_msg = trim(reduction_name)//' requires a non-empty array'
            return
        end if

        select case (trim(reduction_name))
        case ('sum')
            value = i32_immediate(context%session, 0_c_int64_t)
            do i = 0, array_size - 1
                call load_array_linear_element(context, source_index, &
                                               int(i, c_int64_t), candidate, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, value, &
                                          candidate, next_value, error_msg)) return
                value = next_value
            end do
        case ('product')
            value = i32_immediate(context%session, 1_c_int64_t)
            do i = 0, array_size - 1
                call load_array_linear_element(context, source_index, &
                                               int(i, c_int64_t), candidate, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_MUL, value, &
                                          candidate, next_value, error_msg)) return
                value = next_value
            end do
        case ('maxval')
            predicate = LR_CMP_SGT
            call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                           value, error_msg)
            if (len_trim(error_msg) > 0) return
            do i = 1, array_size - 1
                call load_array_linear_element(context, source_index, &
                                               int(i, c_int64_t), candidate, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, predicate, candidate, &
                        value, condition, error_msg)) return
                call select_value(context, condition, candidate, value, selected, &
                                  error_msg)
                if (len_trim(error_msg) > 0) return
                value = selected
            end do
        case ('minval')
            predicate = LR_CMP_SLT
            call load_array_linear_element(context, source_index, 0_c_int64_t, &
                                           value, error_msg)
            if (len_trim(error_msg) > 0) return
            do i = 1, array_size - 1
                call load_array_linear_element(context, source_index, &
                                               int(i, c_int64_t), candidate, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, predicate, candidate, &
                        value, condition, error_msg)) return
                call select_value(context, condition, candidate, value, selected, &
                                  error_msg)
                if (len_trim(error_msg) > 0) return
                value = selected
            end do
        case default
            error_msg = 'unsupported array reduction intrinsic: '//trim(reduction_name)
            return
        end select

        call set_empty(error_msg)
    end subroutine lower_array_reduction_intrinsic

    subroutine resolve_i32_array_argument(arena, node, arg_pos, context, &
                                          intrinsic_name, arg_role, symbol_index, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: arg_pos
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: intrinsic_name
        character(len=*), intent(in) :: arg_role
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: arg_name

        symbol_index = 0
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < arg_pos) then
            error_msg = trim(intrinsic_name)//' requires '//trim(arg_role)//' arguments'
            return
        end if
        if (.not. is_identifier(arena, node%arg_indices(arg_pos))) then
            error_msg = trim(intrinsic_name)//' requires identifier '// &
                        trim(arg_role)//' arguments'
            return
        end if

        call get_identifier_name(arena, node%arg_indices(arg_pos), arg_name, error_msg)
        if (len_trim(error_msg) > 0) return

        symbol_index = find_symbol(context, arg_name)
        if (symbol_index <= 0) then
            error_msg = trim(intrinsic_name)//' '//trim(arg_role)// &
                        ' was not declared: '//trim(arg_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = trim(intrinsic_name)//' requires array '//trim(arg_role)// &
                        ': '//trim(arg_name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
            error_msg = trim(intrinsic_name)//' only supports integer arrays'
            return
        end if

        call set_empty(error_msg)
    end subroutine resolve_i32_array_argument

    subroutine lower_matmul_assignment(arena, node, symbol_index, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a_value
        type(lr_operand_desc_t) :: b_value
        type(lr_operand_desc_t) :: product
        type(lr_operand_desc_t) :: sum
        type(lr_operand_desc_t) :: next_sum
        integer :: a_index
        integer :: b_index
        integer :: i
        integer :: j
        integer :: k
        integer(c_int64_t) :: a_rows
        integer(c_int64_t) :: a_cols
        integer(c_int64_t) :: b_rows
        integer(c_int64_t) :: b_cols
        integer(c_int64_t) :: target_rows
        integer(c_int64_t) :: target_cols
        integer(c_int64_t) :: a_row
        integer(c_int64_t) :: a_col
        integer(c_int64_t) :: b_row
        integer(c_int64_t) :: b_col
        integer(c_int64_t) :: linear_index
        integer(c_int64_t) :: a_row_base
        integer(c_int64_t) :: a_col_base
        integer(c_int64_t) :: b_row_base
        integer(c_int64_t) :: b_col_base

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%name) .or. .not. same_name(rhs%name, 'matmul')) then
                error_msg = 'matmul requires a matmul intrinsic'
                return
            end if
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 2) then
                error_msg = 'matmul requires exactly two array arguments'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'matmul', &
                                            'left operand', a_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call resolve_i32_array_argument(arena, rhs, 2, context, 'matmul', &
                                            'right operand', b_index, error_msg)
            if (len_trim(error_msg) > 0) return

            if (context%symbols(a_index)%array_rank /= 2) then
                error_msg = 'matmul requires rank-2 left operand arrays'
                return
            end if
            if (context%symbols(b_index)%array_rank /= 2) then
                error_msg = 'matmul requires rank-2 right operand arrays'
                return
            end if
            if (context%symbols(symbol_index)%array_rank /= 2) then
                error_msg = 'matmul result must be assigned to a rank-2 integer array'
                return
            end if

            a_rows = int(context%symbols(a_index)%array_dim_sizes(1), c_int64_t)
            a_cols = int(context%symbols(a_index)%array_dim_sizes(2), c_int64_t)
            b_rows = int(context%symbols(b_index)%array_dim_sizes(1), c_int64_t)
            b_cols = int(context%symbols(b_index)%array_dim_sizes(2), c_int64_t)
            target_rows = int(context%symbols(symbol_index)%array_dim_sizes(1), &
                              c_int64_t)
            target_cols = int(context%symbols(symbol_index)%array_dim_sizes(2), &
                              c_int64_t)
            if (a_cols /= b_rows) then
                error_msg = 'matmul requires conforming matrix operands'
                return
            end if
            if (target_rows /= a_rows .or. target_cols /= b_cols) then
                error_msg = 'matmul result shape does not match the target array'
                return
            end if

            a_row_base = int(context%symbols(a_index)%array_dim_lowers(1), c_int64_t)
            a_col_base = int(context%symbols(a_index)%array_dim_lowers(2), c_int64_t)
            b_row_base = int(context%symbols(b_index)%array_dim_lowers(1), c_int64_t)
            b_col_base = int(context%symbols(b_index)%array_dim_lowers(2), c_int64_t)

            do j = 0, int(target_cols) - 1
                do i = 0, int(target_rows) - 1
                    sum = i32_immediate(context%session, 0_c_int64_t)
                    do k = 0, int(a_cols) - 1
                        a_row = a_row_base + int(i, c_int64_t)
                        a_col = a_col_base + int(k, c_int64_t)
                        b_row = b_row_base + int(k, c_int64_t)
                        b_col = b_col_base + int(j, c_int64_t)
                        call load_i32_array_element(context, a_index, a_row, &
                                                    a_value, error_msg, a_col)
                        if (len_trim(error_msg) > 0) return
                        call load_i32_array_element(context, b_index, b_row, &
                                                    b_value, error_msg, b_col)
                        if (len_trim(error_msg) > 0) return
                        if (.not. emit_i32_binary(context%session, LR_OP_MUL, &
                                                  a_value, b_value, product, &
                                                  error_msg)) return
                        if (.not. emit_i32_binary(context%session, LR_OP_ADD, sum, &
                                                  product, next_sum, error_msg)) return
                        sum = next_sum
                    end do
                    linear_index = int(i, c_int64_t) + target_rows * int(j, c_int64_t)
                    call store_array_linear_element(context, symbol_index, &
                                                    linear_index, sum, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
            call set_empty(error_msg)
        class default
            error_msg = 'matmul requires an intrinsic call'
            return
        end select
    end subroutine lower_matmul_assignment

    subroutine lower_transpose_assignment(arena, node, symbol_index, context, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: source_index
        integer :: i
        integer :: j
        integer(c_int64_t) :: source_rows
        integer(c_int64_t) :: source_cols
        integer(c_int64_t) :: target_rows
        integer(c_int64_t) :: target_cols
        integer(c_int64_t) :: source_row_base
        integer(c_int64_t) :: source_col_base
        integer(c_int64_t) :: source_row
        integer(c_int64_t) :: source_col
        integer(c_int64_t) :: linear_index

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%name) .or. .not. same_name(rhs%name, 'transpose')) then
                error_msg = 'transpose requires a transpose intrinsic'
                return
            end if
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 1) then
                error_msg = 'transpose requires exactly one array argument'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'transpose', &
                                            'source array', source_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(source_index)%array_rank /= 2) then
                error_msg = 'transpose requires a rank-2 source array'
                return
            end if
            if (context%symbols(symbol_index)%array_rank /= 2) then
                error_msg = 'transpose result must be assigned to a rank-2 integer array'
                return
            end if

            source_rows = int(context%symbols(source_index)%array_dim_sizes(1), &
                              c_int64_t)
            source_cols = int(context%symbols(source_index)%array_dim_sizes(2), &
                              c_int64_t)
            target_rows = int(context%symbols(symbol_index)%array_dim_sizes(1), &
                              c_int64_t)
            target_cols = int(context%symbols(symbol_index)%array_dim_sizes(2), &
                              c_int64_t)
            if (target_rows /= source_cols .or. target_cols /= source_rows) then
                error_msg = 'transpose result shape does not match the target array'
                return
            end if

            source_row_base = int(context%symbols(source_index)%array_dim_lowers(1), &
                                   c_int64_t)
            source_col_base = int(context%symbols(source_index)%array_dim_lowers(2), &
                                   c_int64_t)

            do j = 0, int(target_cols) - 1
                do i = 0, int(target_rows) - 1
                    source_row = source_row_base + int(j, c_int64_t)
                    source_col = source_col_base + int(i, c_int64_t)
                    call load_i32_array_element(context, source_index, source_row, &
                                                value, error_msg, source_col)
                    if (len_trim(error_msg) > 0) return
                    linear_index = int(i, c_int64_t) + target_rows * int(j, c_int64_t)
                    call store_array_linear_element(context, symbol_index, &
                                                    linear_index, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
            call set_empty(error_msg)
        class default
            error_msg = 'transpose requires an intrinsic call'
            return
        end select
    end subroutine lower_transpose_assignment

    subroutine lower_reshape_assignment(arena, node, symbol_index, context, &
                                        error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: i
        integer :: shape_rank
        integer(c_int64_t) :: shape_value
        integer(c_int64_t) :: target_size
        integer(c_int64_t) :: source_size
        integer(c_int64_t) :: linear_index

        select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(rhs%name) .or. .not. same_name(rhs%name, 'reshape')) then
                error_msg = 'reshape requires a reshape intrinsic'
                return
            end if
            if (.not. allocated(rhs%arg_indices) .or. size(rhs%arg_indices) /= 2) then
                error_msg = 'reshape requires a source array and a shape array'
                return
            end if
            call resolve_i32_array_argument(arena, rhs, 1, context, 'reshape', &
                                            'source array', source_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(symbol_index)%array_rank < 1 .or. &
                context%symbols(symbol_index)%array_rank > 2) then
                error_msg = 'reshape result must be assigned to a rank-1 or rank-2 integer array'
                return
            end if

            select type (shape_node => arena%entries(rhs%arg_indices(2))%node)
            type is (array_literal_node)
                if (.not. allocated(shape_node%element_indices)) then
                    error_msg = 'reshape requires a non-empty shape literal'
                    return
                end if
                shape_rank = size(shape_node%element_indices)
                if (shape_rank /= context%symbols(symbol_index)%array_rank) then
                    error_msg = 'reshape shape rank does not match the target array'
                    return
                end if
                do i = 1, shape_rank
                    call parse_i32_constant(arena, shape_node%element_indices(i), &
                                            shape_value, 'reshape shape', error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (shape_value /= int(context%symbols(symbol_index)%array_dim_sizes(i), &
                                           c_int64_t)) then
                        error_msg = 'reshape shape does not match the target array'
                        return
                    end if
                end do
            class default
                error_msg = 'reshape requires a literal integer shape argument'
                return
            end select

            source_size = int(context%symbols(source_index)%array_size, c_int64_t)
            target_size = int(context%symbols(symbol_index)%array_size, c_int64_t)
            if (source_size /= target_size) then
                error_msg = 'reshape requires source and target arrays of equal size'
                return
            end if

            do linear_index = 0_c_int64_t, target_size - 1_c_int64_t
                call load_array_linear_element(context, source_index, linear_index, &
                                               value, error_msg)
                if (len_trim(error_msg) > 0) return
                call store_array_linear_element(context, symbol_index, linear_index, &
                                                value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call set_empty(error_msg)
        class default
            error_msg = 'reshape requires an intrinsic call'
            return
        end select
    end subroutine lower_reshape_assignment
