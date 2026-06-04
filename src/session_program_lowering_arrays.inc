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
        class default
            error_msg = 'array bound is not a compile-time integer expression'
        end select
    end subroutine eval_i32_constant

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

    subroutine describe_array_section(arena, slice_node, context, info, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        type(lowering_context_t), intent(in) :: context
        type(array_section_info_t), intent(out) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: source_rank
        integer :: dim
        integer :: bounds_index
        integer :: kept_dim
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value
        integer(c_int64_t) :: stride_value
        integer(c_int64_t) :: scalar_value
        integer(c_int64_t) :: extent_value

        info%source_name = ''
        info%source_index = 0
        info%source_rank = 0
        info%result_rank = 0
        info%kept_dims = 0
        info%keep_dim = .false.
        info%source_lowers = 0_c_int64_t
        info%source_sizes = 0_c_int64_t
        info%section_lowers = 0_c_int64_t
        info%section_uppers = 0_c_int64_t
        info%section_strides = 1_c_int64_t
        info%section_extents = 0_c_int64_t
        info%scalar_indices = 0_c_int64_t

        call identifier_name(arena, slice_node%array_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol(context, source_name)
        if (source_index <= 0) then
            error_msg = 'array section base was not declared: '//trim(source_name)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = 'array section requires an array: '//trim(source_name)
            return
        end if
        if (context%symbols(source_index)%value_kind /= VALUE_I32) then
            error_msg = 'array sections only support integer arrays'
            return
        end if

        source_rank = context%symbols(source_index)%array_rank
        if (source_rank < 1 .or. source_rank > 2) then
            call unsupported_feature_error('array section', slice_node%line, &
                                           slice_node%column, &
                                           'direct LIRIC session supports at '// &
                                           'most rank-2 array sections', error_msg)
            return
        end if
        if (slice_node%num_dimensions > source_rank) then
            call unsupported_feature_error('array section', slice_node%line, &
                                           slice_node%column, &
                                           'array section has too many subscripts', &
                                           error_msg)
            return
        end if

        info%source_name = trim(source_name)
        info%source_index = source_index
        info%source_rank = source_rank
        do dim = 1, source_rank
            info%source_lowers(dim) = int(context%symbols(source_index)%array_dim_lowers(dim), &
                                          c_int64_t)
            info%source_sizes(dim) = int(context%symbols(source_index)%array_dim_sizes(dim), &
                                         c_int64_t)
        end do

        kept_dim = 0
        do dim = 1, source_rank
            if (dim <= slice_node%num_dimensions) then
                bounds_index = slice_node%bounds_indices(dim)
            else
                bounds_index = 0
            end if

            if (bounds_index <= 0) then
                info%keep_dim(dim) = .true.
                info%section_lowers(dim) = info%source_lowers(dim)
                info%section_uppers(dim) = info%source_lowers(dim) + &
                                           info%source_sizes(dim) - 1_c_int64_t
                info%section_strides(dim) = 1_c_int64_t
                info%section_extents(dim) = info%source_sizes(dim)
                kept_dim = kept_dim + 1
                info%kept_dims(kept_dim) = dim
                cycle
            end if

            if (.not. node_exists(arena, bounds_index)) then
                error_msg = 'array section bound index does not reference an AST node'
                return
            end if

            select type (bounds_node => arena%entries(bounds_index)%node)
            type is (range_expression_node)
                call resolve_section_range_bound(arena, context, bounds_node%start_index, &
                                                 info%source_lowers(dim), lower_value, &
                                                 error_msg)
                if (len_trim(error_msg) > 0) return
                call resolve_section_range_bound(arena, context, bounds_node%end_index, &
                                                 info%source_lowers(dim) + &
                                                 info%source_sizes(dim) - 1_c_int64_t, &
                                                 upper_value, error_msg)
                if (len_trim(error_msg) > 0) return
                call resolve_section_stride(arena, context, bounds_node%stride_index, &
                                            stride_value, error_msg)
                if (len_trim(error_msg) > 0) return
            type is (array_bounds_node)
                if (bounds_node%is_assumed_rank .or. bounds_node%is_assumed_size .or. &
                    bounds_node%is_assumed_shape .or. bounds_node%is_deferred_shape) then
                    lower_value = info%source_lowers(dim)
                    upper_value = info%source_lowers(dim) + info%source_sizes(dim) - 1_c_int64_t
                    stride_value = 1_c_int64_t
                else
                    call resolve_section_range_bound(arena, context, &
                                                     bounds_node%lower_bound_index, &
                                                     info%source_lowers(dim), lower_value, &
                                                     error_msg)
                    if (len_trim(error_msg) > 0) return
                    call resolve_section_range_bound(arena, context, &
                                                     bounds_node%upper_bound_index, &
                                                     info%source_lowers(dim) + &
                                                     info%source_sizes(dim) - 1_c_int64_t, &
                                                     upper_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call resolve_section_stride(arena, context, bounds_node%stride_index, &
                                                stride_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            class default
                call resolve_section_scalar(arena, context, bounds_index, scalar_value, &
                                            error_msg)
                if (len_trim(error_msg) > 0) return
                info%keep_dim(dim) = .false.
                info%scalar_indices(dim) = scalar_value
                cycle
            end select

            if (stride_value == 0_c_int64_t) then
                call unsupported_feature_error('array section', slice_node%line, &
                                               slice_node%column, &
                                               'array section stride must be nonzero', &
                                               error_msg)
                return
            end if
            if (stride_value > 0_c_int64_t) then
                if (lower_value > upper_value) then
                    call unsupported_feature_error('array section', slice_node%line, &
                                                   slice_node%column, &
                                                   'array section produces a non-positive extent', &
                                                   error_msg)
                    return
                end if
                extent_value = (upper_value - lower_value) / stride_value + 1_c_int64_t
            else
                if (lower_value < upper_value) then
                    call unsupported_feature_error('array section', slice_node%line, &
                                                   slice_node%column, &
                                                   'array section produces a non-positive extent', &
                                                   error_msg)
                    return
                end if
                extent_value = (lower_value - upper_value) / abs(stride_value) + 1_c_int64_t
            end if
            if (extent_value <= 0_c_int64_t) then
                call unsupported_feature_error('array section', slice_node%line, &
                                               slice_node%column, &
                                               'array section produces a non-positive extent', &
                                               error_msg)
                return
            end if

            info%keep_dim(dim) = .true.
            info%section_lowers(dim) = lower_value
            info%section_uppers(dim) = upper_value
            info%section_strides(dim) = stride_value
            info%section_extents(dim) = extent_value
            kept_dim = kept_dim + 1
            info%kept_dims(kept_dim) = dim
        end do

        info%result_rank = kept_dim
        call set_empty(error_msg)
    end subroutine describe_array_section

    subroutine resolve_section_range_bound(arena, context, node_index, default_value, &
                                           bound_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(in) :: default_value
        integer(c_int64_t), intent(out) :: bound_value
        character(len=:), allocatable, intent(out) :: error_msg

        if (node_index <= 0) then
            bound_value = default_value
            call set_empty(error_msg)
            return
        end if
        call eval_i32_constant(arena, node_index, context, bound_value, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'array section bounds must be compile-time integer expressions'
        end if
    end subroutine resolve_section_range_bound

    subroutine resolve_section_stride(arena, context, node_index, stride_value, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(out) :: stride_value
        character(len=:), allocatable, intent(out) :: error_msg

        if (node_index <= 0) then
            stride_value = 1_c_int64_t
            call set_empty(error_msg)
            return
        end if
        call eval_i32_constant(arena, node_index, context, stride_value, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'array section bounds must be compile-time integer expressions'
        end if
    end subroutine resolve_section_stride

    subroutine resolve_section_scalar(arena, context, node_index, scalar_value, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(out) :: scalar_value
        character(len=:), allocatable, intent(out) :: error_msg

        call eval_i32_constant(arena, node_index, context, scalar_value, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'array section bounds must be compile-time integer expressions'
        end if
    end subroutine resolve_section_scalar

    integer(c_int64_t) function array_section_total_elements(info) result(total)
        type(array_section_info_t), intent(in) :: info
        integer :: i

        total = 1_c_int64_t
        if (info%result_rank <= 0) return
        do i = 1, info%result_rank
            total = total * info%section_extents(info%kept_dims(i))
        end do
    end function array_section_total_elements

    integer(c_int64_t) function array_section_source_linear_index(info, linear_index) &
            result(source_linear)
        type(array_section_info_t), intent(in) :: info
        integer(c_int64_t), intent(in) :: linear_index
        integer(c_int64_t) :: coord1
        integer(c_int64_t) :: coord2
        integer(c_int64_t) :: inner_index
        integer :: inner_dim
        integer :: outer_dim

        source_linear = 0_c_int64_t
        if (info%source_rank == 1) then
            if (info%result_rank >= 1) then
                inner_dim = info%kept_dims(1)
                coord1 = info%section_lowers(inner_dim) + &
                         linear_index * info%section_strides(inner_dim)
            else
                coord1 = info%scalar_indices(1)
            end if
            source_linear = coord1 - info%source_lowers(1)
            return
        end if

        if (info%result_rank == 2) then
            inner_dim = info%kept_dims(1)
            outer_dim = info%kept_dims(2)
            inner_index = mod(linear_index, info%section_extents(inner_dim))
            coord1 = info%section_lowers(inner_dim) + &
                     inner_index * info%section_strides(inner_dim)
            coord2 = info%section_lowers(outer_dim) + &
                     (linear_index / info%section_extents(inner_dim)) * &
                     info%section_strides(outer_dim)
        else if (info%result_rank == 1) then
            if (info%keep_dim(1)) then
                coord1 = info%section_lowers(1) + linear_index * info%section_strides(1)
                coord2 = info%scalar_indices(2)
            else
                coord1 = info%scalar_indices(1)
                coord2 = info%section_lowers(2) + linear_index * info%section_strides(2)
            end if
        else
            coord1 = info%scalar_indices(1)
            coord2 = info%scalar_indices(2)
        end if

        source_linear = (coord1 - info%source_lowers(1)) + &
                        info%source_sizes(1) * (coord2 - info%source_lowers(2))
    end function array_section_source_linear_index

    subroutine lower_array_section_element_from_info(info, linear_index, context, &
                                                     value, error_msg)
        type(array_section_info_t), intent(in) :: info
        integer(c_int64_t), intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: source_linear
        integer(c_int64_t) :: total_elements

        total_elements = array_section_total_elements(info)
        if (linear_index < 0_c_int64_t .or. linear_index >= total_elements) then
            error_msg = 'array section linear index is out of range'
            return
        end if

        source_linear = array_section_source_linear_index(info, linear_index)
        call load_array_linear_element(context, info%source_index, source_linear, &
                                       value, error_msg)
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine lower_array_section_element_from_info

    subroutine ensure_array_section_shapes_match(context, target_symbol_index, info, &
                                                 error_msg)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: target_symbol_index
        type(array_section_info_t), intent(in) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: target_rank
        integer :: i
        character(len=:), allocatable :: target_name

        target_rank = context%symbols(target_symbol_index)%array_rank
        if (target_rank /= info%result_rank) then
            target_name = trim(context%symbols(target_symbol_index)%name)
            error_msg = 'whole-array assignment requires conforming shapes: '// &
                trim(target_name)//' and array section have different ranks'
            return
        end if
        do i = 1, target_rank
            if (context%symbols(target_symbol_index)%array_dim_sizes(i) /= &
                int(info%section_extents(info%kept_dims(i)))) then
                target_name = trim(context%symbols(target_symbol_index)%name)
                error_msg = 'whole-array assignment requires conforming shapes: '// &
                    trim(target_name)//' and array section differ in extent'
                return
            end if
        end do
        call set_empty(error_msg)
    end subroutine ensure_array_section_shapes_match

    subroutine lower_array_section_value(arena, slice_node, target_symbol_index, &
                                        linear_index, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        integer, intent(in) :: target_symbol_index
        integer(c_int64_t), intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info

        call describe_array_section(arena, slice_node, context, info, error_msg)
        if (len_trim(error_msg) > 0) return
        call ensure_array_section_shapes_match(context, target_symbol_index, info, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_array_section_element_from_info(info, linear_index, context, &
                                                   value, error_msg)
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine lower_array_section_value

    subroutine lower_array_section_print(arena, slice_node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: value
        integer(c_int64_t) :: linear_index
        integer(c_int64_t) :: total_elements

        call describe_array_section(arena, slice_node, context, info, error_msg)
        if (len_trim(error_msg) > 0) return
        total_elements = array_section_total_elements(info)
        if (total_elements <= 0_c_int64_t) then
            error_msg = 'array section has no elements'
            return
        end if
        do linear_index = 0_c_int64_t, total_elements - 1_c_int64_t
            if (.not. emit_liric_print_space(context%session, error_msg)) return
            call lower_array_section_element_from_info(info, linear_index, context, &
                                                       value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_print_i32_value(context%session, &
                                                 context%i32_print_format_id, &
                                                 value, error_msg)) return
        end do
        if (.not. emit_liric_print_newline(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_array_section_print

    recursive subroutine lower_array_elementwise_value(arena, node_index, &
                                                       target_symbol_index, &
                                                       linear_index, context, &
                                                       value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        integer :: opcode
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: bin_op
        character(len=:), allocatable :: id_name
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'whole-array expression index does not reference an AST node'
            return
        end if

        select type (slice_node => arena%entries(node_index)%node)
        type is (array_slice_node)
            call lower_array_section_value(arena, slice_node, &
                                           target_symbol_index, &
                                           int(linear_index, c_int64_t), &
                                           context, value, error_msg)
            return
        end select

        select type (call_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(call_node%name)) then
                if (same_name(call_node%name, 'shape')) then
                    call lower_shape_intrinsic_element(arena, call_node, &
                                                       target_symbol_index, &
                                                       linear_index, context, &
                                                       value, error_msg)
                    return
                end if
            else if (call_node%is_intrinsic) then
                call lower_shape_intrinsic_element(arena, call_node, &
                                                   target_symbol_index, &
                                                   linear_index, context, value, &
                                                   error_msg)
                return
            end if
        end select

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'whole-array expression identifier was not declared: '// &
                            trim(id_name)
                return
            end if
            if (.not. context%symbols(symbol_index)%is_array) then
                call unsupported_feature_error('whole-array expression', &
                                               get_node_line(arena, node_index), &
                                               get_node_column(arena, node_index), &
                                               'whole-array arithmetic requires array '// &
                                               'operands', error_msg)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
                error_msg = 'whole-array arithmetic only supports integer arrays'
                return
            end if
            call ensure_array_shapes_match(context, target_symbol_index, &
                                           symbol_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, symbol_index, &
                                           int(linear_index, c_int64_t), value, &
                                           error_msg)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_elementwise_value(arena, bin_left, &
                                               target_symbol_index, linear_index, &
                                               context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_elementwise_value(arena, bin_right, &
                                               target_symbol_index, linear_index, &
                                               context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (trim(bin_op))
            case ('+')
                opcode = LR_OP_ADD
            case ('-')
                opcode = LR_OP_SUB
            case ('*')
                opcode = LR_OP_MUL
            case default
                call unsupported_feature_error('whole-array arithmetic', &
                                               bin_line, bin_col, &
                                               'direct LIRIC session supports only '// &
                                               '+, -, and * for whole-array '// &
                                               'arithmetic', error_msg)
                return
            end select
            if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, value, &
                                      error_msg)) return
            return
        end if

        call unsupported_feature_error('whole-array expression', &
                                       get_node_line(arena, node_index), &
                                       get_node_column(arena, node_index), &
                                       'whole-array assignment requires array '// &
                                       'operands and +, -, or *', error_msg)
    end subroutine lower_array_elementwise_value

    subroutine ensure_array_shapes_match(context, target_symbol_index, &
                                         source_symbol_index, error_msg)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: source_symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: rank
        integer :: i
        character(len=:), allocatable :: target_name
        character(len=:), allocatable :: source_name

        rank = context%symbols(target_symbol_index)%array_rank
        if (context%symbols(source_symbol_index)%array_rank /= rank) then
            target_name = trim(context%symbols(target_symbol_index)%name)
            source_name = trim(context%symbols(source_symbol_index)%name)
            error_msg = 'whole-array assignment requires conforming shapes: '// &
                trim(target_name)//' and '//trim(source_name)//' have different ranks'
            return
        end if
        do i = 1, rank
            if (context%symbols(target_symbol_index)%array_dim_sizes(i) /= &
                context%symbols(source_symbol_index)%array_dim_sizes(i)) then
                target_name = trim(context%symbols(target_symbol_index)%name)
                source_name = trim(context%symbols(source_symbol_index)%name)
                error_msg = 'whole-array assignment requires conforming shapes: '// &
                    trim(target_name)//' and '//trim(source_name)//' differ in extent'
                return
            end if
        end do
        call set_empty(error_msg)
    end subroutine ensure_array_shapes_match

    subroutine load_array_linear_element(context, symbol_index, linear_index, &
                                         value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr

        if (linear_index < 0_c_int64_t .or. &
            linear_index >= int(context%symbols(symbol_index)%array_size, &
                                c_int64_t)) then
            error_msg = 'array linear index is out of range'
            return
        end if
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, linear_index), addr, &
                error_msg)) return
        if (.not. emit_i32_load(context%session, addr, value, error_msg)) then
            error_msg = 'array_element_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine load_array_linear_element

    subroutine store_array_linear_element(context, symbol_index, linear_index, &
                                          value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr

        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, linear_index), addr, &
                error_msg)) return
        if (.not. emit_i32_store(context%session, value, addr, error_msg)) then
            error_msg = 'array_store: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine store_array_linear_element

    subroutine load_i32_array_element(context, symbol_index, index1, value, error_msg, &
                                      index2)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: index1
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t), intent(in), optional :: index2
        integer(c_int64_t) :: linear_index
        integer(c_int64_t) :: lower1
        integer(c_int64_t) :: lower2
        integer(c_int64_t) :: extent1

        select case (context%symbols(symbol_index)%array_rank)
        case (1)
            if (present(index2)) then
                error_msg = 'rank-1 array element requires one subscript'
                return
            end if
            lower1 = int(context%symbols(symbol_index)%array_lower_bound, &
                          c_int64_t)
            linear_index = index1 - lower1
        case (2)
            if (.not. present(index2)) then
                error_msg = 'rank-2 array element requires two subscripts'
                return
            end if
            lower1 = int(context%symbols(symbol_index)%array_dim_lowers(1), &
                          c_int64_t)
            lower2 = int(context%symbols(symbol_index)%array_dim_lowers(2), &
                          c_int64_t)
            extent1 = int(context%symbols(symbol_index)%array_dim_sizes(1), &
                           c_int64_t)
            linear_index = (index1 - lower1) + extent1 * (index2 - lower2)
        case default
            error_msg = 'direct LIRIC session only supports rank-1 and rank-2 arrays'
            return
        end select

        call load_array_linear_element(context, symbol_index, linear_index, value, &
                                       error_msg)
    end subroutine load_i32_array_element

    subroutine lower_array_element_assignment(arena, node, target, context, value, &
                                              error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_address

        call lower_i32_expression(arena, node%value_index, context, value, error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_i32_array_element_address(arena, target, context, &
                                             element_address, error_msg, &
                                             'array assignment target')
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_store(context%session, value, element_address, &
                                                 error_msg)) then
            error_msg = 'arr_assign_store: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_array_element_assignment

    subroutine lower_i32_array_element_address(arena, node, context, &
                                               element_address, error_msg, &
                                               feature)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: feature
        character(len=:), allocatable :: diagnostic_feature
        integer :: symbol_index
        type(lr_operand_desc_t) :: index_value
        type(lr_operand_desc_t) :: offset
        type(lr_operand_desc_t) :: lower_bound_value
        type(lr_operand_desc_t) :: i_offset
        type(lr_operand_desc_t) :: j_offset
        type(lr_operand_desc_t) :: scaled_index
        integer :: rank
        integer :: arg_count

        if (.not. allocated(node%name)) then
            error_msg = 'array access requires an array name'
            return
        end if

        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'array identifier was not declared: '//trim(node%name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = trim(node%name)//' is not an array'
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_I32) then
            error_msg = 'ffc direct-session lowering only supports integer arrays'
            return
        end if

        rank = context%symbols(symbol_index)%array_rank
        if (rank < 1 .or. rank > 2) then
            call unsupported_feature_error('array access', node%line, &
                                           node%column, &
                                           'direct LIRIC session supports '// &
                                           'at most rank-2 arrays', error_msg)
            return
        end if

        diagnostic_feature = 'array expression'
        if (present(feature)) diagnostic_feature = feature
        arg_count = 0
        if (allocated(node%arg_indices)) then
            arg_count = size(node%arg_indices)
        end if
        if (arg_count /= rank) then
            if (rank == 1) then
                call unsupported_feature_error(diagnostic_feature, node%line, &
                                               node%column, &
                                               'array access requires exactly one '// &
                                               'subscript', error_msg)
            else
                call unsupported_feature_error(diagnostic_feature, node%line, &
                                               node%column, &
                                               'array access requires exactly two '// &
                                               'subscripts', error_msg)
            end if
            return
        end if

        call validate_i32_array_subscript(arena, node%arg_indices(1), context, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_i32_expression(arena, node%arg_indices(1), context, &
                                  index_value, error_msg)
        if (len_trim(error_msg) > 0) return

        if (rank == 1) then
            ! Declared lower bound maps to element zero in LIRIC storage.
            lower_bound_value = i32_immediate(context%session,  &
                                int(context%symbols(symbol_index)%array_lower_bound, &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, index_value, &
                                                      lower_bound_value, &
                                                      offset, error_msg)) then
                error_msg = 'arr_element_sub: '//error_msg
                return
            end if

            if (.not. emit_i32_array_element_addr(context%session,  &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                offset, element_address, error_msg)) then
                error_msg = 'arr_element_gep: '//error_msg
                return
            end if
        else
            ! Rank-2 column-major: linear = (i - lower1) + extent1 * (j - lower2)
            call validate_i32_array_subscript(arena, node%arg_indices(2), context, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return

            call lower_i32_expression(arena, node%arg_indices(1), context, &
                                      i_offset, error_msg)
            if (len_trim(error_msg) > 0) return

            lower_bound_value = i32_immediate(context%session,  &
                                int(context%symbols(symbol_index)%array_dim_lowers(1), &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, i_offset, &
                                                      lower_bound_value, &
                                                      offset, error_msg)) then
                error_msg = 'arr2_i_offset: '//error_msg
                return
            end if

            call lower_i32_expression(arena, node%arg_indices(2), context, &
                                      j_offset, error_msg)
            if (len_trim(error_msg) > 0) return

            lower_bound_value = i32_immediate(context%session,  &
                                int(context%symbols(symbol_index)%array_dim_lowers(2), &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, j_offset, &
                                                      lower_bound_value, &
                                                      index_value, error_msg)) then
                error_msg = 'arr2_j_offset: '//error_msg
                return
            end if

            lower_bound_value = i32_immediate(context%session,  &
                                int(context%symbols(symbol_index)%array_dim_sizes(1), &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, index_value, &
                                                      lower_bound_value, &
                                                      scaled_index, error_msg)) then
                error_msg = 'arr2_stride: '//error_msg
                return
            end if

            if (.not. emit_i32_binary(context%session, LR_OP_ADD, offset, &
                                                      scaled_index, &
                                                      index_value, error_msg)) then
                error_msg = 'arr2_add: '//error_msg
                return
            end if

            if (.not. emit_i32_array_element_addr(context%session,  &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                index_value, element_address, error_msg)) then
                error_msg = 'arr2_gep: '//error_msg
                return
            end if
        end if

        call set_empty(error_msg)
    end subroutine lower_i32_array_element_address

    recursive subroutine validate_i32_array_subscript(arena, node_index, context, &
                                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: literal_value
        integer :: opcode
        integer :: symbol_index
        character(len=:), allocatable :: parse_error
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        character(len=:), allocatable :: lit_value, lit_type
        character(len=:), allocatable :: id_name

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'array subscript index does not reference an AST node'
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call validate_i32_array_subscript(arena, bin_left, context, error_msg)
            if (len_trim(error_msg) > 0) return
            call validate_i32_array_subscript(arena, bin_right, context, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_opcode(bin_op, opcode, parse_error)
            if (len_trim(parse_error) > 0) then
                call unsupported_array_subscript(arena, node_index, &
                                                 'non-integer operators', &
                                                 error_msg)
                return
            end if
            call set_empty(error_msg)
            return
        end if

        if (is_literal(arena, node_index)) then
            call get_literal_info(arena, node_index, lit_value, lit_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call parse_i32_literal(lit_value, literal_value, parse_error)
            if (len_trim(parse_error) > 0) then
                call unsupported_array_subscript(arena, node_index, &
                                                 'non-integer literals', error_msg)
                return
            end if
            call set_empty(error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%is_array .or. &
                    context%symbols(symbol_index)%value_kind /= VALUE_I32) then
                    call unsupported_array_subscript(arena, node_index, &
                                                     'non-integer identifiers', &
                                                     error_msg)
                    return
                end if
            end if
            call set_empty(error_msg)
            return
        end if

        select type (subscript => arena%entries(node_index)%node)
        type is (range_expression_node)
            call unsupported_array_subscript(arena, node_index, &
                                             'ranges and slices', error_msg)
        class default
            call set_empty(error_msg)
        end select
    end subroutine validate_i32_array_subscript

    subroutine unsupported_array_subscript(arena, node_index, detail, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: detail
        character(len=:), allocatable, intent(out) :: error_msg

        call unsupported_feature_error('array subscript', &
                                       get_node_line(arena, node_index), &
                                       get_node_column(arena, node_index), &
                                       'direct LIRIC session supports scalar '// &
                                       'integer array subscripts, not '// &
                                       trim(detail), error_msg)
    end subroutine unsupported_array_subscript
