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
                if (allocated(rhs%name) .and. same_name(rhs%name, 'shape')) then
                    call lower_shape_intrinsic_assignment(arena, node, &
                                                          symbol_index, context, &
                                                          error_msg)
                    return
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
                call load_array_linear_element(context, source_index, i, candidate, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, value, &
                                          candidate, next_value, error_msg)) return
                value = next_value
            end do
        case ('maxval')
            predicate = LR_CMP_SGT
            call load_array_linear_element(context, source_index, 0, value, &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
            do i = 1, array_size - 1
                call load_array_linear_element(context, source_index, i, candidate, &
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
            call load_array_linear_element(context, source_index, 0, value, &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
            do i = 1, array_size - 1
                call load_array_linear_element(context, source_index, i, candidate, &
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
            call load_array_linear_element(context, symbol_index, linear_index, &
                                           value, error_msg)
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
        integer, intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr

        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, int(linear_index, c_int64_t)), addr, &
                error_msg)) return
        if (.not. emit_i32_load(context%session, addr, value, error_msg)) then
            error_msg = 'array_element_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine load_array_linear_element

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
