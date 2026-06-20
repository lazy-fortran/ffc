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
        select case (context%symbols(source_index)%value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64)
            ! supported element types for array sections
        case default
            error_msg = 'array sections support integer, real(4), and real(8) arrays'
            return
        end select

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
        integer :: vk

        call describe_array_section(arena, slice_node, context, info, error_msg)
        if (len_trim(error_msg) > 0) return
        total_elements = array_section_total_elements(info)
        if (total_elements <= 0_c_int64_t) then
            error_msg = 'array section has no elements'
            return
        end if
        vk = context%symbols(info%source_index)%value_kind
        do linear_index = 0_c_int64_t, total_elements - 1_c_int64_t
            if (.not. emit_liric_print_space(context%session, error_msg)) return
            call lower_array_section_element_from_info(info, linear_index, context, &
                                                       value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk == VALUE_F32) then
                if (.not. emit_liric_print_f32_value(context%session, &
                                                     value, error_msg)) return
            else if (vk == VALUE_F64) then
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
            else
                if (.not. emit_liric_print_i32_value(context%session, &
                                                     context%i32_print_format_id, &
                                                     value, error_msg)) return
            end if
        end do
        if (.not. emit_liric_print_newline(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_array_section_print

    subroutine lower_whole_array_print(arena, expr_index, context, handled, error_msg)
        !! print *, a for a bare integer or real array: emit each element in
        !! storage order with list-directed spacing, matching gfortran. handled
        !! stays false when expr is not a bare supported-array identifier.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg

        call emit_whole_array_print_items(arena, expr_index, context, handled, &
                                          error_msg)
        if (.not. handled .or. len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_newline(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_whole_array_print

    subroutine emit_whole_array_print_items(arena, expr_index, context, handled, &
                                            error_msg)
        !! Emit every element of a bare integer/real array print item in storage
        !! order with list-directed spacing, but no trailing newline, so it can
        !! appear among other items in a multi-item list-directed print.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: value
        integer :: sym
        integer :: vk
        integer(c_int64_t) :: k, n

        handled = .false.
        call set_empty(error_msg)
        if (.not. is_identifier(arena, expr_index)) return
        call identifier_name(arena, expr_index, name, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        sym = find_symbol(context, name)
        if (sym <= 0) return
        if (.not. context%symbols(sym)%is_array) return
        if (context%symbols(sym)%is_allocatable) return
        vk = context%symbols(sym)%value_kind
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. vk /= VALUE_F64) return

        handled = .true.
        n = int(context%symbols(sym)%array_size, c_int64_t)
        do k = 0_c_int64_t, n - 1_c_int64_t
            if (.not. emit_liric_print_space(context%session, error_msg)) return
            call load_array_linear_element(context, sym, k, value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk == VALUE_F32) then
                if (.not. emit_liric_print_f32_value(context%session, &
                                                     value, error_msg)) return
            else if (vk == VALUE_F64) then
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
            else
                if (.not. emit_liric_print_i32_value(context%session, &
                                                     context%i32_print_format_id, &
                                                     value, error_msg)) return
            end if
        end do
        call set_empty(error_msg)
    end subroutine emit_whole_array_print_items

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
                ! A scalar operand inside whole-array arithmetic broadcasts: its
                ! single value participates in every element computation.
                call lower_elementwise_scalar_operand(arena, node_index, &
                                                      target_symbol_index, context, &
                                                      value, error_msg)
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

        if (is_literal(arena, node_index)) then
            ! A scalar literal inside whole-array arithmetic broadcasts.
            call lower_elementwise_scalar_operand(arena, node_index, &
                                                  target_symbol_index, context, &
                                                  value, error_msg)
            return
        end if

        if (is_elemental_real_call(arena, node_index)) then
            call lower_elementwise_elemental_call(arena, node_index, &
                                                  target_symbol_index, &
                                                  linear_index, context, value, &
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
            select case (context%symbols(target_symbol_index)%value_kind)
            case (VALUE_F32, VALUE_F64)
                call real_opcode(bin_op, bin_line, bin_col, opcode, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%symbols(target_symbol_index)%value_kind == VALUE_F32) then
                    if (.not. emit_liric_f32_binary(context%session, opcode, lhs, &
                                                    rhs, value, error_msg)) return
                else
                    if (.not. emit_liric_f64_binary(context%session, opcode, lhs, &
                                                    rhs, value, error_msg)) return
                end if
            case default
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
                                                   'direct LIRIC session supports '// &
                                                   'only +, -, and * for whole-array '// &
                                                   'arithmetic', error_msg)
                    return
                end select
                if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, &
                                          value, error_msg)) return
            end select
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
        integer :: vk

        if (linear_index < 0_c_int64_t .or. &
            linear_index >= int(context%symbols(symbol_index)%array_size, &
                                c_int64_t)) then
            error_msg = 'array linear index is out of range'
            return
        end if
        vk = context%symbols(symbol_index)%value_kind
        if (vk == VALUE_F32) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_liric_f32_load(context%session, addr, value, &
                                          error_msg)) then
                error_msg = 'f32_array_element_load: '//error_msg
                return
            end if
        else if (vk == VALUE_F64) then
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_liric_f64_load(context%session, addr, value, &
                                          error_msg)) then
                error_msg = 'f64_array_element_load: '//error_msg
                return
            end if
        else
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i32_load(context%session, addr, value, error_msg)) then
                error_msg = 'array_element_load: '//error_msg
                return
            end if
        end if
        call set_empty(error_msg)
    end subroutine load_array_linear_element

    subroutine lower_array_section_assignment(arena, node, target, context, error_msg)
        !! Assign to a rank-1 integer array section: arr(lo:hi:st) = rhs, where
        !! rhs is an array constructor, another rank-1 section, or a scalar. The
        !! section bounds are compile-time constants (describe_array_section), so
        !! emit one element store per position. Only rank-1 integer sections are
        !! handled; anything else returns an error and stays xfail.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(array_slice_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        type(array_section_info_t) :: tinfo, sinfo
        type(lr_operand_desc_t) :: value
        integer :: k, k2, n
        integer(c_int64_t) :: tgt_lin, src_lin

        call describe_array_section(arena, target, context, tinfo, error_msg)
        if (len_trim(error_msg) > 0) return
        n = int(tinfo%section_extents(1))

        ! Constructor and section-copy right-hand sides are handled for rank-1
        ! sections only; the scalar-broadcast tail below also covers rank-2.
        if (node_exists(arena, node%value_index) .and. tinfo%result_rank == 1 &
            .and. tinfo%source_rank == 1) then
            select type (rhs => arena%entries(node%value_index)%node)
            type is (array_literal_node)
                if (.not. allocated(rhs%element_indices)) then
                    error_msg = 'array constructor has no elements'
                    return
                end if
                if (size(rhs%element_indices) /= n) then
                    error_msg = 'array section length does not match constructor'
                    return
                end if
                do k = 0, n - 1
                    call lower_i32_expression(arena, rhs%element_indices(k + 1), &
                                              context, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    tgt_lin = tinfo%section_lowers(1) + &
                              int(k, c_int64_t) * tinfo%section_strides(1) - &
                              tinfo%source_lowers(1)
                    call store_array_linear_element(context, tinfo%source_index, &
                                                    tgt_lin, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
                call set_empty(error_msg)
                return
            type is (array_slice_node)
                call describe_array_section(arena, rhs, context, sinfo, error_msg)
                if (len_trim(error_msg) > 0) return
                if (sinfo%source_rank /= 1 .or. sinfo%result_rank /= 1 .or. &
                    int(sinfo%section_extents(1)) /= n) then
                    error_msg = 'array section assignment requires matching '// &
                                'rank-1 shapes'
                    return
                end if
                do k = 0, n - 1
                    src_lin = sinfo%section_lowers(1) + &
                              int(k, c_int64_t) * sinfo%section_strides(1) - &
                              sinfo%source_lowers(1)
                    call load_array_linear_element(context, sinfo%source_index, &
                                                   src_lin, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    tgt_lin = tinfo%section_lowers(1) + &
                              int(k, c_int64_t) * tinfo%section_strides(1) - &
                              tinfo%source_lowers(1)
                    call store_array_linear_element(context, tinfo%source_index, &
                                                    tgt_lin, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
                call set_empty(error_msg)
                return
            end select
        end if

        ! Scalar broadcast: arr(section) = expr, for rank-1 and rank-2 sections.
        if (tinfo%source_rank /= tinfo%result_rank .or. tinfo%result_rank > 2) then
            error_msg = 'array section assignment supports whole rank-1 and '// &
                        'rank-2 sections'
            return
        end if
        select case (context%symbols(tinfo%source_index)%value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, node%value_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node%value_index, context, value, error_msg)
        case default
            call lower_i32_expression(arena, node%value_index, context, value, error_msg)
        end select
        if (len_trim(error_msg) > 0) return
        if (tinfo%result_rank == 1) then
            do k = 0, n - 1
                tgt_lin = tinfo%section_lowers(1) + &
                          int(k, c_int64_t) * tinfo%section_strides(1) - &
                          tinfo%source_lowers(1)
                call store_array_linear_element(context, tinfo%source_index, &
                                                tgt_lin, value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else
            do k2 = 0, int(tinfo%section_extents(2)) - 1
                do k = 0, n - 1
                    tgt_lin = (tinfo%section_lowers(1) + &
                               int(k, c_int64_t) * tinfo%section_strides(1) - &
                               tinfo%source_lowers(1)) + tinfo%source_sizes(1) * &
                              (tinfo%section_lowers(2) + &
                               int(k2, c_int64_t) * tinfo%section_strides(2) - &
                               tinfo%source_lowers(2))
                    call store_array_linear_element(context, tinfo%source_index, &
                                                    tgt_lin, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
        end if
        call set_empty(error_msg)
    end subroutine lower_array_section_assignment

    subroutine store_array_linear_element(context, symbol_index, linear_index, &
                                          value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer :: vk

        vk = context%symbols(symbol_index)%value_kind
        if (vk == VALUE_F32) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_liric_f32_store(context%session, value, addr, &
                                           error_msg)) then
                error_msg = 'f32_array_store: '//error_msg
                return
            end if
        else if (vk == VALUE_F64) then
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_liric_f64_store(context%session, value, addr, &
                                           error_msg)) then
                error_msg = 'f64_array_store: '//error_msg
                return
            end if
        else
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) then
                error_msg = 'array_store: '//error_msg
                return
            end if
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
        integer :: sym_idx
        integer :: vk

        vk = VALUE_I32
        if (allocated(target%name)) then
            sym_idx = find_symbol(context, target%name)
            if (sym_idx > 0) vk = context%symbols(sym_idx)%value_kind
        end if

        if (vk == VALUE_F32) then
            call lower_f32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        else if (vk == VALUE_F64) then
            call lower_f64_expression(arena, node%value_index, context, value, &
                                      error_msg)
        else if (vk == VALUE_LOGICAL) then
            ! Logical occupies an i32 slot; lower the RHS as a logical so
            ! .true./.false. literals and relational results become 1/0.
            call lower_logical_expression(arena, node%value_index, context, value, &
                                          error_msg)
        else
            call lower_i32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_F32 .or. vk == VALUE_F64) then
            call lower_real_array_element_address(arena, target, context, vk, &
                                                  element_address, error_msg, &
                                                  'array assignment target')
        else
            call lower_i32_array_element_address(arena, target, context, &
                                                 element_address, error_msg, &
                                                 'array assignment target')
        end if
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_F32) then
            if (.not. emit_liric_f32_store(context%session, value, &
                                           element_address, error_msg)) then
                error_msg = 'f32_arr_assign_store: '//error_msg
                return
            end if
        else if (vk == VALUE_F64) then
            if (.not. emit_liric_f64_store(context%session, value, &
                                           element_address, error_msg)) then
                error_msg = 'f64_arr_assign_store: '//error_msg
                return
            end if
        else
            if (.not. emit_i32_store(context%session, value, element_address, &
                                     error_msg)) then
                error_msg = 'arr_assign_store: '//error_msg
                return
            end if
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
        integer :: vk
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
        if (context%symbols(symbol_index)%is_allocatable) then
            call lower_allocatable_element_address(arena, node, symbol_index, &
                                                   context, element_address, &
                                                   error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = trim(node%name)//' is not an array'
            return
        end if
        vk = context%symbols(symbol_index)%value_kind

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
            lower_bound_value = i32_immediate(context%session,  &
                                int(context%symbols(symbol_index)%array_lower_bound, &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, index_value, &
                                                      lower_bound_value, &
                                                      offset, error_msg)) then
                error_msg = 'arr_element_sub: '//error_msg
                return
            end if

            if (vk == VALUE_F32) then
                if (.not. emit_f32_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        offset, element_address, error_msg)) then
                    error_msg = 'arr_element_gep_f32: '//error_msg
                    return
                end if
            else if (vk == VALUE_F64) then
                if (.not. emit_f64_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        offset, element_address, error_msg)) then
                    error_msg = 'arr_element_gep_f64: '//error_msg
                    return
                end if
            else
                if (.not. emit_i32_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        offset, element_address, error_msg)) then
                    error_msg = 'arr_element_gep: '//error_msg
                    return
                end if
            end if
        else
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

            if (vk == VALUE_F32) then
                if (.not. emit_f32_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        index_value, element_address, error_msg)) then
                    error_msg = 'arr2_gep_f32: '//error_msg
                    return
                end if
            else if (vk == VALUE_F64) then
                if (.not. emit_f64_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        index_value, element_address, error_msg)) then
                    error_msg = 'arr2_gep_f64: '//error_msg
                    return
                end if
            else
                if (.not. emit_i32_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        index_value, element_address, error_msg)) then
                    error_msg = 'arr2_gep: '//error_msg
                    return
                end if
            end if
        end if

        call set_empty(error_msg)
    end subroutine lower_i32_array_element_address

    subroutine lower_real_array_element_address(arena, node, context, vk, &
                                                element_address, error_msg, &
                                                feature)
        !! GEP for a real (f32 or f64) array element. Subscripts are lowered
        !! as integers; the GEP type uses the appropriate element type.
        !! vk must be VALUE_F32 or VALUE_F64.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
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
            error_msg = 'real array access requires an array name'
            return
        end if
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'real array identifier was not declared: '//trim(node%name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = trim(node%name)//' is not an array'
            return
        end if

        rank = context%symbols(symbol_index)%array_rank
        diagnostic_feature = 'array expression'
        if (present(feature)) diagnostic_feature = feature
        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
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

        call lower_i32_expression(arena, node%arg_indices(1), context, &
                                  index_value, error_msg)
        if (len_trim(error_msg) > 0) return

        if (rank == 1) then
            lower_bound_value = i32_immediate(context%session, &
                                int(context%symbols(symbol_index)%array_lower_bound, &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, index_value, &
                                      lower_bound_value, offset, error_msg)) then
                error_msg = 'real_arr_sub: '//error_msg
                return
            end if
            if (vk == VALUE_F32) then
                if (.not. emit_f32_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        offset, element_address, error_msg)) then
                    error_msg = 'f32_arr_gep: '//error_msg
                    return
                end if
            else
                if (.not. emit_f64_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        offset, element_address, error_msg)) then
                    error_msg = 'f64_arr_gep: '//error_msg
                    return
                end if
            end if
        else
            lower_bound_value = i32_immediate(context%session, &
                                int(context%symbols(symbol_index)%array_dim_lowers(1), &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, index_value, &
                                      lower_bound_value, offset, error_msg)) then
                error_msg = 'real_arr2_i_off: '//error_msg
                return
            end if
            call lower_i32_expression(arena, node%arg_indices(2), context, &
                                      i_offset, error_msg)
            if (len_trim(error_msg) > 0) return
            lower_bound_value = i32_immediate(context%session, &
                                int(context%symbols(symbol_index)%array_dim_lowers(2), &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, i_offset, &
                                      lower_bound_value, j_offset, error_msg)) then
                error_msg = 'real_arr2_j_off: '//error_msg
                return
            end if
            lower_bound_value = i32_immediate(context%session, &
                                int(context%symbols(symbol_index)%array_dim_sizes(1), &
                                    c_int64_t))
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, j_offset, &
                                      lower_bound_value, scaled_index, error_msg)) then
                error_msg = 'real_arr2_stride: '//error_msg
                return
            end if
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, offset, &
                                      scaled_index, index_value, error_msg)) then
                error_msg = 'real_arr2_add: '//error_msg
                return
            end if
            if (vk == VALUE_F32) then
                if (.not. emit_f32_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        index_value, element_address, error_msg)) then
                    error_msg = 'f32_arr2_gep: '//error_msg
                    return
                end if
            else
                if (.not. emit_f64_array_element_addr(context%session, &
                        context%symbols(symbol_index)%array_size, &
                        context%symbols(symbol_index)%element_address, &
                        index_value, element_address, error_msg)) then
                    error_msg = 'f64_arr2_gep: '//error_msg
                    return
                end if
            end if
        end if
        call set_empty(error_msg)
    end subroutine lower_real_array_element_address

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
