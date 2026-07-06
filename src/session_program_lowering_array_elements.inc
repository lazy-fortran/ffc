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
        logical :: handled

        call emit_array_section_print_items(arena, slice_node, context, handled, &
                                            error_msg)
        if (.not. handled .or. len_trim(error_msg) > 0) return
        if (.not. emit_liric_print_newline(context%session, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_array_section_print

    subroutine emit_array_section_print_items(arena, slice_node, context, handled, &
                                              error_msg)
        !! Emit every element of an array-section print item (a(lo:hi:st)) in
        !! storage order with list-directed spacing and no trailing newline, so
        !! a section can sit among other items in a multi-item print. handled
        !! stays false on an unsupported base/kind, leaving the caller to fall
        !! through rather than fail.
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: value
        integer(c_int64_t) :: linear_index
        integer(c_int64_t) :: total_elements
        integer :: vk

        handled = .false.
        call describe_array_section(arena, slice_node, context, info, error_msg)
        if (len_trim(error_msg) > 0) return
        total_elements = array_section_total_elements(info)
        if (total_elements <= 0_c_int64_t) then
            error_msg = 'array section has no elements'
            return
        end if
        handled = .true.
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
        call set_empty(error_msg)
    end subroutine emit_array_section_print_items

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
                                            error_msg, prev_is_char)
        !! Emit every element of a bare integer/real array print item in storage
        !! order with list-directed spacing, but no trailing newline, so it can
        !! appear among other items in a multi-item list-directed print.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        !! True when the immediately preceding print item was a character value, so
        !! a leading separator space is suppressed before a character array.
        logical, intent(in), optional :: prev_is_char
        logical :: suppress_leading_space
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: value
        integer :: sym
        integer :: vk
        integer(c_int64_t) :: k, n

        handled = .false.
        call set_empty(error_msg)
        if (.not. is_identifier(arena, expr_index)) then
            call emit_whole_array_expr_print_items(arena, expr_index, context, &
                                                   handled, error_msg)
            return
        end if
        call identifier_name(arena, expr_index, name, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        sym = find_symbol(context, name)
        if (sym <= 0) return
        if (.not. context%symbols(sym)%is_array .and. &
            .not. context%symbols(sym)%is_allocatable) return
        vk = context%symbols(sym)%value_kind
        if (vk == VALUE_CHARACTER .and. .not. context%symbols(sym)%is_allocatable) &
            then
            suppress_leading_space = .false.
            if (present(prev_is_char)) suppress_leading_space = prev_is_char
            call emit_character_array_print_items(context, sym, &
                suppress_leading_space, handled, error_msg)
            return
        end if
        if (vk == VALUE_CHARACTER .and. context%symbols(sym)%is_allocatable) then
            call emit_allocatable_character_print_items(context, sym, handled, &
                error_msg)
            return
        end if
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. vk /= VALUE_F64 .and. &
            vk /= VALUE_LOGICAL .and. vk /= VALUE_I64 .and. vk /= VALUE_I8 .and. &
            vk /= VALUE_I16) return
        if (context%symbols(sym)%is_allocatable) then
            call emit_allocatable_print_items(context, sym, vk, handled, error_msg)
            return
        end if

        handled = .true.
        n = int(context%symbols(sym)%array_size, c_int64_t)
        do k = 0_c_int64_t, n - 1_c_int64_t
            if (.not. emit_liric_print_space(context%session, error_msg)) return
            call load_array_linear_element(context, sym, k, value, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_typed_array_print_value(context, vk, value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine emit_whole_array_print_items

    subroutine emit_typed_array_print_value(context, vk, value, error_msg)
        !! Emit one array element already loaded into value, formatted by its
        !! value kind with list-directed spacing already written by the caller.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (vk == VALUE_F32) then
            if (.not. emit_liric_print_f32_value(context%session, &
                                                 value, error_msg)) return
        else if (vk == VALUE_F64) then
            if (.not. emit_liric_print_f64_value(context%session, &
                                                 value, error_msg)) return
        else if (vk == VALUE_LOGICAL) then
            ! A logical element occupies an i32 slot (0/1); print it as T/F.
            call lower_print_logical_value(context, value, error_msg)
        else if (vk == VALUE_I64) then
            if (.not. emit_liric_print_i64_value(context%session, &
                                                 context%i64_print_format_id, &
                                                 value, error_msg)) return
        else if (vk == VALUE_I8) then
            if (.not. emit_liric_print_i8_value(context%session, &
                                                context%i8_print_format_id, &
                                                value, error_msg)) return
        else if (vk == VALUE_I16) then
            if (.not. emit_liric_print_i16_value(context%session, &
                                                 context%i16_print_format_id, &
                                                 value, error_msg)) return
        else
            if (.not. emit_liric_print_i32_value(context%session, &
                                                 context%i32_print_format_id, &
                                                 value, error_msg)) return
        end if
    end subroutine emit_typed_array_print_value

    subroutine emit_whole_array_expr_print_items(arena, expr_index, context, &
                                                 handled, error_msg)
        !! Print a whole-array expression that is not a bare identifier, e.g.
        !! print *, sin(x) or print *, a + b. The shape and element kind follow
        !! a whole-array operand inside the expression; each element is
        !! materialised through the elementwise evaluator and printed in storage
        !! order with list-directed spacing. Forms without a resolvable
        !! whole-array operand leave handled false (declined, not an error).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: sym
        integer :: vk
        integer(c_int64_t) :: k, n

        handled = .false.
        call set_empty(error_msg)
        sym = whole_array_expr_shape_symbol(arena, expr_index, context)
        if (sym <= 0) return
        vk = context%symbols(sym)%value_kind
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. vk /= VALUE_F64 .and. &
            vk /= VALUE_LOGICAL .and. vk /= VALUE_I64 .and. vk /= VALUE_I8 .and. &
            vk /= VALUE_I16) return

        handled = .true.
        n = int(context%symbols(sym)%array_size, c_int64_t)
        do k = 0_c_int64_t, n - 1_c_int64_t
            if (.not. emit_liric_print_space(context%session, error_msg)) return
            call lower_array_elementwise_value(arena, expr_index, sym, &
                                               int(k), context, value, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_typed_array_print_value(context, vk, value, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine emit_whole_array_expr_print_items

    recursive integer function whole_array_expr_shape_symbol(arena, expr_index, &
                                                             context) result(sym)
        !! Symbol whose shape and element kind govern a whole-array expression:
        !! the first whole-array identifier operand reached through elemental
        !! intrinsic calls, min/max calls, and elementwise binary operators.
        !! Returns 0 for forms without such an operand (scalars, unsupported
        !! calls, sections), so the caller can decline gracefully.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, err, bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        integer :: i

        sym = 0
        if (.not. node_exists(arena, expr_index)) return

        if (is_identifier(arena, expr_index)) then
            call get_identifier_name(arena, expr_index, id_name, err)
            if (len_trim(err) > 0) return
            i = find_symbol(context, id_name)
            if (i > 0) then
                if (is_elementwise_array_operand(context, i)) sym = i
            end if
            return
        end if

        if (is_elemental_real_call(arena, expr_index)) then
            select type (n => arena%entries(expr_index)%node)
            type is (call_or_subscript_node)
                if (n%is_array_access) return
                sym = whole_array_expr_shape_symbol(arena, n%arg_indices(1), &
                                                    context)
            end select
            return
        end if

        if (is_elemental_minmax_call(arena, expr_index)) then
            select type (n => arena%entries(expr_index)%node)
            type is (call_or_subscript_node)
                do i = 1, size(n%arg_indices)
                    sym = whole_array_expr_shape_symbol(arena, n%arg_indices(i), &
                                                        context)
                    if (sym > 0) return
                end do
            end select
            return
        end if

        if (is_binary_op(arena, expr_index)) then
            call get_binary_op_info(arena, expr_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, err)
            if (len_trim(err) > 0) return
            sym = whole_array_expr_shape_symbol(arena, bin_left, context)
            if (sym > 0) return
            sym = whole_array_expr_shape_symbol(arena, bin_right, context)
        end if
    end function whole_array_expr_shape_symbol

    subroutine emit_array_literal_print_items(arena, expr_index, context, &
                                              handled, error_msg)
        !! List-directed print of an inline array constructor [e1, e2, ...].
        !! Each explicit element is printed like a scalar item (leading
        !! separator space, then the value in its own kind), so a constructor of
        !! literals, variables, or function calls prints byte-for-byte like
        !! gfortran. An implied-do constructor is left unhandled (xfail).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: flatten_error
        integer :: i

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, expr_index)) return
        select type (ctor => arena%entries(expr_index)%node)
        type is (array_literal_node)
            if (allocated(ctor%syntax_style)) then
                if (ctor%syntax_style == 'implied_do') return
            end if
            if (.not. allocated(ctor%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, ctor%element_indices, flat, &
                                              flatten_error)
            if (len_trim(flatten_error) > 0) return
            if (size(flat) == 0) return
            ! Character constructors print adjacent values with no separator
            ! space; that spacing rule is not handled here, so leave them xfail.
            if (char_print_item(arena, flat(1), context)) return
            handled = .true.
            do i = 1, size(flat)
                if (.not. emit_liric_print_space(context%session, error_msg)) return
                call lower_print_expression_value(arena, flat(i), context, &
                                                  error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end select
    end subroutine emit_array_literal_print_items

    subroutine emit_allocatable_print_items(context, sym, vk, handled, error_msg)
        !! Whole-array print of a rank-1 allocatable whose extent is known at
        !! compile time (allocatable_static_size). Loads the descriptor data
        !! pointer and emits one typed print per element with list-directed
        !! spacing. A runtime-only extent leaves handled false (xfail).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        integer, intent(in) :: vk
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, addr, value
        integer(c_int64_t) :: stride, k, n

        handled = .false.
        call set_empty(error_msg)
        if (context%symbols(sym)%array_rank == 2) return
        n = int(context%symbols(sym)%allocatable_static_size, c_int64_t)
        if (n <= 0_c_int64_t) then
            if (context%symbols(sym)%array_rank /= 1) return
            handled = .true.
            call emit_runtime_alloc_print(context, sym, vk, error_msg)
            return
        end if

        handled = .true.
        stride = allocatable_elem_size(vk)
        descriptor = context%symbols(sym)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        do k = 0_c_int64_t, n - 1_c_int64_t
            if (.not. emit_liric_print_space(context%session, error_msg)) return
            if (.not. emit_ptr_offset(context%session, data_ptr, k * stride, &
                                      addr, error_msg)) return
            if (.not. emit_allocatable_element_load(context, vk, addr, value, &
                                                    error_msg)) return
            if (vk == VALUE_F32) then
                if (.not. emit_liric_print_f32_value(context%session, &
                                                     value, error_msg)) return
            else if (vk == VALUE_F64) then
                if (.not. emit_liric_print_f64_value(context%session, &
                                                     value, error_msg)) return
            else if (vk == VALUE_LOGICAL) then
                call lower_print_logical_value(context, value, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                if (.not. emit_liric_print_i32_value(context%session, &
                                                     context%i32_print_format_id, &
                                                     value, error_msg)) return
            end if
        end do
        call set_empty(error_msg)
    end subroutine emit_allocatable_print_items

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

        select type (ctor_node => arena%entries(node_index)%node)
        type is (array_literal_node)
            call lower_array_constructor_operand_element(arena, ctor_node, &
                                                         target_symbol_index, &
                                                         linear_index, context, &
                                                         value, error_msg)
            return
        end select

        select type (comp_node => arena%entries(node_index)%node)
        type is (component_access_node)
            block
                logical :: handled
                call try_load_array_component_element(arena, node_index, &
                    linear_index, context, value, handled, error_msg)
                if (handled .or. len_trim(error_msg) > 0) return
            end block
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

        if (is_elemental_minmax_call(arena, node_index)) then
            call lower_elementwise_minmax_call(arena, node_index, &
                                               target_symbol_index, linear_index, &
                                               context, value, error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'whole-array expression identifier was not declared: '// &
                            trim(id_name)
                return
            end if
            if (.not. is_elementwise_array_operand(context, symbol_index)) then
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

        if (is_elemental_user_call(arena, node_index, context)) then
            call lower_elementwise_user_call(arena, node_index, &
                                             target_symbol_index, linear_index, &
                                             context, value, error_msg)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(adjustl(lowercase_text(bin_op))) == '.not.') then
                call lower_array_not_element(arena, bin_right, &
                                             target_symbol_index, linear_index, &
                                             context, value, error_msg)
                return
            end if
            if (context%symbols(target_symbol_index)%value_kind == &
                    VALUE_LOGICAL .and. is_comparison_operator(bin_op)) then
                call lower_array_compare_element(arena, node_index, &
                                                 linear_index, context, value, &
                                                 error_msg)
                return
            end if
            call lower_array_elementwise_value(arena, bin_left, &
                                               target_symbol_index, linear_index, &
                                               context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (context%symbols(target_symbol_index)%value_kind)
            case (VALUE_F32, VALUE_F64)
                call lower_array_elementwise_value(arena, bin_right, &
                                                   target_symbol_index, linear_index, &
                                                   context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(bin_op) == '**') then
                    if (context%symbols(target_symbol_index)%value_kind == &
                        VALUE_F32) then
                        if (.not. emit_liric_f32_call(context%session, 'powf', &
                                [lhs, rhs], value, error_msg)) return
                    else
                        if (.not. emit_liric_f64_call(context%session, 'pow', &
                                [lhs, rhs], value, error_msg)) return
                    end if
                    return
                end if
                call real_opcode(bin_op, bin_line, bin_col, opcode, error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%symbols(target_symbol_index)%value_kind == VALUE_F32) then
                    if (.not. emit_liric_f32_binary(context%session, opcode, lhs, &
                                                    rhs, value, error_msg)) return
                else
                    if (.not. emit_liric_f64_binary(context%session, opcode, lhs, &
                                                    rhs, value, error_msg)) return
                end if
            case (VALUE_I64, VALUE_I8, VALUE_I16)
                call unsupported_feature_error('whole-array arithmetic', &
                                               bin_line, bin_col, &
                                               'direct LIRIC session supports '// &
                                               'whole-array +, -, *, /, and ** '// &
                                               'for default-kind integer, real, '// &
                                               'and logical arrays only', error_msg)
                return
            case default
                if (trim(bin_op) == '**') then
                    call lower_i32_pow(arena, bin_right, bin_line, bin_col, &
                                       context, lhs, value, error_msg)
                    return
                end if
                call lower_array_elementwise_value(arena, bin_right, &
                                                   target_symbol_index, linear_index, &
                                                   context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call integer_opcode(bin_op, opcode, error_msg)
                if (len_trim(error_msg) > 0) then
                    call unsupported_feature_error('whole-array arithmetic', &
                                                   bin_line, bin_col, &
                                                   'direct LIRIC session supports '// &
                                                   'only +, -, *, /, and ** for '// &
                                                   'whole-array arithmetic', error_msg)
                    return
                end if
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

    subroutine lower_array_constructor_operand_element(arena, ctor, &
                                                        target_symbol_index, &
                                                        linear_index, context, &
                                                        value, error_msg)
        ! An array constructor used as an rvalue operand inside a whole-array
        ! expression (a = b + [1, 2, 3]): select its linear_index'th flattened
        ! element and lower it in the target array's kind.
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: flat(:)

        if (.not. allocated(ctor%element_indices)) then
            error_msg = 'array constructor has no elements'
            return
        end if
        allocate (flat(0))
        call flatten_constructor_elements(arena, ctor%element_indices, flat, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (linear_index < 0 .or. linear_index >= size(flat)) then
            error_msg = 'array constructor operand is shorter than the target array'
            return
        end if
        call lower_elementwise_scalar_operand(arena, flat(linear_index + 1), &
                                              target_symbol_index, context, &
                                              value, error_msg)
    end subroutine lower_array_constructor_operand_element

    logical function is_comparison_operator(source_op) result(is_compare)
        ! True for a relational operator whose whole-array elementwise use
        ! produces a logical array (mask = a > b).
        character(len=*), intent(in) :: source_op

        select case (trim(source_op))
        case ('>', '.gt.', '>=', '.ge.', '<', '.lt.', '<=', '.le.', &
              '==', '=', '.eq.', '/=', '!=', '.ne.')
            is_compare = .true.
        case default
            is_compare = .false.
        end select
    end function is_comparison_operator

    subroutine lower_array_compare_element(arena, node_index, linear_index, &
                                           context, value, error_msg)
        ! Evaluate a whole-array comparison at one element, producing the
        ! logical result for a mask assignment (mask = a > b). The i1
        ! condition is materialized through select_value's branch-and-phi
        ! (as WHERE already does for its masked stores) rather than stored
        ! directly: this backend mis-compiles a raw icmp result written
        ! straight to memory (verified empirically to misorder both "/="
        ! and strict </> at exact operand ties), while a branch-selected
        ! 0/1 literal stores correctly.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond, true_val, false_val

        call set_empty(error_msg)
        call lower_where_mask_element(arena, node_index, linear_index, context, &
                                      cond, error_msg)
        if (len_trim(error_msg) > 0) return
        true_val = i32_immediate(context%session, 1_c_int64_t)
        false_val = i32_immediate(context%session, 0_c_int64_t)
        call select_value(context, cond, true_val, false_val, value, error_msg)
    end subroutine lower_array_compare_element

    subroutine lower_array_not_element(arena, operand_index, &
                                       target_symbol_index, linear_index, &
                                       context, value, error_msg)
        ! Evaluate a whole-array .NOT. at one element (a = .not. b). Unary
        ! .not. is parsed as a binary op whose left operand is a virtual
        ! (invalid) index, so the caller passes only the real right operand.
        ! The inversion is materialized through select_value's branch-and-phi,
        ! matching lower_array_compare_element's workaround for the same
        ! raw-icmp-to-memory miscompile.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: operand_index
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: operand, zero, cond, true_val, false_val

        call set_empty(error_msg)
        if (context%symbols(target_symbol_index)%value_kind /= VALUE_LOGICAL) then
            call unsupported_feature_error('whole-array .not.', &
                                           get_node_line(arena, operand_index), &
                                           get_node_column(arena, operand_index), &
                                           'direct LIRIC session supports '// &
                                           '.not. for logical arrays only', &
                                           error_msg)
            return
        end if
        call lower_array_elementwise_value(arena, operand_index, &
                                           target_symbol_index, linear_index, &
                                           context, operand, error_msg)
        if (len_trim(error_msg) > 0) return
        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, operand, zero, &
                                      cond, error_msg)) return
        true_val = i32_immediate(context%session, 0_c_int64_t)
        false_val = i32_immediate(context%session, 1_c_int64_t)
        call select_value(context, cond, true_val, false_val, value, error_msg)
    end subroutine lower_array_not_element

    logical function is_elementwise_array_operand(context, symbol_index) &
        result(is_operand)
        ! An identifier participates as an array operand in whole-array
        ! elementwise lowering when it is a fixed-size array, or a rank>=1
        ! allocatable (the allocatable itself is not flagged is_array).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index

        is_operand = context%symbols(symbol_index)%is_array .or. &
            (context%symbols(symbol_index)%is_allocatable .and. &
             context%symbols(symbol_index)%array_rank >= 1)
    end function is_elementwise_array_operand

    integer function effective_array_extent(context, symbol_index) result(n)
        ! Element count of a fixed-size or allocatable array symbol.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index

        if (context%symbols(symbol_index)%is_allocatable) then
            n = context%symbols(symbol_index)%allocatable_static_size
        else
            n = context%symbols(symbol_index)%array_size
        end if
    end function effective_array_extent

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

        if (context%symbols(target_symbol_index)%is_allocatable .or. &
            context%symbols(source_symbol_index)%is_allocatable) then
            if (effective_array_extent(context, target_symbol_index) /= &
                effective_array_extent(context, source_symbol_index)) then
                target_name = trim(context%symbols(target_symbol_index)%name)
                source_name = trim(context%symbols(source_symbol_index)%name)
                error_msg = 'whole-array assignment requires conforming shapes: '// &
                    trim(target_name)//' and '//trim(source_name)//' differ in extent'
                return
            end if
            call set_empty(error_msg)
            return
        end if

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

        ! Allocatables hold no static stack base: read element linear_index from
        ! the descriptor, bounded by the compile-time allocatable_static_size.
        if (context%symbols(symbol_index)%is_allocatable) then
            if (linear_index < 0_c_int64_t .or. &
                linear_index >= int( &
                    context%symbols(symbol_index)%allocatable_static_size, &
                    c_int64_t)) then
                error_msg = 'array linear index is out of range'
                return
            end if
            call load_allocatable_linear_element(context, symbol_index, &
                                                 linear_index, value, error_msg)
            return
        end if

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
        else if (vk == VALUE_I64) then
            if (.not. emit_i64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i64_load(context%session, addr, value, error_msg)) then
                error_msg = 'i64_array_element_load: '//error_msg
                return
            end if
        else if (vk == VALUE_I8) then
            if (.not. emit_i8_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i8_load(context%session, addr, value, error_msg)) then
                error_msg = 'i8_array_element_load: '//error_msg
                return
            end if
        else if (vk == VALUE_I16) then
            if (.not. emit_i16_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i16_load(context%session, addr, value, error_msg)) then
                error_msg = 'i16_array_element_load: '//error_msg
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

    subroutine load_allocatable_linear_element(context, symbol_index, &
                                               linear_index, value, error_msg)
        ! Load element linear_index of a rank-1 allocatable from its descriptor
        ! data pointer. The caller bounds linear_index against the compile-time
        ! allocatable_static_size, so reductions can unroll over a constant
        ! extent without a static stack array.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, addr
        integer(c_int64_t) :: stride
        integer :: vk

        vk = context%symbols(symbol_index)%value_kind
        stride = allocatable_elem_size(vk)
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_ptr_offset(context%session, data_ptr, &
                linear_index * stride, addr, error_msg)) return
        if (.not. emit_allocatable_element_load(context, vk, addr, value, &
                                                error_msg)) return
        call set_empty(error_msg)
    end subroutine load_allocatable_linear_element

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
            call lower_f32_operand(arena, node%value_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_operand(arena, node%value_index, context, value, error_msg)
        case (VALUE_I64)
            call lower_i64_expression(arena, node%value_index, context, value, error_msg)
        case (VALUE_I8)
            call lower_i8_expression(arena, node%value_index, context, value, error_msg)
        case (VALUE_I16)
            call lower_i16_expression(arena, node%value_index, context, value, error_msg)
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

        if (context%symbols(symbol_index)%is_allocatable) then
            if (linear_index < 0_c_int64_t .or. &
                linear_index >= int( &
                    context%symbols(symbol_index)%allocatable_static_size, &
                    c_int64_t)) then
                error_msg = 'array linear index is out of range'
                return
            end if
            call store_allocatable_linear_element(context, symbol_index, &
                                                  linear_index, value, error_msg)
            return
        end if

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
        else if (vk == VALUE_I64) then
            if (.not. emit_i64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i64_store(context%session, value, addr, error_msg)) then
                error_msg = 'i64_array_store: '//error_msg
                return
            end if
        else if (vk == VALUE_I8) then
            if (.not. emit_i8_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i8_store(context%session, value, addr, error_msg)) then
                error_msg = 'i8_array_store: '//error_msg
                return
            end if
        else if (vk == VALUE_I16) then
            if (.not. emit_i16_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, linear_index), addr, &
                    error_msg)) return
            if (.not. emit_i16_store(context%session, value, addr, error_msg)) then
                error_msg = 'i16_array_store: '//error_msg
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

    subroutine store_allocatable_linear_element(context, symbol_index, &
                                                linear_index, value, error_msg)
        ! Store element linear_index of a rank-1 allocatable through its
        ! descriptor data pointer. Mirrors load_allocatable_linear_element.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, addr
        integer(c_int64_t) :: stride
        integer :: vk

        vk = context%symbols(symbol_index)%value_kind
        stride = allocatable_elem_size(vk)
        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_ptr_offset(context%session, data_ptr, &
                linear_index * stride, addr, error_msg)) return
        if (.not. emit_allocatable_element_store(context, vk, value, addr, &
                                                 error_msg)) return
        call set_empty(error_msg)
    end subroutine store_allocatable_linear_element

    subroutine store_array_element_at_operand(context, symbol_index, &
                                              index_value, value, error_msg)
        ! Store an element at a runtime i32 index into a fixed-size array. Used
        ! by pack/unpack where the destination position is a loop cursor.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: index_value
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer :: vk

        vk = context%symbols(symbol_index)%value_kind
        if (vk == VALUE_F32) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    index_value, addr, error_msg)) return
            if (.not. emit_liric_f32_store(context%session, value, addr, &
                                           error_msg)) return
        else if (vk == VALUE_F64) then
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    index_value, addr, error_msg)) return
            if (.not. emit_liric_f64_store(context%session, value, addr, &
                                           error_msg)) return
        else
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    index_value, addr, error_msg)) return
            if (.not. emit_i32_store(context%session, value, addr, error_msg)) &
                return
        end if
        call set_empty(error_msg)
    end subroutine store_array_element_at_operand

    subroutine load_array_element_at_operand(context, symbol_index, &
                                             index_value, value, error_msg)
        ! Load an element at a runtime i32 index from a fixed-size array.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: index_value
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer :: vk

        vk = context%symbols(symbol_index)%value_kind
        if (vk == VALUE_F32) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    index_value, addr, error_msg)) return
            if (.not. emit_liric_f32_load(context%session, addr, value, &
                                          error_msg)) return
        else if (vk == VALUE_F64) then
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    index_value, addr, error_msg)) return
            if (.not. emit_liric_f64_load(context%session, addr, value, &
                                          error_msg)) return
        else
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    index_value, addr, error_msg)) return
            if (.not. emit_i32_load(context%session, addr, value, error_msg)) &
                return
        end if
        call set_empty(error_msg)
    end subroutine load_array_element_at_operand

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

        if (vk == VALUE_CHARACTER) then
            if (sym_idx > 0) then
                if (context%symbols(sym_idx)%is_allocatable) then
                    call lower_allocatable_character_element_assignment(arena, &
                        node, target, context, error_msg)
                    return
                end if
            end if
            call lower_character_array_element_assignment(arena, node, target, &
                                                          context, error_msg)
            return
        end if

        if (vk == VALUE_C4 .or. vk == VALUE_C8) then
            call lower_complex_array_element_assignment(arena, node, target, &
                                                        context, sym_idx, vk, &
                                                        error_msg)
            return
        end if

        if (vk == VALUE_F32) then
            call lower_f32_operand(arena, node%value_index, context, value, &
                                   error_msg)
        else if (vk == VALUE_F64) then
            call lower_f64_operand(arena, node%value_index, context, value, &
                                   error_msg)
        else if (vk == VALUE_LOGICAL) then
            ! Logical occupies an i32 slot; lower the RHS as a logical so
            ! .true./.false. literals and relational results become 1/0.
            call lower_logical_expression(arena, node%value_index, context, value, &
                                          error_msg)
        else if (vk == VALUE_I64) then
            call lower_i64_expression(arena, node%value_index, context, value, &
                                      error_msg)
        else if (vk == VALUE_I8) then
            call lower_i8_expression(arena, node%value_index, context, value, &
                                     error_msg)
        else if (vk == VALUE_I16) then
            call lower_i16_expression(arena, node%value_index, context, value, &
                                      error_msg)
        else
            call lower_i32_expression(arena, node%value_index, context, value, &
                                      error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_F32 .or. vk == VALUE_F64) then
            call lower_array_element_address_for_kind(arena, target, context, vk, &
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
        else if (vk == VALUE_I64) then
            if (.not. emit_i64_store(context%session, value, element_address, &
                                     error_msg)) then
                error_msg = 'i64_arr_assign_store: '//error_msg
                return
            end if
        else if (vk == VALUE_I8) then
            if (.not. emit_i8_store(context%session, value, element_address, &
                                    error_msg)) then
                error_msg = 'i8_arr_assign_store: '//error_msg
                return
            end if
        else if (vk == VALUE_I16) then
            if (.not. emit_i16_store(context%session, value, element_address, &
                                     error_msg)) then
                error_msg = 'i16_arr_assign_store: '//error_msg
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
        type(lr_operand_desc_t) :: linear_index

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

        diagnostic_feature = 'array expression'
        if (present(feature)) diagnostic_feature = feature
        call lower_array_element_linear_index(arena, node, context, symbol_index, &
                                              linear_index, error_msg, &
                                              diagnostic_feature)
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_F32) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'arr_element_gep_f32: '//error_msg
                return
            end if
        else if (vk == VALUE_F64) then
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'arr_element_gep_f64: '//error_msg
                return
            end if
        else if (vk == VALUE_I64) then
            if (.not. emit_i64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'arr_element_gep_i64: '//error_msg
                return
            end if
        else if (vk == VALUE_I8) then
            if (.not. emit_i8_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'arr_element_gep_i8: '//error_msg
                return
            end if
        else if (vk == VALUE_I16) then
            if (.not. emit_i16_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'arr_element_gep_i16: '//error_msg
                return
            end if
        else
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'arr_element_gep: '//error_msg
                return
            end if
        end if

        call set_empty(error_msg)
    end subroutine lower_i32_array_element_address

    subroutine lower_array_element_linear_index(arena, node, context, &
                                                symbol_index, linear_index, &
                                                error_msg, feature)
        !! Column-major linear element index of a subscripted fixed-size array of
        !! any rank: sum_k (idx_k - lower_k) * prod_{j<k} extent_j. Each extent is
        !! a compile-time constant, so per-dimension strides are integer
        !! immediates and only the subscripts become runtime operands.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: linear_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: feature
        integer :: rank, arg_count, k
        integer(c_int64_t) :: stride, lower_k
        type(lr_operand_desc_t) :: idx_op, off_op, term_op, acc_op, next_acc

        rank = context%symbols(symbol_index)%array_rank
        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
        if (arg_count /= rank) then
            call unsupported_feature_error(feature, node%line, node%column, &
                'array access subscript count must match the array rank', error_msg)
            return
        end if

        stride = 1_c_int64_t
        do k = 1, rank
            call validate_i32_array_subscript(arena, node%arg_indices(k), context, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, node%arg_indices(k), context, idx_op, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            lower_k = int(context%symbols(symbol_index)%array_dim_lowers(k), &
                          c_int64_t)
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, idx_op, &
                    i32_immediate(context%session, lower_k), off_op, error_msg)) then
                error_msg = 'arr_index_sub: '//error_msg
                return
            end if
            if (stride == 1_c_int64_t) then
                term_op = off_op
            else
                if (.not. emit_i32_binary(context%session, LR_OP_MUL, off_op, &
                        i32_immediate(context%session, stride), term_op, &
                        error_msg)) then
                    error_msg = 'arr_index_mul: '//error_msg
                    return
                end if
            end if
            if (k == 1) then
                acc_op = term_op
            else
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc_op, &
                        term_op, next_acc, error_msg)) then
                    error_msg = 'arr_index_add: '//error_msg
                    return
                end if
                acc_op = next_acc
            end if
            stride = stride * int(context%symbols(symbol_index)%array_dim_sizes(k), &
                                  c_int64_t)
        end do
        linear_index = acc_op
        call set_empty(error_msg)
    end subroutine lower_array_element_linear_index

    subroutine lower_array_element_address_for_kind(arena, node, context, vk, &
                                                    element_address, error_msg, &
                                                    feature)
        !! Real (f32/f64) array element address, routing allocatable references
        !! through the descriptor-based path. Fixed arrays use the GEP path.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: feature
        integer :: symbol_index

        if (allocated(node%name)) then
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%is_allocatable) then
                    call lower_allocatable_element_address(arena, node, &
                        symbol_index, context, element_address, error_msg)
                    return
                end if
            end if
        end if
        call lower_real_array_element_address(arena, node, context, vk, &
                                              element_address, error_msg, feature)
    end subroutine lower_array_element_address_for_kind

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
        type(lr_operand_desc_t) :: linear_index

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

        diagnostic_feature = 'array expression'
        if (present(feature)) diagnostic_feature = feature
        call lower_array_element_linear_index(arena, node, context, symbol_index, &
                                              linear_index, error_msg, &
                                              diagnostic_feature)
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_F32) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'f32_arr_gep: '//error_msg
                return
            end if
        else
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, element_address, error_msg)) then
                error_msg = 'f64_arr_gep: '//error_msg
                return
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

    ! sum/product/count/any/all(source, dim) reduced along one dimension of a
    ! rank-2 source into the rank-1 target symbol. All extents are known at
    ! compile time, so the reduction unrolls to a fixed sequence of loads and
    ! accumulations. Declines (handled=.false., no error) for any form outside
    ! this scope so the caller keeps its existing behaviour.
    subroutine lower_dim_reduction_assignment(arena, rhs, symbol_index, context, &
                                              handled, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: rhs
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: reduce_op, source_arg, dim_arg
        integer :: src_sym, i, j
        integer :: cmp_lsym, cmp_lscalar, cmp_rsym, cmp_rscalar
        integer :: d1, d2, out_size, reduce_extent
        integer :: source_vk, result_vk, compare_vk, predicate
        integer(c_int64_t) :: dim_value, lin
        logical :: is_comparison
        character(len=:), allocatable :: cmp_op
        type(lr_operand_desc_t) :: acc, cond, elem

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(rhs%name)) return
        if (rhs%is_array_access) return
        name = lowercase_text(trim(rhs%name))
        select case (name)
        case ('sum'); reduce_op = DIM_REDUCE_SUM
        case ('product'); reduce_op = DIM_REDUCE_PROD
        case ('count'); reduce_op = DIM_REDUCE_COUNT
        case ('any'); reduce_op = DIM_REDUCE_ANY
        case ('all'); reduce_op = DIM_REDUCE_ALL
        case default; return
        end select

        call split_reduction_args(arena, rhs, source_arg, dim_arg)
        if (source_arg <= 0 .or. dim_arg <= 0) return
        call eval_i32_constant(arena, dim_arg, context, dim_value, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        if (dim_value /= 1_c_int64_t .and. dim_value /= 2_c_int64_t) return

        ! The target must be a plain rank-1 array of statically known size.
        if (context%symbols(symbol_index)%is_allocatable) return
        if (context%symbols(symbol_index)%array_rank /= 1) return
        result_vk = context%symbols(symbol_index)%value_kind

        ! Resolve the source: either a stored rank-2 array, or an elementwise
        ! relational comparison whose array operand is rank-2 (for the mask
        ! reductions count/any/all; the other operand may be a scalar).
        is_comparison = .false.
        src_sym = 0
        cmp_lsym = 0
        cmp_lscalar = 0
        cmp_rsym = 0
        cmp_rscalar = 0
        call dim_reduction_array_symbol(arena, source_arg, context, src_sym)
        if (src_sym <= 0) then
            if (reduce_op == DIM_REDUCE_SUM .or. reduce_op == DIM_REDUCE_PROD) return
            call dim_reduction_comparison(arena, source_arg, context, cmp_op, &
                                          cmp_lsym, cmp_lscalar, cmp_rsym, &
                                          cmp_rscalar, src_sym, compare_vk, &
                                          is_comparison)
            if (.not. is_comparison) return
        end if
        if (context%symbols(src_sym)%array_rank /= 2) return

        source_vk = context%symbols(src_sym)%value_kind
        d1 = context%symbols(src_sym)%array_dim_sizes(1)
        d2 = context%symbols(src_sym)%array_dim_sizes(2)
        if (d1 <= 0 .or. d2 <= 0) return
        if (dim_value == 1_c_int64_t) then
            reduce_extent = d1
            out_size = d2
        else
            reduce_extent = d2
            out_size = d1
        end if
        if (out_size /= context%symbols(symbol_index)%array_size) return

        ! Numeric sum/product keep the source kind; mask reductions work in i32
        ! (count -> integer count, any/all -> logical 0/1).
        if (reduce_op == DIM_REDUCE_SUM .or. reduce_op == DIM_REDUCE_PROD) then
            select case (source_vk)
            case (VALUE_I32, VALUE_F32, VALUE_F64)
            case default; return
            end select
            if (result_vk /= source_vk) return
        else if (.not. is_comparison) then
            if (source_vk /= VALUE_LOGICAL) return
        end if
        if (reduce_op == DIM_REDUCE_COUNT .and. result_vk /= VALUE_I32) return
        if ((reduce_op == DIM_REDUCE_ANY .or. reduce_op == DIM_REDUCE_ALL) .and. &
            result_vk /= VALUE_LOGICAL) return

        if (is_comparison) then
            select case (compare_vk)
            case (VALUE_F32, VALUE_F64)
                call real_compare_predicate(cmp_op, predicate, error_msg)
            case default
                call integer_compare_predicate(cmp_op, predicate, error_msg)
            end select
            if (len_trim(error_msg) > 0) then
                call set_empty(error_msg)
                return
            end if
        end if

        do i = 0, out_size - 1
            call dim_reduction_identity(context, reduce_op, source_vk, acc)
            do j = 0, reduce_extent - 1
                if (dim_value == 1_c_int64_t) then
                    lin = int(j, c_int64_t) + int(i, c_int64_t) * int(d1, c_int64_t)
                else
                    lin = int(i, c_int64_t) + int(j, c_int64_t) * int(d1, c_int64_t)
                end if
                if (reduce_op == DIM_REDUCE_SUM .or. &
                    reduce_op == DIM_REDUCE_PROD) then
                    call load_array_linear_element(context, src_sym, lin, elem, &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                    call dim_reduction_accumulate_numeric(context, reduce_op, &
                        source_vk, acc, elem, error_msg)
                    if (len_trim(error_msg) > 0) return
                else
                    call dim_reduction_mask_element(arena, context, is_comparison, &
                        cmp_lsym, cmp_lscalar, cmp_rsym, cmp_rscalar, src_sym, &
                        compare_vk, predicate, lin, cond, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call dim_reduction_accumulate_mask(context, reduce_op, acc, &
                        cond, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end do
            call store_array_linear_element(context, symbol_index, &
                                            int(i, c_int64_t), acc, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        handled = .true.
        call set_empty(error_msg)
    end subroutine lower_dim_reduction_assignment

    ! Separate a reduction call's actuals into the source (first positional) and
    ! the dim (second positional or dim= keyword). Zero when absent.
    subroutine split_reduction_args(arena, rhs, source_arg, dim_arg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: rhs
        integer, intent(out) :: source_arg
        integer, intent(out) :: dim_arg
        character(len=:), allocatable :: kw_name
        integer :: i, kw_value, npos

        source_arg = 0
        dim_arg = 0
        npos = 0
        if (.not. allocated(rhs%arg_indices)) return
        do i = 1, size(rhs%arg_indices)
            call reshape_keyword_arg(arena, rhs%arg_indices(i), kw_name, kw_value)
            if (len_trim(kw_name) > 0) then
                if (kw_name == 'dim') dim_arg = kw_value
            else
                npos = npos + 1
                if (npos == 1) then
                    source_arg = rhs%arg_indices(i)
                else if (npos == 2 .and. dim_arg == 0) then
                    dim_arg = rhs%arg_indices(i)
                end if
            end if
        end do
    end subroutine split_reduction_args

    ! Bind an identifier actual to a stored (non-allocatable) array symbol.
    subroutine dim_reduction_array_symbol(arena, idx, context, sym)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: sym
        character(len=:), allocatable :: name, name_err
        integer :: s

        sym = 0
        if (.not. is_identifier(arena, idx)) return
        call get_identifier_name(arena, idx, name, name_err)
        if (len_trim(name_err) > 0) return
        s = find_symbol(context, name)
        if (s <= 0) return
        if (.not. context%symbols(s)%is_array) return
        if (context%symbols(s)%is_allocatable) return
        sym = s
    end subroutine dim_reduction_array_symbol

    ! Recognize source = lhs .rel. rhs where at least one operand is a stored
    ! rank-2 array and the other is a conforming rank-2 array or a scalar.
    ! Returns each side as (sym>0 array | scalar_idx>0 expression), the rank-2
    ! array symbol that fixes the shape, and the comparison kind.
    subroutine dim_reduction_comparison(arena, idx, context, op, lsym, lscalar, &
                                        rsym, rscalar, dims_sym, compare_vk, ok)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: op
        integer, intent(out) :: lsym, lscalar, rsym, rscalar
        integer, intent(out) :: dims_sym, compare_vk
        logical, intent(out) :: ok
        integer :: left_idx, right_idx, line, col
        character(len=:), allocatable :: op_err

        ok = .false.
        lsym = 0
        lscalar = 0
        rsym = 0
        rscalar = 0
        dims_sym = 0
        compare_vk = VALUE_I32
        if (.not. is_binary_op(arena, idx)) return
        call get_binary_op_info(arena, idx, op, left_idx, right_idx, line, col, &
                                op_err)
        if (len_trim(op_err) > 0) return
        if (.not. is_relational_operator(op)) return
        call dim_reduction_array_symbol(arena, left_idx, context, lsym)
        call dim_reduction_array_symbol(arena, right_idx, context, rsym)
        if (lsym <= 0) lscalar = left_idx
        if (rsym <= 0) rscalar = right_idx
        if (lsym > 0) then
            dims_sym = lsym
        else if (rsym > 0) then
            dims_sym = rsym
        else
            return
        end if
        if (context%symbols(dims_sym)%array_rank /= 2) return
        ! A second array operand must conform to the shape-defining one.
        if (lsym > 0 .and. rsym > 0) then
            if (context%symbols(lsym)%array_dim_sizes(1) /= &
                context%symbols(rsym)%array_dim_sizes(1)) return
            if (context%symbols(lsym)%array_dim_sizes(2) /= &
                context%symbols(rsym)%array_dim_sizes(2)) return
        end if
        if (lsym > 0) then
            if (.not. supported_reduction_kind( &
                context%symbols(lsym)%value_kind)) return
        end if
        if (rsym > 0) then
            if (.not. supported_reduction_kind( &
                context%symbols(rsym)%value_kind)) return
        end if
        compare_vk = context%symbols(dims_sym)%value_kind
        ok = .true.
    end subroutine dim_reduction_comparison

    ! Seed a reduction accumulator: 0/1 for sum/product, 0 count, 0 (false) any,
    ! 1 (true) all. Mask accumulators always start in i32.
    subroutine dim_reduction_identity(context, reduce_op, source_vk, acc)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: reduce_op, source_vk
        type(lr_operand_desc_t), intent(out) :: acc

        select case (reduce_op)
        case (DIM_REDUCE_SUM)
            select case (source_vk)
            case (VALUE_F32); acc = liric_f32_immediate(context%session, 0.0_c_float)
            case (VALUE_F64); acc = liric_f64_immediate(context%session, 0.0_c_double)
            case default; acc = i32_immediate(context%session, 0_c_int64_t)
            end select
        case (DIM_REDUCE_PROD)
            select case (source_vk)
            case (VALUE_F32); acc = liric_f32_immediate(context%session, 1.0_c_float)
            case (VALUE_F64); acc = liric_f64_immediate(context%session, 1.0_c_double)
            case default; acc = i32_immediate(context%session, 1_c_int64_t)
            end select
        case (DIM_REDUCE_ALL)
            acc = i32_immediate(context%session, 1_c_int64_t)
        case default
            acc = i32_immediate(context%session, 0_c_int64_t)
        end select
    end subroutine dim_reduction_identity

    subroutine dim_reduction_accumulate_numeric(context, reduce_op, source_vk, &
                                                acc, elem, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: reduce_op, source_vk
        type(lr_operand_desc_t), intent(inout) :: acc
        type(lr_operand_desc_t), intent(in) :: elem
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: next

        call set_empty(error_msg)
        select case (source_vk)
        case (VALUE_F32)
            if (reduce_op == DIM_REDUCE_SUM) then
                if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, acc, &
                                                elem, next, error_msg)) return
            else
                if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, acc, &
                                                elem, next, error_msg)) return
            end if
        case (VALUE_F64)
            if (reduce_op == DIM_REDUCE_SUM) then
                if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, acc, &
                                                elem, next, error_msg)) return
            else
                if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, acc, &
                                                elem, next, error_msg)) return
            end if
        case default
            if (reduce_op == DIM_REDUCE_SUM) then
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc, elem, &
                                          next, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_MUL, acc, elem, &
                                          next, error_msg)) return
            end if
        end select
        acc = next
    end subroutine dim_reduction_accumulate_numeric

    ! Produce the i32 0/1 mask element at linear index lin: a compare for the
    ! relational-source form (each side either a rank-2 array at lin or a
    ! broadcast scalar), or a direct logical load otherwise.
    subroutine dim_reduction_mask_element(arena, context, is_comparison, &
            lsym, lscalar, rsym, rscalar, src_sym, compare_vk, predicate, lin, &
            cond, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_comparison
        integer, intent(in) :: lsym, lscalar, rsym, rscalar, src_sym
        integer, intent(in) :: compare_vk, predicate
        integer(c_int64_t), intent(in) :: lin
        type(lr_operand_desc_t), intent(out) :: cond
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs, rhs
        integer, allocatable :: no_flat(:)

        call set_empty(error_msg)
        if (.not. is_comparison) then
            call load_array_linear_element(context, src_sym, lin, cond, error_msg)
            return
        end if
        call reduction_side_element(arena, context, lsym, no_flat, lscalar, &
                                    int(lin), compare_vk, lhs, error_msg)
        if (len_trim(error_msg) > 0) return
        call reduction_side_element(arena, context, rsym, no_flat, rscalar, &
                                    int(lin), compare_vk, rhs, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (compare_vk)
        case (VALUE_F32)
            if (.not. emit_liric_f32_fcmp(context%session, predicate, lhs, rhs, &
                                          cond, error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_fcmp(context%session, predicate, lhs, rhs, &
                                          cond, error_msg)) return
        case default
            if (.not. emit_liric_i32_icmp(context%session, predicate, lhs, rhs, &
                                          cond, error_msg)) return
        end select
    end subroutine dim_reduction_mask_element

    subroutine dim_reduction_accumulate_mask(context, reduce_op, acc, cond, &
                                             error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: reduce_op
        type(lr_operand_desc_t), intent(inout) :: acc
        type(lr_operand_desc_t), intent(in) :: cond
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: next

        call set_empty(error_msg)
        select case (reduce_op)
        case (DIM_REDUCE_ANY)
            if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, cond, next, &
                                      error_msg)) return
        case (DIM_REDUCE_ALL)
            if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, cond, next, &
                                      error_msg)) return
        case default
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc, cond, next, &
                                      error_msg)) return
        end select
        acc = next
    end subroutine dim_reduction_accumulate_mask
