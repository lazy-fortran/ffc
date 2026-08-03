    subroutine describe_array_section(arena, slice_node, context, info, &
                                      error_msg, allow_derived)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        type(lowering_context_t), intent(in) :: context
        type(array_section_info_t), intent(out) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        ! Derived-element sections carry no value semantics here; only the
        ! argument paths, which pass a section by reference, may ask for one.
        logical, intent(in), optional :: allow_derived
        logical :: derived_ok
        character(len=:), allocatable :: source_name
        integer :: source_index
        integer :: source_rank
        integer :: dim
        integer :: bounds_index
        integer :: runtime_bounds_index
        integer :: kept_dim
        integer(c_int64_t) :: lower_value
        integer(c_int64_t) :: upper_value
        integer(c_int64_t) :: stride_value
        integer(c_int64_t) :: scalar_value
        integer(c_int64_t) :: extent_value
        logical :: runtime_extent_known

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
        info%has_runtime_bounds = .false.

        call identifier_name(arena, slice_node%array_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol_compat(context, source_name)
        if (source_index <= 0) then
            error_msg = 'array section base was not declared: '//trim(source_name)
            return
        end if
        if (.not. context%symbols(source_index)%is_array) then
            error_msg = 'array section requires an array: '//trim(source_name)
            return
        end if
        derived_ok = .false.
        if (present(allow_derived)) derived_ok = allow_derived
        select case (context%symbols(source_index)%value_kind)
        case (VALUE_I32, VALUE_F32, VALUE_F64)
            ! supported element types for array sections
        case (VALUE_DERIVED)
            if (.not. derived_ok) then
                error_msg = 'array sections support integer, real(4), and '// &
                            'real(8) arrays'
                return
            end if
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
                if (len_trim(error_msg) > 0) then
                    info%has_runtime_bounds = .true.
                    lower_value = info%source_lowers(dim)
                    call set_empty(error_msg)
                end if
                call resolve_section_range_bound(arena, context, bounds_node%end_index, &
                                                 info%source_lowers(dim) + &
                                                 info%source_sizes(dim) - 1_c_int64_t, &
                                                 upper_value, error_msg)
                if (len_trim(error_msg) > 0) then
                    info%has_runtime_bounds = .true.
                    upper_value = info%source_lowers(dim) + &
                                  info%source_sizes(dim) - 1_c_int64_t
                    call set_empty(error_msg)
                end if
                call resolve_section_stride(arena, context, bounds_node%stride_index, &
                                            stride_value, error_msg)
                if (len_trim(error_msg) > 0) then
                    info%has_runtime_bounds = .true.
                    stride_value = 1_c_int64_t
                    call set_empty(error_msg)
                end if
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
                    if (len_trim(error_msg) > 0) then
                        info%has_runtime_bounds = .true.
                        lower_value = info%source_lowers(dim)
                        call set_empty(error_msg)
                    end if
                    call resolve_section_range_bound(arena, context, &
                                                     bounds_node%upper_bound_index, &
                                                     info%source_lowers(dim) + &
                                                     info%source_sizes(dim) - 1_c_int64_t, &
                                                     upper_value, error_msg)
                    if (len_trim(error_msg) > 0) then
                        info%has_runtime_bounds = .true.
                        upper_value = info%source_lowers(dim) + &
                                      info%source_sizes(dim) - 1_c_int64_t
                        call set_empty(error_msg)
                    end if
                    call resolve_section_stride(arena, context, bounds_node%stride_index, &
                                                stride_value, error_msg)
                    if (len_trim(error_msg) > 0) then
                        info%has_runtime_bounds = .true.
                        stride_value = 1_c_int64_t
                        call set_empty(error_msg)
                    end if
                end if
            class default
                call resolve_section_scalar(arena, context, bounds_index, scalar_value, &
                                            error_msg)
                if (len_trim(error_msg) > 0) then
                    ! Runtime scalar subscripts are lowered by the descriptor
                    ! element path below.  Keep the static description usable
                    ! for validation and let the value/assignment caller route
                    ! the actual element through that path.
                    info%has_runtime_bounds = .true.
                    scalar_value = info%source_lowers(dim)
                    call set_empty(error_msg)
                end if
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
                    extent_value = 0_c_int64_t
                else
                    extent_value = (upper_value - lower_value) / stride_value + 1_c_int64_t
                end if
            else
                if (lower_value < upper_value) then
                    extent_value = 0_c_int64_t
                else
                    extent_value = (lower_value - upper_value) / abs(stride_value) + 1_c_int64_t
                end if
            end if

            info%keep_dim(dim) = .true.
            info%section_lowers(dim) = lower_value
            info%section_uppers(dim) = upper_value
            info%section_strides(dim) = stride_value
            info%section_extents(dim) = extent_value
            kept_dim = kept_dim + 1
            info%kept_dims(kept_dim) = dim
        end do

        if (info%has_runtime_bounds) then
            ! Runtime coordinates still often have a compile-time extent, e.g.
            ! y:y+8-1.  Preserve that extent for bounded expansion while the
            ! actual coordinates are computed from live values at runtime.
            do dim = 1, source_rank
                if (info%keep_dim(dim)) then
                    runtime_bounds_index = 0
                    if (dim <= slice_node%num_dimensions) then
                        runtime_bounds_index = slice_node%bounds_indices(dim)
                    end if
                    call try_runtime_range_extent(arena, context, &
                        runtime_bounds_index, extent_value, runtime_extent_known)
                    if (runtime_extent_known) then
                        info%section_extents(dim) = extent_value
                    else
                        info%section_extents(dim) = 0_c_int64_t
                    end if
                end if
            end do
        end if

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

    subroutine try_runtime_range_extent(arena, context, bounds_index, extent, &
                                        extent_known)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: bounds_index
        integer(c_int64_t), intent(out) :: extent
        logical, intent(out) :: extent_known
        integer :: lower_index, upper_index, stride_index
        integer(c_int64_t) :: lower_const, upper_const
        integer(c_int64_t) :: stride_value
        character(len=64) :: lower_names(8), upper_names(8)
        integer(c_int64_t) :: lower_coeffs(8), upper_coeffs(8)
        integer :: lower_count, upper_count, i, j
        logical :: lower_known, upper_known, term_found
        character(len=:), allocatable :: error_msg

        extent = 0_c_int64_t
        extent_known = .false.
        if (.not. node_exists(arena, bounds_index)) return
        lower_index = 0
        upper_index = 0
        stride_index = 0
        select type (bounds_node => arena%entries(bounds_index)%node)
        type is (range_expression_node)
            lower_index = bounds_node%start_index
            upper_index = bounds_node%end_index
            stride_index = bounds_node%stride_index
        type is (array_bounds_node)
            if (bounds_node%is_assumed_rank .or. bounds_node%is_assumed_size .or. &
                bounds_node%is_assumed_shape .or. bounds_node%is_deferred_shape) return
            lower_index = bounds_node%lower_bound_index
            upper_index = bounds_node%upper_bound_index
            stride_index = bounds_node%stride_index
        class default
            return
        end select
        if (lower_index <= 0 .or. upper_index <= 0) return
        call affine_bound_expression(arena, lower_index, context, lower_names, &
                                     lower_coeffs, lower_count, lower_const, lower_known)
        call affine_bound_expression(arena, upper_index, context, upper_names, &
                                     upper_coeffs, upper_count, upper_const, upper_known)
        if (.not. lower_known .or. .not. upper_known) return
        if (lower_count /= upper_count) return
        do i = 1, lower_count
            term_found = .false.
            do j = 1, upper_count
                if (trim(lower_names(i)) == trim(upper_names(j)) .and. &
                    lower_coeffs(i) == upper_coeffs(j)) then
                    term_found = .true.
                    exit
                end if
            end do
            if (.not. term_found) return
        end do
        call resolve_section_stride(arena, context, stride_index, stride_value, &
                                    error_msg)
        if (len_trim(error_msg) > 0 .or. stride_value == 0_c_int64_t) return
        if (stride_value > 0_c_int64_t) then
            if (upper_const < lower_const) return
            extent = (upper_const - lower_const) / stride_value + 1_c_int64_t
        else
            if (upper_const > lower_const) return
            extent = (lower_const - upper_const) / abs(stride_value) + 1_c_int64_t
        end if
        extent_known = extent > 0_c_int64_t
    end subroutine try_runtime_range_extent

    recursive subroutine affine_bound_expression(arena, node_index, context, &
                                                 variable_names, coefficients, &
                                                 term_count, constant, known)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(out) :: variable_names(:)
        integer(c_int64_t), intent(out) :: coefficients(:)
        integer, intent(out) :: term_count
        integer(c_int64_t), intent(out) :: constant
        logical, intent(out) :: known
        character(len=:), allocatable :: name, op, err
        character(len=64) :: left_names(size(variable_names))
        character(len=64) :: right_names(size(variable_names))
        character(len=64) :: source_names(size(variable_names))
        integer :: left_index, right_index, line, column
        integer(c_int64_t) :: left_coeffs(size(variable_names))
        integer(c_int64_t) :: right_coeffs(size(variable_names))
        integer(c_int64_t) :: source_coeffs(size(variable_names))
        integer(c_int64_t) :: left_const, right_const
        integer :: left_count, right_count, i, j, sign, source_count
        logical :: left_known, right_known

        variable_names = ''
        coefficients = 0_c_int64_t
        term_count = 0
        constant = 0_c_int64_t
        known = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, err)
            if (len_trim(err) > 0) return
            call eval_i32_constant(arena, node_index, context, constant, err)
            if (len_trim(err) == 0) then
                known = .true.
                return
            end if
            variable_names(1) = trim(name)
            coefficients(1) = 1_c_int64_t
            term_count = 1
            known = .true.
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, left_index, right_index, &
                                    line, column, err)
            if (len_trim(err) > 0) return
            call affine_bound_expression(arena, left_index, context, left_names, &
                                         left_coeffs, left_count, left_const, left_known)
            call affine_bound_expression(arena, right_index, context, right_names, &
                                         right_coeffs, right_count, right_const, right_known)
            if (.not. left_known .or. .not. right_known) return
            select case (trim(op))
            case ('+', '.plus.')
                sign = 1
                source_names = right_names
                source_coeffs = right_coeffs
                source_count = right_count
                constant = left_const + right_const
            case ('-', '.minus.')
                sign = -1
                source_names = right_names
                source_coeffs = right_coeffs
                source_count = right_count
                constant = left_const - right_const
            case ('*', '.times.')
                if (left_count > 0 .and. right_count > 0) return
                if (left_count == 0) then
                    variable_names(1:right_count) = right_names(1:right_count)
                    coefficients(1:right_count) = left_const * &
                        right_coeffs(1:right_count)
                    term_count = right_count
                    constant = left_const * right_const
                else
                    variable_names(1:left_count) = left_names(1:left_count)
                    coefficients(1:left_count) = right_const * &
                        left_coeffs(1:left_count)
                    term_count = left_count
                    constant = right_const * left_const
                end if
                known = .true.
                return
            case default
                return
            end select

            variable_names(1:left_count) = left_names(1:left_count)
            coefficients(1:left_count) = left_coeffs(1:left_count)
            term_count = left_count
            do i = 1, source_count
                if (source_coeffs(i) == 0_c_int64_t) cycle
                do j = 1, term_count
                    if (trim(variable_names(j)) == trim(source_names(i))) exit
                end do
                if (j <= term_count) then
                    coefficients(j) = coefficients(j) + sign * source_coeffs(i)
                else
                    if (term_count >= size(variable_names)) return
                    term_count = term_count + 1
                    variable_names(term_count) = source_names(i)
                    coefficients(term_count) = sign * source_coeffs(i)
                end if
            end do
            do i = term_count, 1, -1
                if (coefficients(i) /= 0_c_int64_t) cycle
                if (i < term_count) then
                    variable_names(i:term_count-1) = variable_names(i+1:term_count)
                    coefficients(i:term_count-1) = coefficients(i+1:term_count)
                end if
                term_count = term_count - 1
            end do
            known = .true.
            return
        end if
        call eval_i32_constant(arena, node_index, context, constant, err)
        known = len_trim(err) == 0
    end subroutine affine_bound_expression

    integer(c_int64_t) function array_section_total_elements(info) result(total)
        type(array_section_info_t), intent(in) :: info
        integer :: i

        total = 1_c_int64_t
        if (info%result_rank <= 0) return
        do i = 1, info%result_rank
            if (info%section_extents(info%kept_dims(i)) <= 0_c_int64_t) then
                total = 0_c_int64_t
                return
            end if
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
        if (source_linear < 0_c_int64_t .or. source_linear >= int( &
            effective_array_extent(context, info%source_index), c_int64_t)) then
            ! An out-of-bounds section is diagnosed by the reference compiler,
            ! but remains compilable Fortran.  Keep ffc's compile-only path
            ! recoverable; the runtime value of such a reference is undefined.
            select case (context%symbols(info%source_index)%value_kind)
            case (VALUE_F32)
                value = liric_f32_immediate(context%session, 0.0_c_float)
            case (VALUE_F64)
                value = liric_f64_immediate(context%session, 0.0_c_double)
            case (VALUE_I64)
                value = i64_immediate(context%session, 0_c_int64_t)
            case (VALUE_I8)
                value = i8_immediate(context%session, 0_c_int64_t)
            case (VALUE_I16)
                value = i16_immediate(context%session, 0_c_int64_t)
            case default
                value = i32_immediate(context%session, 0_c_int64_t)
            end select
            call set_empty(error_msg)
            return
        end if
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
        if (info%has_runtime_bounds) then
            if (context%symbols(target_symbol_index)%array_rank /= 1) then
                error_msg = 'runtime array sections currently require a rank-1 '// &
                            'array-expression target'
                return
            end if
            call lower_runtime_array_section_element(arena, slice_node, linear_index, &
                                                     context, value, error_msg)
            return
        end if
        call ensure_array_section_shapes_match(context, target_symbol_index, info, &
                                               error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_array_section_element_from_info(info, linear_index, context, &
                                                   value, error_msg)
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end subroutine lower_array_section_value

    subroutine lower_runtime_array_section_element(arena, slice_node, linear_index, &
                                                   context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        integer(c_int64_t), intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: element_address
        character(len=:), allocatable :: source_name
        integer :: source_index

        call lower_runtime_array_section_element_address(arena, slice_node, &
            linear_index, context, element_address, error_msg)
        if (len_trim(error_msg) > 0) return
        call identifier_name(arena, slice_node%array_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol_compat(context, source_name)
        if (source_index <= 0) then
            error_msg = 'array section base was not declared: '//trim(source_name)
            return
        end if
        call emit_array_value_load(context, context%symbols(source_index)%value_kind, &
                                   element_address, value, error_msg)
    end subroutine lower_runtime_array_section_element

    subroutine lower_runtime_array_section_element_address(arena, slice_node, &
                                                           linear_index, context, &
                                                           element_address, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        integer(c_int64_t), intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg

        call lower_runtime_array_section_element_address_operand(arena, slice_node, &
            i32_immediate(context%session, linear_index), context, element_address, &
            error_msg)
    end subroutine lower_runtime_array_section_element_address

    subroutine lower_runtime_array_section_element_address_operand(arena, slice_node, &
                                                                   linear_index, context, &
                                                                   element_address, error_msg)
        !! Lower one element of a runtime-bounded section.  The direct session
        !! still expands the result's known target extent element by element,
        !! but computes each source coordinate from the live section bounds.
        !! This covers the rank-2 -> rank-1 views used by blocked matmul code.
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        type(lr_operand_desc_t), intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source_name
        integer :: source_index, source_rank, dim, bounds_index, kept_dim
        logical :: keep_dim(2)
        type(lr_operand_desc_t) :: lower_op(2), upper_op(2), stride_op(2)
        type(lr_operand_desc_t) :: extent_op(2), source_extent_op(2), coord_op(2)
        type(lr_operand_desc_t) :: temp_op, diff_op, term_op, source_linear
        integer(c_int64_t) :: source_lower, source_extent

        call identifier_name(arena, slice_node%array_index, source_name, error_msg)
        if (len_trim(error_msg) > 0) return
        source_index = find_symbol_compat(context, source_name)
        if (source_index <= 0) then
            error_msg = 'array section base was not declared: '//trim(source_name)
            return
        end if
        source_rank = context%symbols(source_index)%array_rank
        if (source_rank < 1 .or. source_rank > 2) then
            error_msg = 'runtime array sections support rank-1 and rank-2 arrays'
            return
        end if
        if (slice_node%num_dimensions > source_rank) then
            error_msg = 'array section has too many subscripts'
            return
        end if

        keep_dim = .false.
        kept_dim = 0
        do dim = 1, source_rank
            source_lower = int(context%symbols(source_index)%array_dim_lowers(dim), &
                               c_int64_t)
            if (context%symbols(source_index)%has_runtime_dim_size(dim)) then
                source_extent_op(dim) = &
                    context%symbols(source_index)%runtime_dim_size(dim)
            else
                source_extent = int(context%symbols(source_index)%array_dim_sizes(dim), &
                                     c_int64_t)
                source_extent_op(dim) = i32_immediate(context%session, source_extent)
            end if

            if (dim > slice_node%num_dimensions .or. &
                slice_node%bounds_indices(dim) <= 0) then
                keep_dim(dim) = .true.
                kept_dim = kept_dim + 1
                lower_op(dim) = i32_immediate(context%session, source_lower)
                stride_op(dim) = i32_immediate(context%session, 1_c_int64_t)
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, lower_op(dim), &
                    source_extent_op(dim), temp_op, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_SUB, temp_op, &
                    i32_immediate(context%session, 1_c_int64_t), upper_op(dim), &
                    error_msg)) return
                cycle
            end if

            bounds_index = slice_node%bounds_indices(dim)
            if (.not. node_exists(arena, bounds_index)) then
                error_msg = 'array section bound index does not reference an AST node'
                return
            end if
            select type (bounds_node => arena%entries(bounds_index)%node)
            type is (range_expression_node)
                keep_dim(dim) = .true.
                kept_dim = kept_dim + 1
                if (bounds_node%start_index > 0) then
                    call lower_i32_expression(arena, bounds_node%start_index, context, &
                                              lower_op(dim), error_msg)
                    if (len_trim(error_msg) > 0) return
                else
                    lower_op(dim) = i32_immediate(context%session, source_lower)
                end if
                if (bounds_node%stride_index > 0) then
                    call lower_i32_expression(arena, bounds_node%stride_index, context, &
                                              stride_op(dim), error_msg)
                    if (len_trim(error_msg) > 0) return
                else
                    stride_op(dim) = i32_immediate(context%session, 1_c_int64_t)
                end if
                if (bounds_node%end_index > 0) then
                    call lower_i32_expression(arena, bounds_node%end_index, context, &
                                              upper_op(dim), error_msg)
                    if (len_trim(error_msg) > 0) return
                else
                    if (.not. emit_i32_binary(context%session, LR_OP_ADD, &
                        i32_immediate(context%session, source_lower), &
                        source_extent_op(dim), temp_op, error_msg)) return
                    if (.not. emit_i32_binary(context%session, LR_OP_SUB, temp_op, &
                        i32_immediate(context%session, 1_c_int64_t), upper_op(dim), &
                        error_msg)) return
                end if
            type is (array_bounds_node)
                if (bounds_node%is_assumed_rank .or. bounds_node%is_assumed_size .or. &
                    bounds_node%is_assumed_shape .or. bounds_node%is_deferred_shape) then
                    keep_dim(dim) = .true.
                    kept_dim = kept_dim + 1
                    lower_op(dim) = i32_immediate(context%session, source_lower)
                    stride_op(dim) = i32_immediate(context%session, 1_c_int64_t)
                    if (.not. emit_i32_binary(context%session, LR_OP_ADD, &
                        i32_immediate(context%session, source_lower), &
                        source_extent_op(dim), temp_op, error_msg)) return
                    if (.not. emit_i32_binary(context%session, LR_OP_SUB, temp_op, &
                        i32_immediate(context%session, 1_c_int64_t), upper_op(dim), &
                        error_msg)) return
                else
                    keep_dim(dim) = .true.
                    kept_dim = kept_dim + 1
                    if (bounds_node%lower_bound_index > 0) then
                        call lower_i32_expression(arena, bounds_node%lower_bound_index, &
                            context, lower_op(dim), error_msg)
                        if (len_trim(error_msg) > 0) return
                    else
                        lower_op(dim) = i32_immediate(context%session, source_lower)
                    end if
                    if (bounds_node%upper_bound_index > 0) then
                        call lower_i32_expression(arena, bounds_node%upper_bound_index, &
                            context, upper_op(dim), error_msg)
                        if (len_trim(error_msg) > 0) return
                    else
                        if (.not. emit_i32_binary(context%session, LR_OP_ADD, &
                            i32_immediate(context%session, source_lower), &
                            source_extent_op(dim), temp_op, error_msg)) return
                        if (.not. emit_i32_binary(context%session, LR_OP_SUB, temp_op, &
                            i32_immediate(context%session, 1_c_int64_t), upper_op(dim), &
                            error_msg)) return
                    end if
                    if (bounds_node%stride_index > 0) then
                        call lower_i32_expression(arena, bounds_node%stride_index, &
                            context, stride_op(dim), error_msg)
                        if (len_trim(error_msg) > 0) return
                    else
                        stride_op(dim) = i32_immediate(context%session, 1_c_int64_t)
                    end if
                end if
            class default
                call lower_i32_expression(arena, bounds_index, context, &
                                           coord_op(dim), error_msg)
                if (len_trim(error_msg) > 0) return
            end select

            if (keep_dim(dim)) then
                if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_op(dim), &
                    lower_op(dim), temp_op, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_SDIV, temp_op, &
                    stride_op(dim), diff_op, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, diff_op, &
                    i32_immediate(context%session, 1_c_int64_t), extent_op(dim), &
                    error_msg)) return
            end if
        end do

        if (kept_dim /= 1) then
            error_msg = 'runtime array sections currently support one kept dimension'
            return
        end if
        do dim = 1, source_rank
            if (keep_dim(dim)) then
                if (.not. emit_i32_binary(context%session, LR_OP_MUL, &
                    linear_index, stride_op(dim), &
                    temp_op, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, lower_op(dim), &
                    temp_op, coord_op(dim), error_msg)) return
            end if
        end do

        source_lower = int(context%symbols(source_index)%array_dim_lowers(1), c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, coord_op(1), &
            i32_immediate(context%session, source_lower), diff_op, error_msg)) return
        if (source_rank == 1) then
            source_linear = diff_op
        else
            source_lower = int(context%symbols(source_index)%array_dim_lowers(2), c_int64_t)
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, coord_op(2), &
                i32_immediate(context%session, source_lower), temp_op, error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, temp_op, &
                    source_extent_op(1), term_op, error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, diff_op, term_op, &
                    source_linear, error_msg)) return
        end if
        call lower_runtime_descriptor_array_element_address(context, source_index, &
            source_linear, element_address, error_msg)
    end subroutine lower_runtime_array_section_element_address_operand

    recursive integer function array_expression_symbol(arena, node_index, context) &
            result(symbol_index)
        !! Find one stored array operand that determines an elementwise
        !! expression's extent.  Runtime section assignment uses this only to
        !! obtain the already-known rank-1 RHS extent.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err, op
        integer :: left_index, right_index, line, column, candidate

        symbol_index = 0
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, err)
            if (len_trim(err) > 0) return
            candidate = find_symbol_compat(context, name)
            if (candidate > 0 .and. is_elementwise_array_operand(context, candidate)) &
                symbol_index = candidate
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, left_index, right_index, &
                                    line, column, err)
            if (len_trim(err) > 0) return
            symbol_index = array_expression_symbol(arena, left_index, context)
            if (symbol_index <= 0) then
                symbol_index = array_expression_symbol(arena, right_index, context)
            end if
            return
        end if
        symbol_index = cached_array_expression_symbol(context, node_index)
    end function array_expression_symbol

    recursive subroutine array_section_assignment_rhs_extent(arena, node_index, &
                                                             context, extent, known, &
                                                             error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer(c_int64_t), intent(out) :: extent
        logical, intent(out) :: known
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, op, err
        integer :: symbol_index, left_index, right_index, line, column
        integer(c_int64_t) :: left_extent, right_extent
        logical :: left_known, right_known
        integer, allocatable :: flat(:)

        extent = 0_c_int64_t
        known = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = resolve_symbol_at_node(context, node_index, name)
            if (symbol_index <= 0) return
            if (.not. is_elementwise_array_operand(context, symbol_index)) return
            extent = int(effective_array_extent(context, symbol_index), c_int64_t)
            known = extent > 0_c_int64_t
            return
        end if
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, left_index, right_index, &
                                    line, column, error_msg)
            if (len_trim(error_msg) > 0) return
            call array_section_assignment_rhs_extent(arena, left_index, context, &
                                                     left_extent, left_known, err)
            if (len_trim(err) > 0) return
            call array_section_assignment_rhs_extent(arena, right_index, context, &
                                                     right_extent, right_known, err)
            if (len_trim(err) > 0) return
            if (left_known .and. right_known) then
                if (left_extent /= right_extent) then
                    error_msg = 'array expression operands have nonconforming extents'
                    return
                end if
                extent = left_extent
                known = .true.
            else if (left_known) then
                extent = left_extent
                known = .true.
            else if (right_known) then
                extent = right_extent
                known = .true.
            end if
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (array_slice_node)
            call section_available_element_count(arena, node, context, extent, known, &
                                                 error_msg)
        type is (array_literal_node)
            if (.not. allocated(node%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, node%element_indices, flat, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            extent = int(size(flat), c_int64_t)
            known = extent > 0_c_int64_t
        end select
    end subroutine array_section_assignment_rhs_extent

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
            ! Empty sections are valid Fortran values.  In a list-directed
            ! print they contribute no fields but still belong to the record.
            handled = .true.
            call set_empty(error_msg)
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
        sym = find_symbol_compat(context, name)
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
        if (vk == VALUE_C4 .or. vk == VALUE_C8) then
            handled = .true.
            n = int(context%symbols(sym)%array_size, c_int64_t)
            do k = 0_c_int64_t, n - 1_c_int64_t
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
                block
                    type(lr_operand_desc_t) :: re_addr, im_addr, re_value, im_value
                    call complex_array_linear_addresses(context, sym, int(k), vk, &
                        re_addr, im_addr, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (vk == VALUE_C4) then
                        if (.not. emit_liric_f32_load(context%session, re_addr, &
                                                     re_value, error_msg)) return
                        if (.not. emit_liric_f32_load(context%session, im_addr, &
                                                     im_value, error_msg)) return
                        if (.not. emit_complex4_print_call(context%session, &
                            re_value, im_value, error_msg)) return
                    else
                        if (.not. emit_liric_f64_load(context%session, re_addr, &
                                                     re_value, error_msg)) return
                        if (.not. emit_liric_f64_load(context%session, im_addr, &
                                                     im_value, error_msg)) return
                        if (.not. emit_complex8_print_call(context%session, &
                            re_value, im_value, error_msg)) return
                    end if
                end block
            end do
            return
        end if
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. vk /= VALUE_F64 .and. &
            vk /= VALUE_LOGICAL .and. vk /= VALUE_I64 .and. vk /= VALUE_I8 .and. &
            vk /= VALUE_I16) return
        if (context%symbols(sym)%is_allocatable) then
            call emit_allocatable_print_items(context, sym, vk, handled, error_msg)
            return
        end if
        if (context%symbols(sym)%is_runtime_array) then
            handled = .true.
            call emit_runtime_local_array_print(context, sym, vk, error_msg)
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
        integer :: target_sym
        integer :: print_sym
        integer :: constructor_extent
        integer :: vk
        integer(c_int64_t) :: k, n
        logical :: logical_expr

        handled = .false.
        call set_empty(error_msg)
        sym = whole_array_expr_shape_symbol(arena, expr_index, context)
        constructor_extent = whole_array_expr_constructor_extent(arena, expr_index)
        logical_expr = whole_array_expr_is_logical(arena, expr_index)
        if (sym > 0) then
            n = int(effective_array_extent(context, sym), c_int64_t)
        else
            if (constructor_extent <= 0) return
            n = int(constructor_extent, c_int64_t)
        end if
        if (n <= 0) return

        target_sym = sym
        if (logical_expr) then
            if (target_sym <= 0 .or. &
                context%symbols(target_sym)%value_kind /= VALUE_LOGICAL) then
                call create_array_expression_temp(context, VALUE_LOGICAL, int(n), &
                                                   0, target_sym, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        else if (target_sym <= 0) then
            return
        end if

        vk = context%symbols(target_sym)%value_kind
        if (vk /= VALUE_I32 .and. vk /= VALUE_F32 .and. vk /= VALUE_F64 .and. &
            vk /= VALUE_LOGICAL .and. vk /= VALUE_I64 .and. vk /= VALUE_I8 .and. &
            vk /= VALUE_I16) return

        handled = .true.
        if (sym > 0) then
            call create_array_expression_temp(context, vk, int(n), 0, print_sym, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            do k = 0_c_int64_t, n - 1_c_int64_t
                call lower_array_elementwise_value(arena, expr_index, target_sym, &
                                                   int(k), context, value, error_msg)
                if (len_trim(error_msg) > 0) return
                call store_array_linear_element(context, print_sym, k, value, &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            do k = 0_c_int64_t, n - 1_c_int64_t
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
                call load_array_linear_element(context, print_sym, k, value, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                call emit_typed_array_print_value(context, vk, value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else
            do k = 0_c_int64_t, n - 1_c_int64_t
                if (.not. emit_liric_print_space(context%session, error_msg)) &
                    return
                call lower_array_elementwise_value(arena, expr_index, target_sym, &
                                                   int(k), context, value, error_msg)
                if (len_trim(error_msg) > 0) return
                call emit_typed_array_print_value(context, vk, value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
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
            i = find_symbol_compat(context, id_name)
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

    recursive integer function whole_array_expr_constructor_extent(arena, &
                                                                    expr_index) &
        result(extent)
        !! Return the flattened extent contributed by array constructors in an
        !! expression. A zero result means that the expression has no constructor
        !! that can establish an array shape on its own.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: error_msg, bin_op
        integer :: bin_left, bin_right, bin_line, bin_col

        extent = 0
        if (.not. node_exists(arena, expr_index)) return

        select type (literal_node => arena%entries(expr_index)%node)
        type is (array_literal_node)
            if (.not. allocated(literal_node%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, literal_node%element_indices, &
                                              flat, error_msg)
            if (len_trim(error_msg) > 0) return
            extent = size(flat)
            return
        end select

        if (.not. is_binary_op(arena, expr_index)) return
        call get_binary_op_info(arena, expr_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, error_msg)
        if (len_trim(error_msg) > 0) return
        if (trim(adjustl(lowercase_text(bin_op))) == '.not.' .or. &
            trim(adjustl(lowercase_text(bin_op))) == 'not') then
            extent = whole_array_expr_constructor_extent(arena, bin_right)
        else
            extent = max(whole_array_expr_constructor_extent(arena, bin_left), &
                         whole_array_expr_constructor_extent(arena, bin_right))
        end if
    end function whole_array_expr_constructor_extent

    recursive logical function whole_array_expr_is_logical(arena, expr_index) &
        result(is_logical)
        !! Identify a non-scalar expression whose element type is logical. The
        !! caller uses this to give comparisons and logical connectives a logical
        !! target even when their only array-shaped operand is numeric.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        character(len=:), allocatable :: bin_op, error_msg
        integer :: bin_left, bin_right, bin_line, bin_col

        is_logical = .false.
        if (.not. is_binary_op(arena, expr_index)) return
        call get_binary_op_info(arena, expr_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, error_msg)
        if (len_trim(error_msg) > 0) return
        is_logical = is_logical_connective(bin_op) .or. &
                     is_comparison_operator(bin_op) .or. &
                     trim(adjustl(lowercase_text(bin_op))) == '.not.' .or. &
                     trim(adjustl(lowercase_text(bin_op))) == 'not'
    end function whole_array_expr_is_logical

    subroutine emit_array_literal_print_items(arena, expr_index, context, &
                                              handled, error_msg)
        !! List-directed print of an inline array constructor [e1, e2, ...].
        !! Each explicit element is printed like a scalar item (leading
        !! separator space, then the value in its own kind), so a constructor of
        !! literals, variables, or function calls prints byte-for-byte like
        !! gfortran. An implied-do constructor is expanded through the same
        !! list-directed path as an I/O implied-do item.
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
            if (is_implied_do_constructor(arena, ctor)) then
                call emit_array_constructor_implied_do_print_items(arena, ctor, &
                    context, handled, error_msg)
                return
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

    integer function cached_array_expression_symbol(context, expr_index) &
            result(symbol_index)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: expr_index
        integer :: i

        symbol_index = 0
        if (expr_index <= 0) return
        if (.not. allocated(context%array_expression_cache_nodes)) return
        do i = context%array_expression_cache_count, 1, -1
            if (context%array_expression_cache_nodes(i) == expr_index) then
                symbol_index = context%array_expression_cache_symbols(i)
                return
            end if
        end do
    end function cached_array_expression_symbol

    subroutine cache_array_expression_symbol(context, expr_index, symbol_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: expr_index
        integer, intent(in) :: symbol_index
        integer, allocatable :: old_nodes(:), old_symbols(:)
        integer :: capacity

        if (.not. allocated(context%array_expression_cache_nodes)) then
            allocate (context%array_expression_cache_nodes(8))
            allocate (context%array_expression_cache_symbols(8))
        else if (context%array_expression_cache_count >= &
                 size(context%array_expression_cache_nodes)) then
            capacity = size(context%array_expression_cache_nodes)
            call move_alloc(context%array_expression_cache_nodes, old_nodes)
            call move_alloc(context%array_expression_cache_symbols, old_symbols)
            allocate (context%array_expression_cache_nodes(2 * capacity))
            allocate (context%array_expression_cache_symbols(2 * capacity))
            context%array_expression_cache_nodes(1:capacity) = old_nodes
            context%array_expression_cache_symbols(1:capacity) = old_symbols
        end if
        context%array_expression_cache_count = &
            context%array_expression_cache_count + 1
        context%array_expression_cache_nodes( &
            context%array_expression_cache_count) = expr_index
        context%array_expression_cache_symbols( &
            context%array_expression_cache_count) = symbol_index
    end subroutine cache_array_expression_symbol

    subroutine create_array_expression_temp(context, element_kind, array_size, &
                                             character_length, symbol_index, &
                                             error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: element_kind
        integer, intent(in) :: array_size
        integer, intent(in) :: character_length
        integer, intent(out) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64) :: temp_name
        integer(c_int64_t) :: byte_count

        call set_empty(error_msg)
        symbol_index = 0
        if (array_size <= 0) then
            error_msg = 'array expression temporary requires a positive extent'
            return
        end if

        call grow_symbols(context)
        symbol_index = context%symbol_count + 1
        context%symbols(symbol_index) = symbol_t()
        write (temp_name, '("__ffc.array.expr.",I0)') symbol_index
        context%symbols(symbol_index)%name = trim(temp_name)
        context%symbols(symbol_index)%value_kind = element_kind
        context%symbols(symbol_index)%is_array = .true.
        context%symbols(symbol_index)%array_rank = 1
        context%symbols(symbol_index)%array_size = array_size
        context%symbols(symbol_index)%array_lower_bound = 1
        context%symbols(symbol_index)%array_dim_lowers(1) = 1
        context%symbols(symbol_index)%array_dim_sizes(1) = array_size

        select case (element_kind)
        case (VALUE_CHARACTER)
            if (character_length <= 0) then
                error_msg = 'array expression temporary requires a fixed '// &
                            'character length'
                return
            end if
            context%symbols(symbol_index)%character_length = character_length
            byte_count = int(array_size, c_int64_t) * &
                         int(character_length, c_int64_t)
            if (.not. emit_alloca_bytes(context%session, &
                    i64_immediate(context%session, byte_count), &
                    context%symbols(symbol_index)%element_address, error_msg)) then
                error_msg = 'array expression character temporary: '//error_msg
                return
            end if
        case (VALUE_F32)
            if (.not. emit_f32_array_alloca(context%session, array_size, &
                    context%symbols(symbol_index)%element_address, error_msg)) then
                error_msg = 'array expression real(4) temporary: '//error_msg
                return
            end if
        case (VALUE_F64)
            if (.not. emit_f64_array_alloca(context%session, array_size, &
                    context%symbols(symbol_index)%element_address, error_msg)) then
                error_msg = 'array expression real(8) temporary: '//error_msg
                return
            end if
        case default
            if (.not. emit_i32_array_alloca(context%session, array_size, &
                    context%symbols(symbol_index)%element_address, error_msg)) then
                error_msg = 'array expression logical/integer temporary: '// &
                            error_msg
                return
            end if
        end select
        context%symbol_count = symbol_index
    end subroutine create_array_expression_temp

    recursive subroutine materialize_array_expression_calls(arena, expr_index, &
                                                             context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: cached, element_kind, array_size, character_length
        integer :: i, left_index, right_index, line, column, symbol_index
        character(len=:), allocatable :: op

        call set_empty(error_msg)
        if (.not. node_exists(arena, expr_index)) return

        select type (call_node => arena%entries(expr_index)%node)
        type is (call_or_subscript_node)
            if (is_array_result_call(arena, expr_index, context)) then
                cached = cached_array_expression_symbol(context, expr_index)
                if (cached > 0) return
                call contained_array_result_info(arena, context, call_node%name, &
                    element_kind, array_size, error_msg, character_length)
                if (len_trim(error_msg) > 0) return
                call create_array_expression_temp(context, element_kind, &
                    array_size, character_length, symbol_index, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_array_result_call(arena, expr_index, symbol_index, &
                                             context, error_msg)
                if (len_trim(error_msg) > 0) return
                call cache_array_expression_symbol(context, expr_index, symbol_index)
                return
            end if
            if (allocated(call_node%arg_indices)) then
                do i = 1, size(call_node%arg_indices)
                    call materialize_array_expression_calls(arena, &
                        call_node%arg_indices(i), context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
            return
        end select

        if (is_binary_op(arena, expr_index)) then
            call get_binary_op_info(arena, expr_index, op, left_index, &
                right_index, line, column, error_msg)
            if (len_trim(error_msg) > 0) return
            call materialize_array_expression_calls(arena, left_index, context, &
                                                    error_msg)
            if (len_trim(error_msg) > 0) return
            call materialize_array_expression_calls(arena, right_index, context, &
                                                    error_msg)
            return
        end if

        select type (literal_node => arena%entries(expr_index)%node)
        type is (array_literal_node)
            if (allocated(literal_node%element_indices)) then
                do i = 1, size(literal_node%element_indices)
                    call materialize_array_expression_calls(arena, &
                        literal_node%element_indices(i), context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
        end select
    end subroutine materialize_array_expression_calls

    subroutine build_array_expression(context, target_symbol_index, expr_index, &
                                      plan, error_msg)
        !! Derive the rank, shape, and element type of one array-valued
        !! expression once, before any element is emitted. The shape and the
        !! element type of a whole-array assignment are those of its target:
        !! array-array operands must conform to it and a scalar operand
        !! broadcasts, both of which are checked as operands are reached by
        !! `ensure_array_shapes_match`.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: expr_index
        type(array_expr_plan_t), intent(out) :: plan
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: d

        plan%target_symbol = target_symbol_index
        plan%expr_index = expr_index
        plan%element_kind = context%symbols(target_symbol_index)%value_kind
        plan%rank = context%symbols(target_symbol_index)%array_rank
        if (plan%rank < 1 .or. plan%rank > ARRAY_EXPR_MAX_RANK) then
            error_msg = 'array expression requires a rank between 1 and 7'
            return
        end if
        if (context%symbols(target_symbol_index)%is_allocatable) then
            ! An allocatable carries a single flattened extent, so its plan is
            ! the rank-1 view of that extent.
            plan%rank = 1
            plan%extents(1) = &
                context%symbols(target_symbol_index)%allocatable_static_size
        else
            do d = 1, plan%rank
                plan%extents(d) = &
                    context%symbols(target_symbol_index)%array_dim_sizes(d)
            end do
        end if
        call validate_conformability(plan, error_msg)
    end subroutine build_array_expression

    subroutine validate_conformability(plan, error_msg)
        !! Reject a plan that cannot be iterated. An expression whose shape is
        !! not known to be positive fails here rather than emitting an
        !! iteration over an unknown extent.
        type(array_expr_plan_t), intent(in) :: plan
        character(len=:), allocatable, intent(out) :: error_msg

        if (plan%element_count() <= 0) then
            error_msg = 'array expression requires a positive element count'
            return
        end if
        call set_empty(error_msg)
    end subroutine validate_conformability

    subroutine emit_array_expression_assignment(arena, plan, context, error_msg)
        !! Iterate the plan's elements, produce every right-hand-side element
        !! value, and only then store them into the target.
        !!
        !! Producing all values before the first store is what makes the
        !! assignment behave as if the right-hand side were fully computed
        !! first (Fortran 2018 clause 10.2.1.3), so a right-hand side that
        !! overlaps the target -- `a = a(n:1:-1)` -- reads pre-assignment
        !! values regardless of traversal order.
        !!
        !! The linear index runs over the plan's extents in column-major
        !! order, so dimension 1 varies fastest.
        type(ast_arena_t), intent(in) :: arena
        type(array_expr_plan_t), intent(in) :: plan
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: values(:)
        integer :: i, n

        context%array_expression_cache_count = 0
        n = plan%element_count()
        if (n <= 0) then
            error_msg = 'array expression requires a positive element count'
            return
        end if
        call materialize_array_expression_calls(arena, plan%expr_index, context, &
                                                error_msg)
        if (len_trim(error_msg) > 0) return
        allocate (values(n))
        do i = 1, n
            call lower_array_elementwise_value(arena, plan%expr_index, &
                                               plan%target_symbol, i - 1, &
                                               context, values(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        do i = 1, n
            call store_array_linear_element(context, plan%target_symbol, &
                                            int(i - 1, c_int64_t), values(i), &
                                            error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine emit_array_expression_assignment

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
        type(lr_operand_desc_t) :: logical_xor
        type(lr_operand_desc_t) :: logical_cond

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

        symbol_index = cached_array_expression_symbol(context, node_index)
        if (symbol_index > 0) then
            call ensure_array_shapes_match(context, target_symbol_index, &
                                           symbol_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, symbol_index, &
                                           int(linear_index, c_int64_t), value, &
                                           error_msg)
            return
        end if

        if (is_elemental_minmax_call(arena, node_index)) then
            call lower_elementwise_minmax_call(arena, node_index, &
                                               target_symbol_index, linear_index, &
                                               context, value, error_msg)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol_compat(context, id_name)
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

        select type (logical_call => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(logical_call%name) .and. &
                is_contained_logical_function(context, logical_call%name)) then
                ! A scalar logical function broadcasts across the array. It is
                ! deliberately separate from the elemental user-call path,
                ! which handles numeric function results.
                call lower_elementwise_scalar_operand(arena, node_index, &
                                                      target_symbol_index, context, &
                                                      value, error_msg)
                return
            end if
        end select

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
            if (trim(adjustl(lowercase_text(bin_op))) == '.not.' .or. &
                trim(adjustl(lowercase_text(bin_op))) == 'not') then
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
            if (context%symbols(target_symbol_index)%value_kind == &
                    VALUE_LOGICAL .and. is_logical_connective(bin_op)) then
                call lower_array_elementwise_value(arena, bin_left, &
                                                   target_symbol_index, linear_index, &
                                                   context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_array_elementwise_value(arena, bin_right, &
                                                   target_symbol_index, linear_index, &
                                                   context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                select case (trim(adjustl(lowercase_text(bin_op))))
                case ('.and.', 'and')
                    if (.not. emit_i32_binary(context%session, LR_OP_AND, lhs, &
                                              rhs, value, error_msg)) return
                case ('.or.', 'or')
                    if (.not. emit_i32_binary(context%session, LR_OP_OR, lhs, &
                                              rhs, value, error_msg)) return
                case ('.neqv.', 'neqv', '.xor.', 'xor')
                    if (.not. emit_i32_binary(context%session, LR_OP_XOR, lhs, &
                                              rhs, value, error_msg)) return
                case ('.eqv.', 'eqv')
                    if (.not. emit_i32_binary(context%session, LR_OP_XOR, lhs, &
                                              rhs, logical_xor, error_msg)) return
                    if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, &
                            logical_xor, i32_immediate(context%session, &
                                                       0_c_int64_t), logical_cond, &
                            error_msg)) return
                    if (.not. emit_liric_i1_to_i32(context%session, logical_cond, &
                                                  value, error_msg)) return
                case default
                    error_msg = 'unsupported logical connective in whole-array expression: '// &
                                trim(bin_op)
                end select
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
        if (whole_array_expr_shape_symbol(arena, node_index, context) <= 0 .and. &
            whole_array_expr_constructor_extent(arena, node_index) <= 0) then
            call lower_i1_condition(arena, node_index, context, cond, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i1_to_i32(context%session, cond, value, &
                                           error_msg)) return
            return
        end if
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
        if (context%symbols(symbol_index)%has_runtime_array_stride) then
            call lower_runtime_strided_array_element_address(context, symbol_index, &
                i32_immediate(context%session, linear_index), addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_value_load(context, vk, addr, value, error_msg)
            return
        end if
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
        type(array_expr_plan_t) :: expression_plan
        type(lr_operand_desc_t) :: value
        type(lr_operand_desc_t) :: addr
        type(lr_operand_desc_t), allocatable :: rhs_values(:)
        integer :: k, k2, n, expression_symbol, expression_count
        integer(c_int64_t) :: rhs_count
        logical :: rhs_known
        integer(c_int64_t) :: tgt_lin, src_lin

        if (node_exists(arena, target%array_index)) then
            select type (component => arena%entries(target%array_index)%node)
            type is (component_access_node)
                if (component_access_is_alloc_array(arena, component, context)) then
                    call lower_alloc_array_component_section_assignment(arena, node, &
                        target, context, error_msg)
                    return
                end if
            class default
            end select
        end if

        call describe_array_section(arena, target, context, tinfo, error_msg)
        if (len_trim(error_msg) > 0) return

        if (tinfo%has_runtime_bounds) then
            if (tinfo%result_rank /= 1) then
                error_msg = 'runtime array section assignment requires one kept dimension'
                return
            end if

            ! A scalar RHS has no extent to report.  Evaluate it once before
            ! entering the loop, then broadcast that value through the live
            ! section coordinates.
            if (.not. actual_is_array_expression(arena, node%value_index, context)) then
                select case (context%symbols(tinfo%source_index)%value_kind)
                case (VALUE_F32)
                    call lower_f32_operand(arena, node%value_index, context, value, &
                                           error_msg)
                case (VALUE_F64)
                    call lower_f64_operand(arena, node%value_index, context, value, &
                                           error_msg)
                case (VALUE_I64)
                    call lower_i64_expression(arena, node%value_index, context, value, &
                                              error_msg)
                case (VALUE_I8)
                    call lower_i8_expression(arena, node%value_index, context, value, &
                                             error_msg)
                case (VALUE_I16)
                    call lower_i16_expression(arena, node%value_index, context, value, &
                                             error_msg)
                case default
                    call lower_i32_expression(arena, node%value_index, context, value, &
                                              error_msg)
                end select
                if (len_trim(error_msg) > 0) return
                call lower_runtime_array_section_scalar_broadcast(arena, target, &
                    tinfo, context, value, error_msg)
                return
            end if

            call array_section_assignment_rhs_extent(arena, node%value_index, context, &
                                                     rhs_count, rhs_known, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. rhs_known .or. rhs_count <= 0_c_int64_t) then
                error_msg = 'runtime array section assignment requires a known '// &
                            'positive RHS extent'
                return
            end if
            if (.not. actual_is_array_expression(arena, node%value_index, context)) then
                error_msg = 'runtime array section assignment requires an array RHS'
                return
            end if
            n = int(rhs_count)
            call create_array_expression_temp(context, &
                context%symbols(tinfo%source_index)%value_kind, n, 0, &
                expression_symbol, error_msg)
            if (len_trim(error_msg) > 0) return
            call build_array_expression(context, expression_symbol, node%value_index, &
                                        expression_plan, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_expression_assignment(arena, expression_plan, context, &
                                                  error_msg)
            if (len_trim(error_msg) > 0) return
            allocate (rhs_values(n))
            do k = 0, n - 1
                call load_array_linear_element(context, expression_symbol, &
                    int(k, c_int64_t), rhs_values(k + 1), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            do k = 0, n - 1
                call lower_runtime_array_section_element_address(arena, target, &
                    int(k, c_int64_t), context, addr, error_msg)
                if (len_trim(error_msg) > 0) return
                call emit_array_value_store(context, &
                    context%symbols(tinfo%source_index)%value_kind, &
                    rhs_values(k + 1), addr, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call set_empty(error_msg)
            return
        end if
        n = int(tinfo%section_extents(tinfo%kept_dims(1)))

        ! An array-valued expression has no single scalar value to broadcast.
        ! Materialize it into a rank-1 temporary before touching the target so
        ! mixed-rank expressions such as u(0,:) + A0(:) * B(k,:) retain both
        ! element-wise semantics and Fortran's fully-evaluated RHS rule.
        if (tinfo%result_rank == 1 .and. actual_is_array_expression( &
                arena, node%value_index, context)) then
            expression_count = n
            if (expression_count <= 0) then
                error_msg = 'array section expression requires a positive extent'
                return
            end if
            call create_array_expression_temp(context, &
                context%symbols(tinfo%source_index)%value_kind, expression_count, &
                0, expression_symbol, error_msg)
            if (len_trim(error_msg) > 0) return
            call build_array_expression(context, expression_symbol, node%value_index, &
                                        expression_plan, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_expression_assignment(arena, expression_plan, context, &
                                                  error_msg)
            if (len_trim(error_msg) > 0) return
            allocate (rhs_values(expression_count))
            do k = 0, expression_count - 1
                call load_array_linear_element(context, expression_symbol, &
                                               int(k, c_int64_t), rhs_values(k + 1), &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call store_section_elements(context, tinfo, rhs_values, error_msg)
            return
        end if

        ! Constructor and section-copy right-hand sides are handled for rank-1
        ! sections only; the scalar-broadcast tail below also covers rank-2.
        if (node_exists(arena, node%value_index) .and. tinfo%result_rank == 1 &
            .and. tinfo%source_rank >= 1 .and. tinfo%source_rank <= 2) then
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
                allocate (rhs_values(n))
                do k = 0, n - 1
                    call lower_i32_expression(arena, rhs%element_indices(k + 1), &
                                              context, rhs_values(k + 1), error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
                call store_section_elements(context, tinfo, rhs_values, error_msg)
                return
            type is (array_slice_node)
                call describe_array_section(arena, rhs, context, sinfo, error_msg)
                if (len_trim(error_msg) > 0) return
                if (sinfo%result_rank /= 1 .or. &
                    int(sinfo%section_extents(sinfo%kept_dims(1))) /= n) then
                    error_msg = 'array section assignment requires matching '// &
                                'rank-1 shapes'
                    return
                end if
                allocate (rhs_values(n))
                do k = 0, n - 1
                    src_lin = array_section_source_linear_index(sinfo, &
                                                                  int(k, c_int64_t))
                    call load_array_linear_element(context, sinfo%source_index, &
                                                   src_lin, rhs_values(k + 1), &
                                                   error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
                call store_section_elements(context, tinfo, rhs_values, error_msg)
                return
            end select
        end if

        ! Scalar broadcast: arr(section) = expr, for rank-1 and rank-2 sections.
        if (tinfo%result_rank < 1 .or. tinfo%result_rank > 2 .or. &
            tinfo%source_rank < tinfo%result_rank) then
            error_msg = 'array section assignment supports rank-1 and rank-2 '// &
                        'result sections'
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
                tgt_lin = array_section_source_linear_index(tinfo, &
                                                             int(k, c_int64_t))
                call store_array_linear_element(context, tinfo%source_index, &
                                                tgt_lin, value, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else
            do k2 = 0, int(tinfo%section_extents(2)) - 1
                do k = 0, n - 1
                    tgt_lin = array_section_source_linear_index(tinfo, &
                        int(k, c_int64_t) + int(tinfo%section_extents( &
                            tinfo%kept_dims(1)), c_int64_t) * int(k2, c_int64_t))
                    call store_array_linear_element(context, tinfo%source_index, &
                                                    tgt_lin, value, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end do
        end if
        call set_empty(error_msg)
    end subroutine lower_array_section_assignment

    logical function is_call_or_subscript_array_section(arena, target, context) &
            result(is_section)
        !! Some FortFront paths retain a section as a call_or_subscript node.
        !! Its range metadata is authoritative in arg_indices; scalar subscripts
        !! use the same node type and must continue through the scalar path.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(in) :: context
        integer :: i, symbol_index

        is_section = .false.
        if (.not. allocated(target%name)) return
        if (.not. allocated(target%arg_indices)) return
        symbol_index = find_symbol_compat(context, target%name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_array) return
        do i = 1, size(target%arg_indices)
            if (.not. node_exists(arena, target%arg_indices(i))) return
            select type (subscript => arena%entries(target%arg_indices(i))%node)
            type is (range_expression_node)
                is_section = .true.
                return
            type is (array_bounds_node)
                is_section = .true.
                return
            end select
        end do
    end function is_call_or_subscript_array_section

    subroutine lower_call_or_subscript_section_assignment(arena, node, target, &
                                                          context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_slice_node) :: slice
        integer :: base_index, i

        base_index = array_section_base_identifier(arena, target, context)
        if (base_index <= 0) then
            error_msg = 'array section target has no resolvable array base'
            return
        end if
        slice%array_index = base_index
        slice%num_dimensions = min(size(target%arg_indices), &
                                   size(slice%bounds_indices))
        slice%bounds_indices = 0
        do i = 1, slice%num_dimensions
            slice%bounds_indices(i) = target%arg_indices(i)
        end do
        slice%line = target%line
        slice%column = target%column
        call lower_array_section_assignment(arena, node, slice, context, error_msg)
    end subroutine lower_call_or_subscript_section_assignment

    integer function array_section_base_identifier(arena, target, context) &
            result(base_index)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: candidate_name, name_error
        integer :: i, target_symbol

        base_index = 0
        target_symbol = find_symbol_compat(context, target%name)
        if (target_symbol <= 0) return
        do i = arena%size, 1, -1
            if (.not. node_exists(arena, i)) cycle
            select type (candidate => arena%entries(i)%node)
            type is (identifier_node)
                call get_identifier_name(arena, i, candidate_name, name_error)
                if (len_trim(name_error) > 0) cycle
                if (.not. same_name(candidate_name, target%name)) cycle
                if (resolve_symbol_at_node(context, i, candidate_name) == &
                    target_symbol) then
                    base_index = i
                    return
                end if
            end select
        end do
    end function array_section_base_identifier

    subroutine lower_runtime_array_section_scalar_broadcast(arena, target, tinfo, &
                                                            context, value, error_msg)
        !! Broadcast one already-evaluated scalar through a runtime-bounded
        !! rank-1 section. The loop count and section coordinates are live
        !! LIRIC operands because an automatic array can use a dummy bound.
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: target
        type(array_section_info_t), intent(in) :: tinfo
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: extent, entry_idx, header_idx, backedge_idx
        type(lr_operand_desc_t) :: condition, next_idx, address
        integer(c_int32_t) :: entry_block, header_block, body_block
        integer(c_int32_t) :: latch_block, exit_block, idx_vreg

        call lower_runtime_array_section_extent(arena, target, tinfo, context, &
                                                extent, error_msg)
        if (len_trim(error_msg) > 0) return

        entry_idx = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, header_block, error_msg)) return
        context%current_block_id = header_block
        context%current_block_terminated = .false.
        idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg == 0_c_int32_t) idx_vreg = reserve_i32_vreg(context%session)
        if (idx_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve a positive section index vreg'
            return
        end if
        backedge_idx = i32_vreg(context%session, idx_vreg)
        if (.not. emit_liric_i32_phi(context%session, entry_idx, entry_block, &
                backedge_idx, latch_block, header_idx, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_idx, &
                extent, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                exit_block, error_msg)) return

        if (.not. set_liric_block(context%session, body_block, error_msg)) return
        context%current_block_id = body_block
        context%current_block_terminated = .false.
        call lower_runtime_array_section_element_address_operand(arena, target, header_idx, &
            context, address, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_array_value_store(context, context%symbols(tinfo%source_index)%value_kind, &
                                    value, address, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return

        if (.not. set_liric_block(context%session, latch_block, error_msg)) return
        context%current_block_id = latch_block
        context%current_block_terminated = .false.
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_idx, &
                i32_immediate(context%session, 1_c_int64_t), idx_vreg, next_idx, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return

        if (.not. set_liric_block(context%session, exit_block, error_msg)) return
        context%current_block_id = exit_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine lower_runtime_array_section_scalar_broadcast

    subroutine lower_runtime_array_section_extent(arena, target, tinfo, context, &
                                                   extent, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: target
        type(array_section_info_t), intent(in) :: tinfo
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: extent
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim, bounds_index
        type(lr_operand_desc_t) :: lower_op, upper_op, stride_op, upper_with_extent
        type(lr_operand_desc_t) :: difference, quotient, raw_extent, condition
        type(lr_operand_desc_t) :: zero_extent
        integer(c_int32_t) :: nonempty_block, empty_block, merge_block

        if (tinfo%result_rank /= 1) then
            error_msg = 'runtime scalar section broadcast requires one kept dimension'
            return
        end if
        dim = tinfo%kept_dims(1)
        bounds_index = 0
        if (dim <= target%num_dimensions) bounds_index = target%bounds_indices(dim)
        if (bounds_index <= 0) then
            if (context%symbols(tinfo%source_index)%has_runtime_dim_size(dim)) then
                extent = context%symbols(tinfo%source_index)%runtime_dim_size(dim)
            else
                extent = i32_immediate(context%session, int( &
                    context%symbols(tinfo%source_index)%array_dim_sizes(dim), c_int64_t))
            end if
            call set_empty(error_msg)
            return
        end if

        if (.not. node_exists(arena, bounds_index)) then
            error_msg = 'array section bound index does not reference an AST node'
            return
        end if
        select type (bounds_node => arena%entries(bounds_index)%node)
        type is (range_expression_node)
            if (bounds_node%start_index > 0) then
                call lower_i32_expression(arena, bounds_node%start_index, context, &
                                          lower_op, error_msg)
            else
                lower_op = i32_immediate(context%session, int( &
                    context%symbols(tinfo%source_index)%array_dim_lowers(dim), c_int64_t))
            end if
            if (len_trim(error_msg) > 0) return
            if (bounds_node%end_index > 0) then
                call lower_i32_expression(arena, bounds_node%end_index, context, &
                                          upper_op, error_msg)
            else if (context%symbols(tinfo%source_index)%has_runtime_dim_size(dim)) then
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, &
                        i32_immediate(context%session, int( &
                            context%symbols(tinfo%source_index)%array_dim_lowers(dim), &
                            c_int64_t)), &
                        context%symbols(tinfo%source_index)%runtime_dim_size(dim), &
                        upper_with_extent, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_with_extent, &
                        i32_immediate(context%session, 1_c_int64_t), upper_op, &
                        error_msg)) return
            else
                upper_op = i32_immediate(context%session, int( &
                    context%symbols(tinfo%source_index)%array_dim_lowers(dim) + &
                    context%symbols(tinfo%source_index)%array_dim_sizes(dim) - 1, &
                    c_int64_t))
            end if
            if (len_trim(error_msg) > 0) return
            if (bounds_node%stride_index > 0) then
                call lower_i32_expression(arena, bounds_node%stride_index, context, &
                                          stride_op, error_msg)
            else
                stride_op = i32_immediate(context%session, 1_c_int64_t)
            end if
        type is (array_bounds_node)
            if (bounds_node%lower_bound_index > 0) then
                call lower_i32_expression(arena, bounds_node%lower_bound_index, context, &
                                          lower_op, error_msg)
            else
                lower_op = i32_immediate(context%session, int( &
                    context%symbols(tinfo%source_index)%array_dim_lowers(dim), c_int64_t))
            end if
            if (len_trim(error_msg) > 0) return
            if (bounds_node%upper_bound_index > 0) then
                call lower_i32_expression(arena, bounds_node%upper_bound_index, context, &
                                          upper_op, error_msg)
            else if (context%symbols(tinfo%source_index)%has_runtime_dim_size(dim)) then
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, &
                        i32_immediate(context%session, int( &
                            context%symbols(tinfo%source_index)%array_dim_lowers(dim), &
                            c_int64_t)), &
                        context%symbols(tinfo%source_index)%runtime_dim_size(dim), &
                        upper_with_extent, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_with_extent, &
                        i32_immediate(context%session, 1_c_int64_t), upper_op, &
                        error_msg)) return
            else
                upper_op = i32_immediate(context%session, int( &
                    context%symbols(tinfo%source_index)%array_dim_lowers(dim) + &
                    context%symbols(tinfo%source_index)%array_dim_sizes(dim) - 1, &
                    c_int64_t))
            end if
            if (len_trim(error_msg) > 0) return
            if (bounds_node%stride_index > 0) then
                call lower_i32_expression(arena, bounds_node%stride_index, context, &
                                          stride_op, error_msg)
            else
                stride_op = i32_immediate(context%session, 1_c_int64_t)
            end if
        class default
            error_msg = 'runtime scalar section broadcast requires a range bound'
            return
        end select
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, upper_op, lower_op, &
                difference, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_SDIV, difference, stride_op, &
                quotient, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, quotient, &
                i32_immediate(context%session, 1_c_int64_t), raw_extent, &
                error_msg)) return

        ! A runtime section can be empty when the bound direction disagrees
        ! with the stride.  Clamp the trip count before the loop so the LIR
        ! extent itself remains nonnegative.
        nonempty_block = create_liric_block(context%session)
        empty_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGT, raw_extent, &
                i32_immediate(context%session, 0_c_int64_t), condition, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, nonempty_block, &
                empty_block, error_msg)) return

        if (.not. set_liric_block(context%session, nonempty_block, error_msg)) return
        context%current_block_id = nonempty_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, empty_block, error_msg)) return
        context%current_block_id = empty_block
        context%current_block_terminated = .false.
        zero_extent = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_i32_phi(context%session, raw_extent, nonempty_block, &
                zero_extent, empty_block, extent, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_runtime_array_section_extent

    subroutine store_section_elements(context, tinfo, values, error_msg)
        !! Store an already-evaluated rank-1 right-hand side into a rank-1
        !! section. Every value in `values` was produced before the first store,
        !! so the assignment behaves as if the right-hand side were fully
        !! computed first (Fortran 2018 clause 10.2.1.3) even when target and
        !! source overlap in storage.
        type(lowering_context_t), intent(inout) :: context
        type(array_section_info_t), intent(in) :: tinfo
        type(lr_operand_desc_t), intent(in) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k
        integer(c_int64_t) :: tgt_lin

        do k = 0, size(values) - 1
            tgt_lin = array_section_source_linear_index(tinfo, int(k, c_int64_t))
            call store_array_linear_element(context, tinfo%source_index, &
                                            tgt_lin, values(k + 1), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine store_section_elements

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
        if (context%symbols(symbol_index)%has_runtime_array_stride) then
            call lower_runtime_strided_array_element_address(context, symbol_index, &
                i32_immediate(context%session, linear_index), addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_value_store(context, vk, value, addr, error_msg)
            return
        end if
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
        if (context%symbols(symbol_index)%is_allocatable .or. &
                context%symbols(symbol_index)%has_runtime_dim_size(1)) then
            call lower_runtime_descriptor_array_element_address(context, &
                symbol_index, index_value, addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_value_store(context, vk, value, addr, error_msg)
            return
        end if
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
        if (context%symbols(symbol_index)%is_allocatable .or. &
                context%symbols(symbol_index)%has_runtime_dim_size(1)) then
            call lower_runtime_descriptor_array_element_address(context, &
                symbol_index, index_value, addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_value_load(context, vk, addr, value, error_msg)
            return
        end if
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
            sym_idx = find_symbol_compat(context, target%name)
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

    subroutine lower_runtime_strided_array_element_address(context, symbol_index, &
            linear_index, element_address, error_msg)
        ! Address one element of a rank-1 view whose byte stride came from the
        ! canonical array descriptor. The linear index is zero based.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: index_i64, byte_offset

        if (.not. emit_liric_i32_to_i64(context%session, linear_index, index_i64, &
                                       error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, index_i64, &
                context%symbols(symbol_index)%runtime_array_stride_bytes, &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, &
                context%symbols(symbol_index)%element_address, byte_offset, &
                element_address, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_runtime_strided_array_element_address

    subroutine lower_runtime_descriptor_array_element_address(context, symbol_index, &
            linear_index, element_address, error_msg)
        ! Address one element of a runtime-shaped array. Allocatable arrays get
        ! their base from the allocatable descriptor; descriptor-bound dummies
        ! already expose their base through element_address. Rank-1 strided
        ! dummies retain their descriptor stride, while rank-2 paths in this
        ! direct-session slice use the contiguous column-major element stride.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: base, index_i64, byte_offset
        type(lr_operand_desc_t) :: descriptor
        integer :: value_kind

        value_kind = context%symbols(symbol_index)%value_kind
        if (context%symbols(symbol_index)%is_allocatable) then
            descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
            if (.not. emit_ptr_load(context%session, descriptor, base, error_msg)) &
                return
        else if (context%symbols(symbol_index)%has_runtime_array_stride) then
            call lower_runtime_strided_array_element_address(context, symbol_index, &
                linear_index, element_address, error_msg)
            return
        else
            base = context%symbols(symbol_index)%element_address
        end if
        if (.not. emit_liric_i32_to_i64(context%session, linear_index, index_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, index_i64, &
                i64_immediate(context%session, allocatable_elem_size(value_kind)), &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, base, byte_offset, &
                element_address, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_runtime_descriptor_array_element_address

    subroutine emit_array_value_load(context, value_kind, addr, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: addr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: ok

        select case (value_kind)
        case (VALUE_F32)
            ok = emit_liric_f32_load(context%session, addr, value, error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_load(context%session, addr, value, error_msg)
        case (VALUE_I64)
            ok = emit_i64_load(context%session, addr, value, error_msg)
        case (VALUE_I8)
            ok = emit_i8_load(context%session, addr, value, error_msg)
        case (VALUE_I16)
            ok = emit_i16_load(context%session, addr, value, error_msg)
        case default
            ok = emit_i32_load(context%session, addr, value, error_msg)
        end select
        if (.not. ok) return
        call set_empty(error_msg)
    end subroutine emit_array_value_load

    subroutine emit_array_value_store(context, value_kind, value, addr, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: ok

        select case (value_kind)
        case (VALUE_F32)
            ok = emit_liric_f32_store(context%session, value, addr, error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_store(context%session, value, addr, error_msg)
        case (VALUE_I64)
            ok = emit_i64_store(context%session, value, addr, error_msg)
        case (VALUE_I8)
            ok = emit_i8_store(context%session, value, addr, error_msg)
        case (VALUE_I16)
            ok = emit_i16_store(context%session, value, addr, error_msg)
        case default
            ok = emit_i32_store(context%session, value, addr, error_msg)
        end select
        if (.not. ok) return
        call set_empty(error_msg)
    end subroutine emit_array_value_store

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

        symbol_index = find_symbol_compat(context, node%name)
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
        if (context%symbols(symbol_index)%has_runtime_array_stride) then
            call lower_runtime_strided_array_element_address(context, symbol_index, &
                linear_index, element_address, error_msg)
            return
        end if

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

        if (context%symbols(symbol_index)%has_runtime_dim_size(1)) then
            call lower_runtime_element_linear_index(arena, node, context, &
                symbol_index, rank, linear_index, error_msg)
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

    subroutine lower_runtime_element_linear_index(arena, node, context, &
                                                  symbol_index, rank, linear_index, &
                                                  error_msg)
        !! Column-major linear element index of a subscripted array whose extents
        !! are only known at run time (an assumed-shape dummy bound through the
        !! hidden i64 extent ABI). Same formula as the compile-time path,
        !! sum_k (idx_k - lower_k) * prod_{j<k} extent_j, but each leading extent
        !! is the runtime_dim_size operand rather than a compile-time immediate.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: rank
        type(lr_operand_desc_t), intent(out) :: linear_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: k
        integer(c_int64_t) :: lower_k
        type(lr_operand_desc_t) :: idx_op, off_op, term_op, acc_op, next_acc
        type(lr_operand_desc_t) :: stride_op, next_stride, extent_k

        stride_op = i32_immediate(context%session, 1_c_int64_t)
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
                error_msg = 'rt_arr_index_sub: '//error_msg
                return
            end if
            if (k == 1) then
                term_op = off_op
                acc_op = off_op
            else
                if (.not. emit_i32_binary(context%session, LR_OP_MUL, off_op, &
                        stride_op, term_op, error_msg)) then
                    error_msg = 'rt_arr_index_mul: '//error_msg
                    return
                end if
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc_op, &
                        term_op, next_acc, error_msg)) then
                    error_msg = 'rt_arr_index_add: '//error_msg
                    return
                end if
                acc_op = next_acc
            end if
            if (k < rank) then
                if (context%symbols(symbol_index)%has_runtime_dim_size(k)) then
                    extent_k = context%symbols(symbol_index)%runtime_dim_size(k)
                else
                    extent_k = i32_immediate(context%session, &
                        int(context%symbols(symbol_index)%array_dim_sizes(k), &
                            c_int64_t))
                end if
                if (k == 1) then
                    stride_op = extent_k
                else
                    if (.not. emit_i32_binary(context%session, LR_OP_MUL, &
                            stride_op, extent_k, next_stride, error_msg)) then
                        error_msg = 'rt_arr_stride_mul: '//error_msg
                        return
                    end if
                    stride_op = next_stride
                end if
            end if
        end do
        linear_index = acc_op
        call set_empty(error_msg)
    end subroutine lower_runtime_element_linear_index

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
            symbol_index = find_symbol_compat(context, node%name)
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
        symbol_index = find_symbol_compat(context, node%name)
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
        if (context%symbols(symbol_index)%has_runtime_array_stride) then
            call lower_runtime_strided_array_element_address(context, symbol_index, &
                linear_index, element_address, error_msg)
            return
        end if

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
            symbol_index = find_symbol_compat(context, id_name)
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
    ! rank-2 source into the rank-1 target symbol. The source extents are known
    ! at compile time, so the reduction unrolls to a fixed sequence of loads
    ! and accumulations. A runtime-shaped target is also supported when its
    ! descriptor supplies the destination storage. Declines (handled=.false.,
    ! no error) for any form outside this scope so the caller keeps its
    ! existing behaviour.
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
        logical :: is_comparison, target_runtime
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
        case ('norm2'); reduce_op = DIM_REDUCE_NORM2
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

        ! The target must be a plain rank-1 array. Runtime-shaped targets are
        ! valid here because their descriptor provides the destination storage.
        if (context%symbols(symbol_index)%is_allocatable) return
        if (context%symbols(symbol_index)%array_rank /= 1) return
        target_runtime = context%symbols(symbol_index)%has_runtime_dim_size(1)
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
            if (reduce_op == DIM_REDUCE_NORM2) return
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
        if (.not. target_runtime .and. out_size /= &
                context%symbols(symbol_index)%array_size) return

        ! Numeric sum/product keep the source kind; mask reductions work in i32
        ! (count -> integer count, any/all -> logical 0/1).
        if (reduce_op == DIM_REDUCE_NORM2) then
            select case (source_vk)
            case (VALUE_F32, VALUE_F64)
            case default; return
            end select
            if (result_vk /= source_vk) return
        else if (reduce_op == DIM_REDUCE_SUM .or. reduce_op == DIM_REDUCE_PROD) then
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
            if (reduce_op == DIM_REDUCE_NORM2) then
                call dim_reduction_norm2_slice(context, src_sym, source_vk, &
                    dim_value, d1, i, reduce_extent, acc, error_msg)
                if (len_trim(error_msg) > 0) return
                if (target_runtime) then
                    call store_array_element_at_operand(context, symbol_index, &
                        i32_immediate(context%session, int(i, c_int64_t)), acc, &
                        error_msg)
                else
                    call store_array_linear_element(context, symbol_index, &
                                                    int(i, c_int64_t), acc, &
                                                    error_msg)
                end if
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            call reduction_identity(context, reduce_op, source_vk, acc, error_msg)
            if (len_trim(error_msg) > 0) return
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
                    call reduction_combine(context, reduce_op, source_vk, acc, &
                                           elem, error_msg)
                    if (len_trim(error_msg) > 0) return
                else
                    call dim_reduction_mask_element(arena, context, is_comparison, &
                        cmp_lsym, cmp_lscalar, cmp_rsym, cmp_rscalar, src_sym, &
                        compare_vk, predicate, lin, cond, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call reduction_combine(context, reduce_op, source_vk, acc, &
                                           cond, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end do
            if (target_runtime) then
                call store_array_element_at_operand(context, symbol_index, &
                    i32_immediate(context%session, int(i, c_int64_t)), acc, &
                    error_msg)
            else
                call store_array_linear_element(context, symbol_index, &
                                                int(i, c_int64_t), acc, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do
        handled = .true.
        call set_empty(error_msg)
    end subroutine lower_dim_reduction_assignment

    ! One norm2 slice of a dim-wise reduction: the reduce_extent elements of
    ! the rank-2 source that contribute to output element i, reduced with the
    ! scaled sum-of-squares algorithm.
    subroutine dim_reduction_norm2_slice(context, src_sym, source_vk, dim_value, &
                                         d1, i, reduce_extent, acc, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: src_sym
        integer, intent(in) :: source_vk
        integer(c_int64_t), intent(in) :: dim_value
        integer, intent(in) :: d1
        integer, intent(in) :: i
        integer, intent(in) :: reduce_extent
        type(lr_operand_desc_t), intent(out) :: acc
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: elems(:)
        integer(c_int64_t) :: lin
        integer :: j

        call set_empty(error_msg)
        allocate (elems(reduce_extent))
        do j = 0, reduce_extent - 1
            if (dim_value == 1_c_int64_t) then
                lin = int(j, c_int64_t) + int(i, c_int64_t) * int(d1, c_int64_t)
            else
                lin = int(i, c_int64_t) + int(j, c_int64_t) * int(d1, c_int64_t)
            end if
            call load_array_linear_element(context, src_sym, lin, elems(j + 1), &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call emit_norm2_from_elements(context, source_vk, elems, acc, error_msg)
    end subroutine dim_reduction_norm2_slice

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
        s = find_symbol_compat(context, name)
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
