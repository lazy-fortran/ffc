    ! Assumed-shape dummy ABI (#334). A rank-1 through rank-4 assumed-shape dummy
    ! `a(:)` / `a(:,:)` / `a(:,:,:)` / `a(:,:,:,:)` is bound through exactly one canonical array descriptor
    ! (`ffc_array_descriptor`, `docs/ARRAY_DESCRIPTOR_ABI.md`). The caller
    ! materializes a descriptor for the actual and passes its address in the
    ! dummy's single visible parameter slot. No hidden extent arguments are
    ! appended, and the callee reads bounds, extents, and the base address from
    ! that descriptor alone.
    !
    ! The descriptor describes the DUMMY's view of the actual, which is what
    ! F2018 15.5.2.4 associates: `base` is the address of the actual's first
    ! element; a bare-colon dummy has lower bound 1, while an explicit lower
    ! bound such as 0: is retained. Extents and byte strides are the actual's,
    ! so column-major layout and element identity are preserved exactly.

    integer(c_int64_t) function assumed_shape_dim_field_offset(dim, field) &
            result(offset)
        ! Byte offset of one per-dimension field inside the descriptor.
        integer, intent(in) :: dim
        integer(c_int64_t), intent(in) :: field

        offset = int(ARRAY_DESCRIPTOR_DIM_OFFSET, c_int64_t) &
                 + int(ARRAY_DIMENSION_BYTES, c_int64_t)*int(dim - 1, c_int64_t) &
                 + field
    end function assumed_shape_dim_field_offset

    integer(c_int32_t) function assumed_shape_element_type(value_kind) &
            result(element_type)
        ! Canonical descriptor element type code for a lowering value kind.
        integer, intent(in) :: value_kind

        select case (value_kind)
        case (VALUE_F32, VALUE_F64)
            element_type = ARRAY_ELEMENT_REAL
        case (VALUE_LOGICAL)
            element_type = ARRAY_ELEMENT_LOGICAL
        case (VALUE_C4, VALUE_C8)
            element_type = ARRAY_ELEMENT_COMPLEX
        case (VALUE_CHARACTER)
            element_type = ARRAY_ELEMENT_CHARACTER
        case (VALUE_DERIVED)
            element_type = ARRAY_ELEMENT_DERIVED
        case default
            element_type = ARRAY_ELEMENT_INTEGER
        end select
    end function assumed_shape_element_type

    integer function assumed_shape_actual_symbol(arena, node_index, context) &
            result(symbol_index)
        ! Symbol holding the storage an assumed-shape actual designates: the
        ! identifier itself, or the source array of a contiguous section.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable :: id_name, err
        type(array_section_info_t) :: info
        integer :: type_index, component_index

        symbol_index = 0
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, err)
            if (len_trim(err) > 0) return
            symbol_index = resolve_symbol_at_node(context, node_index, id_name)
            return
        end if
        select type (slice => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, slice, context, info, err, &
                                        allow_derived=.true.)
            if (len_trim(err) > 0) return
            symbol_index = info%source_index
        type is (component_access_node)
            call resolve_derived_array_component(arena, node_index, context, &
                                                 symbol_index, type_index, &
                                                 component_index, err)
        end select
    end function assumed_shape_actual_symbol

    logical function assumed_shape_descriptor_forward_actual(arena, node_index, &
                                                             context, source_symbol) &
            result(is_forward)
        ! An assumed-shape dummy or its identity section is already a
        ! descriptor-backed view. Forwarding it composes only the dummy bound;
        ! base, extent, and signed byte stride remain those of the incoming
        ! borrowed descriptor.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: source_symbol
        integer :: bounds_index

        is_forward = .false.
        source_symbol = 0
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            source_symbol = assumed_shape_actual_symbol(arena, node_index, context)
        else
            select type (slice => arena%entries(node_index)%node)
            type is (array_slice_node)
                if (slice%num_dimensions /= 1) return
                source_symbol = assumed_shape_actual_symbol(arena, &
                    slice%array_index, context)
                if (source_symbol <= 0) return
                bounds_index = slice%bounds_indices(1)
                if (.not. node_exists(arena, bounds_index)) return
                select type (bounds => arena%entries(bounds_index)%node)
                type is (array_bounds_node)
                    if (.not. bounds%is_assumed_shape .and. &
                        .not. bounds%is_deferred_shape) return
                type is (range_expression_node)
                    if (bounds%start_index > 0) return
                    if (bounds%end_index > 0) return
                    if (bounds%stride_index > 0) return
                class default
                    return
                end select
            class default
                return
            end select
        end if
        if (source_symbol <= 0) return
        if (.not. context%symbols(source_symbol)%is_assumed_shape_dummy) return
        if (context%symbols(source_symbol)%array_rank < 1 .or. &
            context%symbols(source_symbol)%array_rank > 4) return
        if (.not. context%symbols(source_symbol)%has_runtime_descriptor) return
        is_forward = .true.
    end function assumed_shape_descriptor_forward_actual

    logical function assumed_shape_empty_constructor(arena, node_index) &
            result(is_empty)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_empty = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (actual => arena%entries(node_index)%node)
        type is (array_literal_node)
            if (allocated(actual%element_indices)) then
                is_empty = size(actual%element_indices) == 0
            end if
        class default
        end select
    end function assumed_shape_empty_constructor

    integer function assumed_shape_actual_rank(arena, node_index, context) &
            result(rank)
        ! Rank the actual presents to the dummy: a whole array's declared rank,
        ! or 1 for a contiguous rank-1 section.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer :: symbol_index, type_index, component_index
        character(len=:), allocatable :: id_name, err

        rank = 0
        if (assumed_shape_empty_constructor(arena, node_index)) then
            rank = 1
            return
        end if
        if (actual_is_rank1_section_view(arena, node_index, context) .or. &
            actual_is_contiguous_section(arena, node_index, context, &
                                         allow_derived=.true.)) then
            rank = 1
            return
        end if
        select type (component => arena%entries(node_index)%node)
        type is (component_access_node)
            call resolve_derived_array_component(arena, node_index, context, &
                                                 symbol_index, type_index, &
                                                 component_index, err)
            if (len_trim(err) == 0) then
                rank = context%symbols(symbol_index)%array_rank
            else
                rank = 0
            end if
            return
        class default
        end select
        if (.not. is_identifier(arena, node_index)) return
        symbol_index = assumed_shape_actual_symbol(arena, node_index, context)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_array) then
            if (.not. context%symbols(symbol_index)%is_allocatable) return
        end if
        rank = context%symbols(symbol_index)%array_rank
        if (rank >= 1) return
        ! An allocatable array symbol carries its rank only in its declaration
        ! (its shape is not fixed at declaration time), so fall back to the
        ! declared dimension count.
        call get_identifier_name(arena, node_index, id_name, err)
        if (len_trim(err) > 0) return
        rank = declared_dimension_count(context, id_name)
    end function assumed_shape_actual_rank

    integer function declared_dimension_count(context, var_name) result(rank)
        ! Declared rank of var_name: the number of dimensions on its array
        ! declaration, colon-shaped or not.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: var_name
        integer :: n
        type(declaration_node), pointer :: decl

        rank = 0
        do n = 1, context%arena%size
            if (.not. node_exists(context%arena, n)) cycle
            decl => declaration_named(context%arena, n, var_name)
            if (.not. associated(decl)) cycle
            if (.not. decl%is_array) cycle
            if (.not. allocated(decl%dimension_indices)) cycle
            rank = size(decl%dimension_indices)
            return
        end do
    end function declared_dimension_count

    subroutine assumed_shape_actual_base(arena, node_index, context, base, &
                                         error_msg)
        ! Address of the actual's first element, whatever storage class it has.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: base
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index, type_index, component_index

        if (assumed_shape_empty_constructor(arena, node_index)) then
            base = null_ptr_operand(context)
            call set_empty(error_msg)
            return
        end if

        if (component_access_is_derived_array(arena, node_index, context)) then
            call resolve_derived_array_component(arena, node_index, context, &
                                                 symbol_index, type_index, &
                                                 component_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call derived_array_component_address_at(context, symbol_index, &
                type_index, component_index, &
                i32_immediate(context%session, 0_c_int64_t), base, error_msg)
        else if (actual_is_allocatable_array(arena, node_index, context)) then
            call allocatable_array_actual_data_address(arena, node_index, &
                                                       context, base, error_msg)
        else if (actual_is_rank1_section_view(arena, node_index, context)) then
            call rank1_section_actual_base(arena, node_index, context, base, &
                                           error_msg)
        else if (actual_is_contiguous_section(arena, node_index, context, &
                                              allow_derived=.true.)) then
            call section_actual_address(arena, node_index, context, base, &
                                        error_msg)
        else if (actual_is_whole_array(arena, node_index, context)) then
            call whole_array_actual_address(arena, node_index, context, base, &
                                            error_msg)
        else
            error_msg = 'assumed-shape actual must be a whole array, an '// &
                        'allocatable array, or a contiguous array section'
        end if
    end subroutine assumed_shape_actual_base

    subroutine rank1_section_actual_base(arena, node_index, context, base, &
                                         error_msg)
        ! Return the first selected element of a rank-1 section.  Static views
        ! use the existing linear-address helper; runtime bounds use the same
        ! live coordinate lowering as section assignment.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: base
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info

        select type (slice => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, slice, context, info, error_msg, &
                                        allow_derived=.true.)
            if (len_trim(error_msg) > 0) return
            if (info%has_runtime_bounds) then
                call lower_runtime_array_section_element_address_operand(arena, slice, &
                    i32_immediate(context%session, 0_c_int64_t), context, base, &
                    error_msg)
            else
                call section_actual_address(arena, node_index, context, base, &
                                            error_msg)
            end if
        class default
            error_msg = 'assumed-shape actual expected an array section'
        end select
    end subroutine rank1_section_actual_base

    subroutine assumed_shape_actual_extent(arena, node_index, context, rank, &
                                           dim, extent, error_msg)
        ! Extent of the actual's dimension `dim` as an i64 operand. An
        ! allocatable actual reads its live descriptor bounds; every other
        ! supported actual folds to its declared compile-time extent.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: rank
        integer, intent(in) :: dim
        type(lr_operand_desc_t), intent(out) :: extent
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: extents(ARRAY_MAX_RANK)
        character(len=:), allocatable :: id_name
        logical :: ok
        integer :: symbol_index, type_index, component_index
        type(array_section_info_t) :: section_info
        type(lr_operand_desc_t) :: extent_i32

        if (assumed_shape_empty_constructor(arena, node_index)) then
            if (dim /= 1) then
                error_msg = 'empty array constructor assumed-shape actual supports '// &
                            'rank 1 only'
                return
            end if
            extent = i64_immediate(context%session, 0_c_int64_t)
            call set_empty(error_msg)
            return
        end if

        call set_empty(error_msg)
        if (component_access_is_derived_array(arena, node_index, context)) then
            call resolve_derived_array_component(arena, node_index, context, &
                                                 symbol_index, type_index, &
                                                 component_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (dim < 1 .or. dim > rank .or. dim > &
                context%symbols(symbol_index)%array_rank) then
                error_msg = 'derived array component assumed-shape actual has '// &
                            'no resolvable extent for this dimension'
                return
            end if
            if (context%symbols(symbol_index)%array_dim_sizes(dim) <= 0) then
                error_msg = 'derived array component assumed-shape actual has '// &
                            'no resolvable extent'
                return
            end if
            extent = i64_immediate(context%session, int( &
                context%symbols(symbol_index)%array_dim_sizes(dim), c_int64_t))
            return
        end if
        if (actual_is_allocatable_array(arena, node_index, context)) then
            call actual_extent_i64(arena, node_index, context, dim, extent, &
                                   error_msg)
            return
        end if

        extents = 0
        ok = .false.
        if (actual_is_rank1_section_view(arena, node_index, context)) then
            select type (slice => arena%entries(node_index)%node)
            type is (array_slice_node)
                call describe_array_section(arena, slice, context, section_info, &
                                            error_msg, allow_derived=.true.)
                if (len_trim(error_msg) > 0) return
                if (section_info%has_runtime_bounds) then
                    call lower_runtime_array_section_extent(arena, slice, &
                        section_info, context, extent_i32, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_liric_i32_to_i64(context%session, extent_i32, &
                                                   extent, error_msg)) return
                else
                    extent = i64_immediate(context%session, section_info%section_extents(1))
                end if
                return
            class default
            end select
        else if (actual_is_contiguous_section(arena, node_index, context, &
                                              allow_derived=.true.)) then
            call section_actual_decl_extents(context, node_index, 1, extents, ok)
        else if (is_identifier(arena, node_index)) then
            symbol_index = assumed_shape_actual_symbol(arena, node_index, context)
            if (symbol_index > 0) then
                if (context%symbols(symbol_index)%array_rank >= dim) then
                    if (context%symbols(symbol_index)%has_runtime_dim_size(dim)) then
                        if (.not. emit_liric_i32_to_i64(context%session, &
                                context%symbols(symbol_index)%runtime_dim_size(dim), &
                                extent, error_msg)) return
                        return
                    end if
                    if (context%symbols(symbol_index)%array_dim_sizes(dim) > 0) then
                        extent = i64_immediate(context%session, int( &
                            context%symbols(symbol_index)%array_dim_sizes(dim), &
                            c_int64_t))
                        return
                    end if
                end if
            end if
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            call array_decl_extents(context, id_name, rank, extents, ok)
        end if
        if (.not. ok) then
            error_msg = 'assumed-shape actual has no resolvable extent'
            return
        end if
        if (extents(dim) < 0) then
            error_msg = 'assumed-shape actual has no resolvable extent'
            return
        end if
        extent = i64_immediate(context%session, int(extents(dim), c_int64_t))
    end subroutine assumed_shape_actual_extent

    subroutine store_descriptor_i32(context, value, descriptor, offset, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: descriptor
        integer(c_int64_t), intent(in) :: offset
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: field

        if (.not. emit_ptr_offset(context%session, descriptor, offset, field, &
                                  error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, int(value, c_int64_t)), field, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine store_descriptor_i32

    subroutine build_assumed_shape_descriptor(arena, node_index, context, rank, &
                                              callee_name, param_pos, descriptor, &
                                              error_msg, assumed_rank)
        ! Materialize the canonical descriptor an assumed-shape dummy receives.
        ! One stack descriptor per call argument; it borrows the actual's
        ! storage and never claims ownership, so passing an array never changes
        ! the actual's allocation lifetime.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: rank
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in), optional :: assumed_rank
        type(lr_operand_desc_t) :: base, extent, stride, running
        type(lr_operand_desc_t) :: dynamic_type_addr
        type(lr_operand_desc_t) :: dynamic_type_id, dynamic_type_i32
        type(lr_operand_desc_t) :: section_stride
        integer :: symbol_index, value_kind, actual_rank, dim
        integer :: expression_symbol, expression_extent, expression_kind
        integer :: dummy_lowers(ARRAY_MAX_RANK)
        integer :: component_type, component_index
        integer(c_int64_t) :: element_bytes
        integer(c_int32_t) :: flags
        integer(c_int32_t) :: class_star_type_id
        logical :: is_component_projection, has_dummy_lowers, is_empty_actual
        logical :: is_class_star_dummy
        logical :: is_section_actual, section_is_contiguous
        logical :: is_materialized_expression, expression_extent_known
        logical :: is_forwarded_actual
        type(array_expr_plan_t) :: expression_plan
        character(len=32) :: rank_text, actual_rank_text
        character(len=:), allocatable :: dummy_type_name
        integer :: dummy_type_index
        integer :: forwarded_symbol

        is_materialized_expression = .false.
        expression_symbol = 0
        is_forwarded_actual = assumed_shape_descriptor_forward_actual(arena, &
            node_index, context, forwarded_symbol)

        ! Array-valued expressions have no storage for the descriptor to borrow.
        ! Materialize a fixed-size expression before rank and descriptor
        ! construction; this is the same temporary representation used by the
        ! explicit-shape array-argument path. Storage-backed sections retain
        ! their existing descriptor construction below.
        if (.not. actual_is_whole_array(arena, node_index, context) .and. &
            .not. is_forwarded_actual .and. &
            .not. actual_is_contiguous_section(arena, node_index, context, &
                                               allow_derived=.true.) .and. &
            .not. actual_is_rank1_section_view(arena, node_index, context) .and. &
            actual_is_array_expression(arena, node_index, context) .and. &
            .not. assumed_shape_empty_constructor(arena, node_index)) then
            call array_expression_static_extent(arena, node_index, context, &
                                                expression_extent, &
                                                expression_extent_known)
            if (.not. expression_extent_known .or. expression_extent <= 0) then
                error_msg = 'array-valued actual has no statically known extent in '// &
                    trim(callee_name)
                return
            end if
            expression_kind = expression_value_kind(arena, node_index, context, &
                                                    VALUE_F32)
            call create_array_expression_temp(context, expression_kind, &
                                              expression_extent, 0, &
                                              expression_symbol, error_msg)
            if (len_trim(error_msg) > 0) return
            call build_array_expression(context, expression_symbol, node_index, &
                                        expression_plan, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_array_expression_assignment(arena, expression_plan, context, &
                                                  error_msg)
            if (len_trim(error_msg) > 0) return
            is_materialized_expression = .true.
        end if

        if (assumed_shape_descriptor_forward_actual(arena, node_index, context, &
                                                    forwarded_symbol)) then
            call build_forwarded_assumed_shape_descriptor(arena, context, rank, &
                callee_name, param_pos, forwarded_symbol, descriptor, error_msg)
            return
        end if

        if (is_materialized_expression) then
            actual_rank = 1
        else
            actual_rank = assumed_shape_actual_rank(arena, node_index, context)
        end if
        if (actual_rank /= rank) then
            write (rank_text, '(I0)') rank
            write (actual_rank_text, '(I0)') actual_rank
            error_msg = 'Rank mismatch in argument to '//trim(callee_name)// &
                        ': assumed-shape dummy of rank '//trim(rank_text)// &
                        ' received an actual of rank '//trim(actual_rank_text)
            return
        end if

        if (present(assumed_rank)) then
            if (assumed_rank) then
                dummy_lowers(1:rank) = 1
                has_dummy_lowers = .true.
            else
                call callee_dummy_assumed_shape_lowers(arena, callee_name, param_pos, &
                                                       context, rank, dummy_lowers, &
                                                       has_dummy_lowers, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
        else
            call callee_dummy_assumed_shape_lowers(arena, callee_name, param_pos, &
                                                   context, rank, dummy_lowers, &
                                                   has_dummy_lowers, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (.not. has_dummy_lowers) dummy_lowers(1:rank) = 1

        is_class_star_dummy = callee_dummy_is_class_star(arena, callee_name, &
                                                         param_pos)
        class_star_type_id = 0_c_int32_t

        is_empty_actual = assumed_shape_empty_constructor(arena, node_index)
        if (is_materialized_expression) then
            symbol_index = expression_symbol
        else
            symbol_index = assumed_shape_actual_symbol(arena, node_index, context)
        end if
        if (symbol_index <= 0 .and. .not. is_empty_actual) then
            error_msg = 'assumed-shape actual was not declared'
            return
        end if
        ! A rank-2 whole column has rank-one result shape and is contiguous,
        ! but is not a rank-one source section view.  Classify it here so the
        ! descriptor gets the section's base/extent/byte stride rather than
        ! inheriting the source array's whole-array view.
        is_section_actual = actual_is_rank1_section_view(arena, node_index, context) .or. &
                            actual_is_contiguous_section(arena, node_index, context, &
                                                         allow_derived=.true.)
        section_is_contiguous = .false.
        is_component_projection = .false.
        select type (actual => arena%entries(node_index)%node)
        type is (component_access_node)
            call resolve_derived_array_component(arena, node_index, context, &
                                                 symbol_index, component_type, &
                                                 component_index, error_msg)
            if (len_trim(error_msg) > 0) return
            is_component_projection = .true.
        class default
        end select
        if (is_empty_actual) then
            call callee_dummy_declared_derived_type(arena, callee_name, param_pos, &
                                                    .false., dummy_type_name)
            if (len_trim(dummy_type_name) == 0) then
                error_msg = 'empty array constructor actual needs a derived-type '// &
                            'assumed-shape dummy'
                return
            end if
            dummy_type_index = find_derived_type(context, dummy_type_name)
            if (dummy_type_index <= 0) then
                error_msg = 'empty array constructor actual derived type was not '// &
                            'registered'
                return
            end if
            value_kind = VALUE_DERIVED
            element_bytes = 4_c_int64_t*int(total_component_slots(context, &
                dummy_type_index), c_int64_t)
        else
            value_kind = context%symbols(symbol_index)%value_kind
            dummy_type_index = 0
            if (value_kind == VALUE_DERIVED) then
                call callee_dummy_declared_derived_type(arena, callee_name, &
                    param_pos, .true., dummy_type_name)
                if (len_trim(dummy_type_name) > 0) dummy_type_index = &
                    find_derived_type(context, dummy_type_name)
            end if
        end if
        if (is_class_star_dummy) then
            select type (actual => arena%entries(node_index)%node)
            type is (array_slice_node)
                error_msg = 'class(*) assumed-shape actuals do not support '// &
                    'sections and non-array actuals'
                return
            class default
            end select
            if (rank < 1 .or. rank > 4) then
                error_msg = 'class(*) assumed-shape lowering currently supports '// &
                    'only rank-1 through rank-4 whole-array actuals'
                return
            end if
            if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
                value_kind /= VALUE_F64) then
                error_msg = 'class(*) assumed-shape lowering currently supports '// &
                    'only default integer and real(8) whole-array actuals'
                return
            end if
            if (value_kind == VALUE_F32 .or. value_kind == VALUE_F64) then
                class_star_type_id = int(TYPE_ID_REAL, c_int32_t)
            else
                class_star_type_id = int(TYPE_ID_INTEGER, c_int32_t)
            end if
        end if
        if (is_empty_actual) then
            base = null_ptr_operand(context)
        else if (is_component_projection) then
            value_kind = component_kind(context, component_type, component_index)
            element_bytes = int(equivalence_kind_size(value_kind), c_int64_t)
        else if (value_kind == VALUE_DERIVED) then
            ! A derived element spans its type's whole flat slot block, not one
            ! intrinsic slot, so the descriptor's element size and strides step
            ! by whole instances (#406).
            element_bytes = 4_c_int64_t*int(total_component_slots(context, &
                context%symbols(symbol_index)%derived_type_index), c_int64_t)
        else
            element_bytes = int(equivalence_kind_size(value_kind), c_int64_t)
        end if
        if (is_section_actual) then
            call reject_side_effectful_section_bounds(arena, node_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call rank1_section_stride_bytes(arena, node_index, context, &
                                            element_bytes, section_stride, &
                                            section_is_contiguous, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (is_materialized_expression) then
            base = context%symbols(symbol_index)%element_address
        else
            call assumed_shape_actual_base(arena, node_index, context, base, error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(ARRAY_DESCRIPTOR_BYTES, c_int64_t)), descriptor, &
                error_msg)) return

        if (.not. emit_ptr_store(context%session, base, descriptor, error_msg)) &
            return
        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, element_bytes), descriptor, &
                int(ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, c_int64_t), &
                error_msg)) return
        call store_descriptor_i32(context, assumed_shape_element_type(value_kind), &
            descriptor, int(ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, c_int64_t), &
            error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, int(rank, c_int32_t), descriptor, &
            int(ARRAY_DESCRIPTOR_RANK_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        ! The dummy borrows the actual's storage: allocated and associated, but
        ! never owning. A section view is contiguous only when its retained
        ! rank-1 stride is one element.
        flags = ARRAY_FLAG_ALLOCATED + ARRAY_FLAG_ASSOCIATED
        if (.not. is_component_projection .and. .not. is_empty_actual) then
            if (is_section_actual) then
                if (section_is_contiguous) flags = flags + ARRAY_FLAG_CONTIGUOUS
            else if (context%symbols(symbol_index)%array_stride_bytes == 0 .or. &
                     context%symbols(symbol_index)%array_stride_bytes == element_bytes) then
                flags = flags + ARRAY_FLAG_CONTIGUOUS
            end if
        end if
        call store_descriptor_i32(context, flags, descriptor, &
            int(ARRAY_DESCRIPTOR_FLAGS_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, 0_c_int32_t, descriptor, &
            int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        if (is_class_star_dummy) then
            call store_descriptor_i32(context, class_star_type_id, descriptor, &
                int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (dummy_type_index > 0) then
            ! For a CLASS(t) derived-array dummy, carry the actual dynamic type
            ! in the otherwise-unused header word.  The callee binds this word
            ! as its SELECT TYPE identity while retaining the descriptor's
            ! actual stride and section base.
            call class_actual_dynamic_type(context, symbol_index, dynamic_type_id, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i64_to_i32(context%session, dynamic_type_id, &
                                            dynamic_type_i32, error_msg)) return
            if (.not. emit_ptr_offset(context%session, descriptor, &
                    int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), &
                    dynamic_type_addr, error_msg)) return
            if (.not. emit_i32_store(context%session, dynamic_type_i32, &
                                     dynamic_type_addr, error_msg)) return
        end if

        ! Column-major strides: dimension 1 steps one element, dimension d steps
        ! the product of every lower dimension's extent.
        if (is_section_actual) then
            running = section_stride
        else if (is_empty_actual) then
            running = i64_immediate(context%session, element_bytes)
        else if (is_component_projection) then
            running = i64_immediate(context%session, int( &
                total_component_slots(context, component_type) * 4, c_int64_t))
        else if (context%symbols(symbol_index)%array_stride_bytes > 0) then
            running = i64_immediate(context%session, int( &
                context%symbols(symbol_index)%array_stride_bytes, c_int64_t))
        else
            running = i64_immediate(context%session, element_bytes)
        end if
        do dim = 1, rank
            if (is_materialized_expression) then
                extent = i64_immediate(context%session, int( &
                    context%symbols(symbol_index)%array_dim_sizes(dim), &
                    c_int64_t))
            else
                call assumed_shape_actual_extent(arena, node_index, context, rank, &
                                                 dim, extent, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            ! A bare-colon dummy has lower bound 1; an explicit lower bound such
            ! as 0: is part of the dummy view (F2018 8.5.8.3).
            if (.not. emit_i64_store_at(context%session, &
                    i64_immediate(context%session, int(dummy_lowers(dim), &
                                                        c_int64_t)), descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_LOWER_OFFSET, c_int64_t)), &
                    error_msg)) return
            if (.not. emit_i64_store_at(context%session, extent, descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                    error_msg)) return
            stride = running
            if (.not. emit_i64_store_at(context%session, stride, descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_STRIDE_OFFSET, c_int64_t)), &
                    error_msg)) return
            if (dim < rank) then
                if (.not. emit_i64_binary(context%session, LR_OP_MUL, running, &
                                          extent, running, error_msg)) return
            end if
        end do

        call set_empty(error_msg)
    end subroutine build_assumed_shape_descriptor

    subroutine build_forwarded_assumed_shape_descriptor(arena, context, rank, &
                                                        callee_name, param_pos, &
                                                        source_symbol, descriptor, &
                                                        error_msg)
        ! Compose a rank-1 through rank-4 descriptor for a forwarded assumed-shape dummy. The
        ! source descriptor is borrowed and remains the owner of no storage;
        ! the new stack descriptor copies its base, element metadata, extent,
        ! and signed stride, changing only the callee's lower bound.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: rank
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer, intent(in) :: source_symbol
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: source_descriptor, value_i64
        integer :: dummy_lowers(ARRAY_MAX_RANK)
        integer :: dim
        logical :: has_dummy_lowers

        if (rank < 1 .or. rank > 4) then
            error_msg = 'assumed-shape descriptor forwarding supports ranks 1 through 4'
            return
        end if
        source_descriptor = context%symbols(source_symbol)%runtime_descriptor_address
        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(ARRAY_DESCRIPTOR_BYTES, &
                                                     c_int64_t)), descriptor, &
                error_msg)) return

        if (.not. emit_i64_load_at(context%session, source_descriptor, &
                0_c_int64_t, value_i64, &
                error_msg)) return
        if (.not. emit_i64_store_at(context%session, value_i64, descriptor, &
                0_c_int64_t, error_msg)) return
        if (.not. emit_i64_load_at(context%session, source_descriptor, &
                int(ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, c_int64_t), value_i64, &
                error_msg)) return
        if (.not. emit_i64_store_at(context%session, value_i64, descriptor, &
                int(ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, c_int64_t), error_msg)) return
        call copy_forwarded_descriptor_i32(context, source_descriptor, descriptor, &
            int(ARRAY_DESCRIPTOR_ELEMENT_TYPE_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call copy_forwarded_descriptor_i32(context, source_descriptor, descriptor, &
            int(ARRAY_DESCRIPTOR_RANK_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call copy_forwarded_descriptor_i32(context, source_descriptor, descriptor, &
            int(ARRAY_DESCRIPTOR_FLAGS_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call copy_forwarded_descriptor_i32(context, source_descriptor, descriptor, &
            int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return

        call callee_dummy_assumed_shape_lowers(arena, callee_name, param_pos, &
                                               context, rank, dummy_lowers, &
                                               has_dummy_lowers, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. has_dummy_lowers) dummy_lowers(1:rank) = 1
        do dim = 1, rank
            if (.not. emit_i64_store_at(context%session, &
                    i64_immediate(context%session, int(dummy_lowers(dim), &
                                                         c_int64_t)), descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_LOWER_OFFSET, c_int64_t)), error_msg)) return
            if (.not. emit_i64_load_at(context%session, source_descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), value_i64, &
                    error_msg)) return
            if (.not. emit_i64_store_at(context%session, value_i64, descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), error_msg)) return
            if (.not. emit_i64_load_at(context%session, source_descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_STRIDE_OFFSET, c_int64_t)), value_i64, &
                    error_msg)) return
            if (.not. emit_i64_store_at(context%session, value_i64, descriptor, &
                    assumed_shape_dim_field_offset(dim, &
                        int(ARRAY_DIMENSION_STRIDE_OFFSET, c_int64_t)), error_msg)) return
        end do
        call set_empty(error_msg)
    end subroutine build_forwarded_assumed_shape_descriptor

    subroutine copy_forwarded_descriptor_i32(context, source_descriptor, descriptor, &
                                             offset, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: source_descriptor, descriptor
        integer(c_int64_t), intent(in) :: offset
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: source_field, target_field, value

        if (.not. emit_ptr_offset(context%session, source_descriptor, offset, &
                                  source_field, error_msg)) return
        if (.not. emit_i32_load(context%session, source_field, value, error_msg)) &
            return
        if (.not. emit_ptr_offset(context%session, descriptor, offset, target_field, &
                                  error_msg)) return
        if (.not. emit_i32_store(context%session, value, target_field, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine copy_forwarded_descriptor_i32

    subroutine rank1_section_stride_bytes(arena, node_index, context, element_bytes, &
                                          stride_bytes, is_contiguous, error_msg)
        ! Compute the view stride independently of the view base.  Bounds may
        ! be runtime expressions, but a runtime stride is still a signed byte
        ! stride in the canonical descriptor; no ownership or copy is involved.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer(c_int64_t), intent(in) :: element_bytes
        type(lr_operand_desc_t), intent(out) :: stride_bytes
        logical, intent(out) :: is_contiguous
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: stride_i32, stride_i64
        integer :: stride_index
        integer(c_int64_t) :: static_stride
        logical :: stride_is_runtime
        character(len=:), allocatable :: stride_error

        is_contiguous = .false.
        select type (slice => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, slice, context, info, error_msg, &
                                        allow_derived=.true.)
            if (len_trim(error_msg) > 0) return
            static_stride = info%section_strides(1)
            stride_index = 0
            if (slice%num_dimensions >= 1) then
                if (node_exists(arena, slice%bounds_indices(1))) then
                    select type (bounds => arena%entries( &
                            slice%bounds_indices(1))%node)
                    type is (range_expression_node)
                        stride_index = bounds%stride_index
                    type is (array_bounds_node)
                        stride_index = bounds%stride_index
                    class default
                    end select
                end if
            end if
            stride_is_runtime = .false.
            if (stride_index > 0) then
                call eval_i32_constant(arena, stride_index, context, &
                                       static_stride, stride_error)
                stride_is_runtime = len_trim(stride_error) > 0
            end if
            if (stride_is_runtime) then
                call lower_i32_expression(arena, stride_index, context, stride_i32, &
                                          error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_to_i64(context%session, stride_i32, &
                        stride_i64, error_msg, sign_extend=.true.)) return
                if (.not. emit_i64_binary(context%session, LR_OP_MUL, stride_i64, &
                        i64_immediate(context%session, element_bytes), stride_bytes, &
                        error_msg)) return
            else
                stride_bytes = i64_immediate(context%session, &
                                             static_stride*element_bytes)
            end if
            ! Runtime lower/upper bounds do not make a unit-stride view
            ! non-contiguous. An omitted, literal, or folded unit stride is
            ! contiguous; only a runtime-unknown stride withholds the flag.
            is_contiguous = static_stride == 1_c_int64_t .and. &
                            .not. stride_is_runtime
        class default
            error_msg = 'assumed-shape actual expected an array section'
        end select
    end subroutine rank1_section_stride_bytes

    subroutine reject_side_effectful_section_bounds(arena, node_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim, bounds_index

        call set_empty(error_msg)
        select type (slice => arena%entries(node_index)%node)
        type is (array_slice_node)
            do dim = 1, slice%num_dimensions
                bounds_index = slice%bounds_indices(dim)
                if (.not. node_exists(arena, bounds_index)) cycle
                select type (bounds => arena%entries(bounds_index)%node)
                type is (range_expression_node)
                    if (section_bound_has_user_function(arena, bounds%start_index) .or. &
                        section_bound_has_user_function(arena, bounds%end_index) .or. &
                        section_bound_has_user_function(arena, bounds%stride_index)) then
                        error_msg = 'side-effectful array-section bounds are not yet supported'
                        return
                    end if
                type is (array_bounds_node)
                    if (section_bound_has_user_function(arena, bounds%lower_bound_index) .or. &
                        section_bound_has_user_function(arena, bounds%upper_bound_index) .or. &
                        section_bound_has_user_function(arena, bounds%stride_index)) then
                        error_msg = 'side-effectful array-section bounds are not yet supported'
                        return
                    end if
                end select
            end do
        end select
    end subroutine reject_side_effectful_section_bounds

    recursive logical function section_bound_has_user_function(arena, node_index) &
            result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer :: left_index, right_index, line, column, i
        character(len=:), allocatable :: op, error_msg

        found = .false.
        if (node_index <= 0 .or. .not. node_exists(arena, node_index)) return
        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, op, left_index, right_index, &
                                    line, column, error_msg)
            if (len_trim(error_msg) > 0) return
            found = section_bound_has_user_function(arena, left_index) .or. &
                    section_bound_has_user_function(arena, right_index)
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. node%is_array_access .and. allocated(node%name) .and. &
                arena_has_function_def_named(arena, node%name)) then
                found = .true.
                return
            end if
            if (allocated(node%arg_indices)) then
                do i = 1, size(node%arg_indices)
                    if (section_bound_has_user_function(arena, node%arg_indices(i))) then
                        found = .true.
                        return
                    end if
                end do
            end if
        end select
    end function section_bound_has_user_function

    subroutine bind_assumed_shape_descriptor_params(arena, param_indices, &
                                                    context, callee_name, &
                                                    error_msg)
        ! Callee entry: every rank-1/rank-2/rank-3/rank-4 assumed-shape dummy's visible
        ! parameter is a descriptor pointer. Replace the symbol's parameter
        ! address with the descriptor's base address and cache each dimension's
        ! extent, so element addressing and SIZE read the descriptor and nothing
        ! else. Runs after define_reference_parameters and before the dummy's own
        ! declaration is lowered (bind_one_assumed_shape).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, sym_index, rank, dim
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: descriptor, base, extent_i64, extent_i32
        type(lr_operand_desc_t) :: stride_i64

        call set_empty(error_msg)
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            rank = callee_dummy_assumed_shape_dims(arena, callee_name, i)
            if (rank < 1) cycle
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym_index = find_symbol_compat(context, name)
            if (sym_index <= 0) then
                error_msg = 'assumed-shape dummy was not registered: '//trim(name)
                return
            end if
            context%symbols(sym_index)%is_assumed_shape_dummy = .true.
            descriptor = context%symbols(sym_index)%address
            if (callee_dummy_assumed_shape_optional(arena, callee_name, i)) then
                ! An absent OPTIONAL dummy receives a null descriptor pointer,
                ! so the loads must not run unconditionally. Read through the
                ! descriptor only when it is present and leave a null base and
                ! zero extents otherwise; the program must test PRESENT before
                ! using the dummy either way (F2018 15.5.2.12).
                call bind_optional_assumed_shape_descriptor(context, sym_index, &
                                                            rank, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (.not. emit_ptr_load(context%session, descriptor, base, &
                                    error_msg)) return
            context%symbols(sym_index)%runtime_descriptor_address = descriptor
            context%symbols(sym_index)%has_runtime_descriptor = .true.
            context%symbols(sym_index)%address = base
            context%symbols(sym_index)%element_address = base
            context%symbols(sym_index)%has_address = .true.
            context%symbols(sym_index)%is_reference = .true.
            if (rank == 1) then
                if (.not. emit_i64_load_at(context%session, descriptor, &
                        assumed_shape_dim_field_offset(1, &
                            int(ARRAY_DIMENSION_STRIDE_OFFSET, c_int64_t)), &
                        stride_i64, error_msg)) return
                context%symbols(sym_index)%has_runtime_array_stride = .true.
                context%symbols(sym_index)%runtime_array_stride_bytes = stride_i64
            end if
            call bind_polymorphic_array_element_stride(arena, context, &
                callee_name, i, sym_index, descriptor, error_msg)
            if (len_trim(error_msg) > 0) return
            if (callee_dummy_is_class_star(arena, callee_name, i)) then
                ! Reuse the scalar SELECT TYPE descriptor fields while the
                ! declaration path below binds this symbol as an array view.
                context%symbols(sym_index)%value_kind = VALUE_CLASS_STAR
                context%symbols(sym_index)%deferred_data = descriptor
                if (.not. emit_ptr_offset(context%session, descriptor, &
                        int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), &
                        context%symbols(sym_index)%deferred_length, error_msg)) &
                    return
            end if
            do dim = 1, rank
                if (.not. emit_i64_load_at(context%session, descriptor, &
                        assumed_shape_dim_field_offset(dim, &
                            int(ARRAY_DIMENSION_EXTENT_OFFSET, c_int64_t)), &
                        extent_i64, error_msg)) return
                if (.not. emit_liric_i64_to_i32(context%session, extent_i64, &
                                                extent_i32, error_msg)) return
                context%symbols(sym_index)%has_runtime_dim_size(dim) = .true.
                context%symbols(sym_index)%runtime_dim_size(dim) = extent_i32
            end do
        end do
    end subroutine bind_assumed_shape_descriptor_params

    subroutine bind_assumed_rank_descriptor_params(arena, param_indices, &
                                                   body_indices, context, callee_name, &
                                                   error_msg)
        ! An assumed-rank dummy receives the same canonical descriptor as an
        ! assumed-shape dummy, but its rank is read later by SELECT RANK. Keep
        ! the descriptor address until that construct establishes rank(1).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        integer, allocatable, intent(in) :: body_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, sym_index
        character(len=:), allocatable :: name
        type(lr_operand_desc_t) :: descriptor, descriptor_base

        call set_empty(error_msg)
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. dummy_is_assumed_rank(arena, body_indices, name)) cycle
            if (assumed_rank_dummy_value_kind(arena, param_indices, body_indices, &
                                              i, context) /= VALUE_F32 .and. &
                assumed_rank_dummy_value_kind(arena, param_indices, body_indices, &
                                              i, context) /= VALUE_F64) cycle
            sym_index = find_symbol_compat(context, name)
            if (sym_index <= 0) then
                error_msg = 'assumed-rank dummy was not registered: '//trim(name)
                return
            end if
            descriptor = context%symbols(sym_index)%address
            ! Allocatable dummies use the descriptor-aliasing ABI: the
            ! parameter address is already the caller's descriptor. Ordinary
            ! assumed-rank dummies receive a pointer to a temporary canonical
            ! descriptor and therefore retain the load below.
            if (dummy_is_allocatable_array(arena, body_indices, name)) then
                descriptor_base = descriptor
            else
                if (.not. emit_ptr_load(context%session, descriptor, descriptor_base, &
                                        error_msg)) return
            end if
            context%symbols(sym_index)%is_assumed_rank = .true.
            context%symbols(sym_index)%runtime_descriptor_address = descriptor_base
        end do
    end subroutine bind_assumed_rank_descriptor_params

    subroutine bind_polymorphic_array_element_stride(arena, context, callee_name, &
                                                     param_pos, sym_index, &
                                                     descriptor, error_msg)
        ! A class(t) array dummy accepts actuals whose dynamic element type
        ! extends t, and those elements are wider than t's own layout. The
        ! concrete element size is already in the descriptor the caller built,
        ! so read it once at entry and keep it as the element stride in i32
        ! slots (#422). A type(t) dummy is monomorphic: its element size is its
        ! declared type's size by definition, so it keeps the compile-time
        ! stride and emits nothing here.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer, intent(in) :: sym_index
        type(lr_operand_desc_t), intent(in) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: declared_type_index
        type(lr_operand_desc_t) :: elem_bytes, elem_slots, slots_i32
        type(lr_operand_desc_t) :: dynamic_type_addr, dynamic_type_id

        call set_empty(error_msg)
        call class_dummy_declared_type_index(context, arena, callee_name, &
                                             param_pos, declared_type_index)
        if (declared_type_index <= 0) return

        if (.not. emit_i64_load_at(context%session, descriptor, &
                int(ARRAY_DESCRIPTOR_ELEMENT_SIZE_OFFSET, c_int64_t), &
                elem_bytes, error_msg)) return
        ! Derived storage is a flat block of 4-byte i32 slots, so the stride in
        ! slots is the element size in bytes divided by four; the size is a
        ! positive multiple of four, so the shift and the division agree.
        if (.not. emit_i64_binary(context%session, LR_OP_LSHR, elem_bytes, &
                i64_immediate(context%session, 2_c_int64_t), elem_slots, &
                error_msg)) return
        if (.not. emit_liric_i64_to_i32(context%session, elem_slots, slots_i32, &
                                        error_msg)) return
        context%symbols(sym_index)%has_runtime_element_slots = .true.
        context%symbols(sym_index)%runtime_element_slots = slots_i32
        ! The caller stores the dynamic type id in the descriptor header's
        ! reserved word for CLASS(t) derived arrays.  Expose it as the same
        ! runtime identity used by scalar class selectors.
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), &
                dynamic_type_addr, error_msg)) return
        if (.not. emit_i32_load(context%session, dynamic_type_addr, &
                                dynamic_type_id, error_msg)) return
        context%symbols(sym_index)%dynamic_type_address = dynamic_type_addr
        context%symbols(sym_index)%has_dynamic_type_address = .true.
        context%symbols(sym_index)%is_polymorphic = .true.
    end subroutine bind_polymorphic_array_element_stride

    subroutine bind_optional_assumed_shape_descriptor(context, sym_index, rank, &
                                                       error_msg)
        ! An OPTIONAL assumed-shape dummy is absent whenever its descriptor
        ! pointer is null, so the descriptor must NOT be dereferenced in the
        ! prologue: the binding keeps the parameter pointer itself as the
        ! symbol's address, which is exactly what PRESENT tests against null.
        !
        ! Shape therefore cannot be read unconditionally here. The dummy still
        ! binds, with zero cached extents, so a procedure that only tests
        ! PRESENT lowers correctly. Referencing an absent optional's shape is
        ! not conforming anyway (F2018 15.5.2.12); a PRESENT optional with a
        ! usable shape is covered by the compile-time shape specialization in
        ! bind_one_assumed_shape. Full runtime shape for a present optional
        ! array dummy is an explicit non-goal of #334.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym_index
        integer, intent(in) :: rank
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: dim

        do dim = 1, rank
            context%symbols(sym_index)%has_runtime_dim_size(dim) = .true.
            context%symbols(sym_index)%runtime_dim_size(dim) = &
                i32_immediate(context%session, 0_c_int64_t)
        end do
        call set_empty(error_msg)
    end subroutine bind_optional_assumed_shape_descriptor
