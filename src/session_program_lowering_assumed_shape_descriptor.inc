    ! Assumed-shape dummy ABI (#334). A rank-1 or rank-2 assumed-shape dummy
    ! `a(:)` / `a(:,:)` is bound through exactly one canonical array descriptor
    ! (`ffc_array_descriptor`, `docs/ARRAY_DESCRIPTOR_ABI.md`). The caller
    ! materializes a descriptor for the actual and passes its address in the
    ! dummy's single visible parameter slot. No hidden extent arguments are
    ! appended, and the callee reads bounds, extents, and the base address from
    ! that descriptor alone.
    !
    ! The descriptor describes the DUMMY's view of the actual, which is what
    ! F2018 15.5.2.4 associates: `base` is the address of the actual's first
    ! element and every `lower_bound` is 1, because an assumed-shape dummy
    ! without an explicit lower bound has lower bound 1 whatever the actual's
    ! own bounds are. Extents and byte strides are the actual's, so column-major
    ! layout and element identity are preserved exactly.

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
        end select
    end function assumed_shape_actual_symbol

    integer function assumed_shape_actual_rank(arena, node_index, context) &
            result(rank)
        ! Rank the actual presents to the dummy: a whole array's declared rank,
        ! or 1 for a contiguous rank-1 section.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer :: symbol_index
        character(len=:), allocatable :: id_name, err

        rank = 0
        if (actual_is_contiguous_section(arena, node_index, context)) then
            rank = 1
            return
        end if
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

        if (actual_is_allocatable_array(arena, node_index, context)) then
            call allocatable_array_actual_data_address(arena, node_index, &
                                                       context, base, error_msg)
        else if (actual_is_contiguous_section(arena, node_index, context)) then
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

        call set_empty(error_msg)
        if (actual_is_allocatable_array(arena, node_index, context)) then
            call actual_extent_i64(arena, node_index, context, dim, extent, &
                                   error_msg)
            return
        end if

        extents = 0
        ok = .false.
        if (actual_is_contiguous_section(arena, node_index, context)) then
            call section_actual_decl_extents(context, node_index, 1, extents, ok)
        else if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            call array_decl_extents(context, id_name, rank, extents, ok)
        end if
        if (.not. ok) then
            error_msg = 'assumed-shape actual has no resolvable extent'
            return
        end if
        if (extents(dim) < 1) then
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
                                              callee_name, descriptor, error_msg)
        ! Materialize the canonical descriptor an assumed-shape dummy receives.
        ! One stack descriptor per call argument; it borrows the actual's
        ! storage and never claims ownership, so passing an array never changes
        ! the actual's allocation lifetime.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: rank
        character(len=*), intent(in) :: callee_name
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: base, extent, stride, running
        integer :: symbol_index, value_kind, actual_rank, dim
        integer(c_int64_t) :: element_bytes
        integer(c_int32_t) :: flags
        character(len=32) :: rank_text, actual_rank_text

        actual_rank = assumed_shape_actual_rank(arena, node_index, context)
        if (actual_rank /= rank) then
            write (rank_text, '(I0)') rank
            write (actual_rank_text, '(I0)') actual_rank
            error_msg = 'Rank mismatch in argument to '//trim(callee_name)// &
                        ': assumed-shape dummy of rank '//trim(rank_text)// &
                        ' received an actual of rank '//trim(actual_rank_text)
            return
        end if

        symbol_index = assumed_shape_actual_symbol(arena, node_index, context)
        if (symbol_index <= 0) then
            error_msg = 'assumed-shape actual was not declared'
            return
        end if
        value_kind = context%symbols(symbol_index)%value_kind
        if (value_kind == VALUE_DERIVED) then
            ! A derived element spans its type's whole flat slot block, not one
            ! intrinsic slot, so the descriptor's element size and strides step
            ! by whole instances (#406).
            element_bytes = 4_c_int64_t*int(total_component_slots(context, &
                context%symbols(symbol_index)%derived_type_index), c_int64_t)
        else
            element_bytes = int(equivalence_kind_size(value_kind), c_int64_t)
        end if

        call assumed_shape_actual_base(arena, node_index, context, base, error_msg)
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
        ! The dummy borrows the actual's storage: allocated and associated, and
        ! contiguous for every actual form this ABI accepts, but never owning.
        flags = ARRAY_FLAG_ALLOCATED + ARRAY_FLAG_ASSOCIATED + ARRAY_FLAG_CONTIGUOUS
        call store_descriptor_i32(context, flags, descriptor, &
            int(ARRAY_DESCRIPTOR_FLAGS_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return
        call store_descriptor_i32(context, 0_c_int32_t, descriptor, &
            int(ARRAY_DESCRIPTOR_RESERVED_OFFSET, c_int64_t), error_msg)
        if (len_trim(error_msg) > 0) return

        ! Column-major strides: dimension 1 steps one element, dimension d steps
        ! the product of every lower dimension's extent.
        running = i64_immediate(context%session, element_bytes)
        do dim = 1, rank
            call assumed_shape_actual_extent(arena, node_index, context, rank, &
                                             dim, extent, error_msg)
            if (len_trim(error_msg) > 0) return
            ! An assumed-shape dummy without an explicit lower bound has lower
            ! bound 1 regardless of the actual's own bounds (F2018 8.5.8.3).
            if (.not. emit_i64_store_at(context%session, &
                    i64_immediate(context%session, 1_c_int64_t), descriptor, &
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

    subroutine bind_assumed_shape_descriptor_params(arena, param_indices, &
                                                    context, callee_name, &
                                                    error_msg)
        ! Callee entry: every rank-1/rank-2 assumed-shape dummy's visible
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
            context%symbols(sym_index)%address = base
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
