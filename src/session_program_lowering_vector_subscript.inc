    ! Vector-subscript assignment: a(v) = rhs, where v is a rank-1 integer
    ! array (a named array or an integer array constructor). Fortran requires
    ! the index vector and the whole right-hand side to be evaluated before any
    ! element of the target is redefined, so the lowering here is a two-phase
    ! scatter: every right-hand-side value and every subscript is materialised
    ! first, and only then are the stores emitted. That makes a self-referential
    ! assignment such as a(v) = a(w) alias-safe without a heap temporary,
    ! because the extents are compile-time constants and the whole scatter
    ! unrolls. Repeated indices therefore scatter in array element order, so the
    ! last write wins, matching gfortran.

    subroutine describe_vector_subscript(arena, target, context, base_index, &
                                         vector_index, literal_index, extent, &
                                         handled, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: base_index
        integer, intent(out) :: vector_index
        integer, intent(out) :: literal_index
        integer, intent(out) :: extent
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: subscript_index
        integer :: sym

        handled = .false.
        base_index = 0
        vector_index = 0
        literal_index = 0
        extent = 0
        call set_empty(error_msg)

        if (.not. allocated(target%name)) return
        if (.not. allocated(target%arg_indices)) return
        if (size(target%arg_indices) /= 1) return
        base_index = find_symbol_compat(context, target%name)
        if (base_index <= 0) then
            base_index = 0
            return
        end if
        if (.not. context%symbols(base_index)%is_array) then
            base_index = 0
            return
        end if
        if (context%symbols(base_index)%is_allocatable) then
            base_index = 0
            return
        end if
        if (context%symbols(base_index)%array_rank /= 1) then
            base_index = 0
            return
        end if

        subscript_index = target%arg_indices(1)
        if (.not. node_exists(arena, subscript_index)) then
            base_index = 0
            return
        end if

        if (is_identifier(arena, subscript_index)) then
            call get_identifier_name(arena, subscript_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol_compat(context, id_name)
            if (sym <= 0) then
                base_index = 0
                return
            end if
            if (.not. context%symbols(sym)%is_array) then
                base_index = 0
                return
            end if
            if (context%symbols(sym)%is_allocatable) then
                base_index = 0
                return
            end if
            if (context%symbols(sym)%array_rank /= 1) then
                base_index = 0
                return
            end if
            if (context%symbols(sym)%value_kind /= VALUE_I32) then
                error_msg = 'vector subscript requires an integer index vector: '// &
                            trim(id_name)
                return
            end if
            vector_index = sym
            extent = context%symbols(sym)%array_size
            handled = .true.
            return
        end if

        select type (subscript => arena%entries(subscript_index)%node)
        type is (array_literal_node)
            if (.not. allocated(subscript%element_indices)) then
                base_index = 0
                return
            end if
            literal_index = subscript_index
            extent = size(subscript%element_indices)
            handled = extent > 0
            if (.not. handled) base_index = 0
        end select
    end subroutine describe_vector_subscript

    subroutine lower_vector_subscript_assignment(arena, node, target, context, &
                                                 handled, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: values(:)
        type(lr_operand_desc_t), allocatable :: indices(:)
        integer :: base_index, vector_index, literal_index, extent
        integer :: vk, k

        call describe_vector_subscript(arena, target, context, base_index, &
                                       vector_index, literal_index, extent, &
                                       handled, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. handled) return

        vk = context%symbols(base_index)%value_kind
        select case (vk)
        case (VALUE_I32, VALUE_F32, VALUE_F64)
            ! supported scatter element kinds
        case default
            error_msg = 'vector subscript assignment supports integer, '// &
                        'real(4), and real(8) arrays'
            return
        end select

        allocate (values(extent))
        allocate (indices(extent))

        ! Phase 1: materialise the right-hand side, then the subscripts.
        call materialise_vector_subscript_rhs(arena, node%value_index, context, &
                                              vk, extent, values, error_msg)
        if (len_trim(error_msg) > 0) return
        call materialise_vector_subscript_indices(arena, context, base_index, &
                                                  vector_index, literal_index, &
                                                  extent, indices, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Phase 2: scatter in array element order.
        do k = 1, extent
            call store_array_element_at_operand(context, base_index, indices(k), &
                                                values(k), error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'vector_subscript_store: '//error_msg
                return
            end if
        end do
        call set_empty(error_msg)
    end subroutine lower_vector_subscript_assignment

    subroutine materialise_vector_subscript_indices(arena, context, base_index, &
                                                    vector_index, literal_index, &
                                                    extent, indices, error_msg)
        ! Produce the zero-based linear offsets of the scatter targets. A named
        ! index vector is read element by element at run time; a constructor
        ! subscript folds to constants, which lets bounds be checked here.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: base_index
        integer, intent(in) :: vector_index
        integer, intent(in) :: literal_index
        integer, intent(in) :: extent
        type(lr_operand_desc_t), intent(inout) :: indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: raw
        integer(c_int64_t) :: lower_bound, index_value
        integer :: k

        lower_bound = int(context%symbols(base_index)%array_lower_bound, c_int64_t)
        if (vector_index > 0) then
            do k = 1, extent
                call load_array_linear_element(context, vector_index, &
                                               int(k - 1, c_int64_t), raw, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_SUB, raw, &
                        i32_immediate(context%session, lower_bound), indices(k), &
                        error_msg)) then
                    error_msg = 'vector_subscript_index: '//error_msg
                    return
                end if
            end do
            call set_empty(error_msg)
            return
        end if

        select type (subscript => arena%entries(literal_index)%node)
        type is (array_literal_node)
            do k = 1, extent
                call eval_i32_constant(arena, subscript%element_indices(k), &
                                       context, index_value, error_msg)
                if (len_trim(error_msg) > 0) then
                    error_msg = 'vector subscript requires an integer index vector'
                    return
                end if
                if (index_value < lower_bound) then
                    error_msg = 'vector subscript index is out of bounds'
                    return
                end if
                if (index_value > lower_bound + &
                    int(context%symbols(base_index)%array_size, c_int64_t) - &
                    1_c_int64_t) then
                    error_msg = 'vector subscript index is out of bounds'
                    return
                end if
                indices(k) = i32_immediate(context%session, index_value - lower_bound)
            end do
        class default
            error_msg = 'vector subscript requires an integer index vector'
            return
        end select
        call set_empty(error_msg)
    end subroutine materialise_vector_subscript_indices

    subroutine materialise_vector_subscript_rhs(arena, value_index, context, vk, &
                                                extent, values, error_msg)
        ! Evaluate the right-hand side into `extent` operands before any store.
        ! Array constructors, whole rank-1 arrays, gathers a(w), and scalar
        ! broadcasts are covered; anything else is rejected rather than silently
        ! mis-scattered.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        integer, intent(in) :: extent
        type(lr_operand_desc_t), intent(inout) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: gather_indices(:)
        type(lr_operand_desc_t) :: scalar_value
        integer :: source_index, source_vector, source_literal, source_extent
        integer :: k, sym
        logical :: is_vector_gather

        call set_empty(error_msg)
        if (.not. node_exists(arena, value_index)) then
            error_msg = 'vector subscript assignment has no right-hand side'
            return
        end if

        select type (rhs => arena%entries(value_index)%node)
        type is (array_literal_node)
            if (.not. allocated(rhs%element_indices)) then
                error_msg = 'array constructor has no elements'
                return
            end if
            if (size(rhs%element_indices) /= extent) then
                error_msg = 'vector subscript assignment is nonconformable: '// &
                            'index vector and right-hand side differ in size'
                return
            end if
            do k = 1, extent
                call lower_vector_subscript_scalar(arena, rhs%element_indices(k), &
                                                   context, vk, values(k), &
                                                   error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            return
        type is (identifier_node)
            sym = find_symbol_compat(context, rhs%name)
            if (sym > 0) then
                if (context%symbols(sym)%is_array) then
                    if (context%symbols(sym)%is_allocatable) then
                        error_msg = 'vector subscript assignment requires a '// &
                                    'fixed-size right-hand side'
                        return
                    end if
                    if (context%symbols(sym)%array_rank /= 1) then
                        error_msg = 'vector subscript assignment requires a '// &
                                    'rank-1 right-hand side'
                        return
                    end if
                    if (context%symbols(sym)%array_size /= extent) then
                        error_msg = 'vector subscript assignment is '// &
                                    'nonconformable: index vector and '// &
                                    'right-hand side differ in size'
                        return
                    end if
                    do k = 1, extent
                        call load_array_linear_element(context, sym, &
                                                       int(k - 1, c_int64_t), &
                                                       values(k), error_msg)
                        if (len_trim(error_msg) > 0) return
                    end do
                    return
                end if
            end if
        type is (call_or_subscript_node)
            call describe_vector_subscript(arena, rhs, context, source_index, &
                                           source_vector, source_literal, &
                                           source_extent, is_vector_gather, &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
            if (is_vector_gather) then
                if (source_extent /= extent) then
                    error_msg = 'vector subscript assignment is nonconformable: '// &
                                'index vector and right-hand side differ in size'
                    return
                end if
                allocate (gather_indices(extent))
                call materialise_vector_subscript_indices(arena, context, &
                                                          source_index, &
                                                          source_vector, &
                                                          source_literal, extent, &
                                                          gather_indices, error_msg)
                if (len_trim(error_msg) > 0) return
                do k = 1, extent
                    call load_array_element_at_operand(context, source_index, &
                                                       gather_indices(k), &
                                                       values(k), error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
                return
            end if
        end select

        ! Scalar broadcast.
        call lower_vector_subscript_scalar(arena, value_index, context, vk, &
                                           scalar_value, error_msg)
        if (len_trim(error_msg) > 0) return
        do k = 1, extent
            values(k) = scalar_value
        end do
    end subroutine materialise_vector_subscript_rhs

    subroutine lower_vector_subscript_scalar(arena, expr_index, context, vk, &
                                             value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (vk == VALUE_F32) then
            call lower_f32_operand(arena, expr_index, context, value, error_msg)
        else if (vk == VALUE_F64) then
            call lower_f64_operand(arena, expr_index, context, value, error_msg)
        else
            call lower_i32_expression(arena, expr_index, context, value, error_msg)
        end if
    end subroutine lower_vector_subscript_scalar
