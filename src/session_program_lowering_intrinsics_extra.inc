    ! lbound(a, dim): lower bound of array dimension dim (1-based).
    subroutine lower_lbound_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: array_name
        integer :: sym
        integer(c_int64_t) :: dim_val

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 1) then
            error_msg = 'lbound requires at least one argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, array_name)
        if (sym <= 0) then
            error_msg = 'lbound: array not declared: '//trim(array_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = 'lbound requires an array argument: '//trim(array_name)
            return
        end if
        dim_val = 1_c_int64_t
        if (size(node%arg_indices) >= 2) then
            call eval_i32_constant(arena, node%arg_indices(2), context, dim_val, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (dim_val < 1 .or. dim_val > context%symbols(sym)%array_rank) then
            error_msg = 'lbound dim out of range'
            return
        end if
        value = i32_immediate(context%session, &
                    int(context%symbols(sym)%array_dim_lowers(int(dim_val)), &
                        c_int64_t))
        call set_empty(error_msg)
    end subroutine lower_lbound_intrinsic

    ! ubound(a, dim): upper bound of array dimension dim (1-based).
    subroutine lower_ubound_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: array_name
        integer :: sym
        integer(c_int64_t) :: dim_val
        integer :: lower, size_d

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 1) then
            error_msg = 'ubound requires at least one argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, array_name)
        if (sym <= 0) then
            error_msg = 'ubound: array not declared: '//trim(array_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = 'ubound requires an array argument: '//trim(array_name)
            return
        end if
        dim_val = 1_c_int64_t
        if (size(node%arg_indices) >= 2) then
            call eval_i32_constant(arena, node%arg_indices(2), context, dim_val, &
                                   error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (dim_val < 1 .or. dim_val > context%symbols(sym)%array_rank) then
            error_msg = 'ubound dim out of range'
            return
        end if
        if (dim_val == 1_c_int64_t .and. &
            context%symbols(sym)%has_runtime_dim_size(1)) then
            ! A runtime-extent assumed-shape dummy always has lower bound 1, so
            ! ubound is exactly the runtime extent (1 + extent - 1).
            value = context%symbols(sym)%runtime_dim_size(1)
            call set_empty(error_msg)
            return
        end if
        lower = context%symbols(sym)%array_dim_lowers(int(dim_val))
        size_d = context%symbols(sym)%array_dim_sizes(int(dim_val))
        value = i32_immediate(context%session, int(lower + size_d - 1, c_int64_t))
        call set_empty(error_msg)
    end subroutine lower_ubound_intrinsic

    ! Resolve a mask argument that is an elementwise comparison (a > 2, a == 0,
    ! a >= 0): determine the array operand symbol, its size, the scalar operand
    ! AST index, the comparison predicate, and whether the array is on the left.
    ! Leaves mask_sym = 0 when node is not a supported comparison mask.
    subroutine resolve_comparison_mask(arena, node_index, context, mask_sym, &
                                       scalar_index, array_on_left, source_op, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: mask_sym
        integer, intent(out) :: scalar_index
        logical, intent(out) :: array_on_left
        character(len=:), allocatable, intent(out) :: source_op
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: left_idx, right_idx, line, col
        character(len=:), allocatable :: name, name_err
        integer :: sym

        call set_empty(error_msg)
        mask_sym = 0
        scalar_index = 0
        array_on_left = .true.
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, source_op, left_idx, &
                                right_idx, line, col, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (trim(source_op))
        case ('==', '=', '/=', '!=', '>', '>=', '<', '<=')
        case default
            call set_empty(source_op)
            return
        end select

        if (is_identifier(arena, left_idx)) then
            call get_identifier_name(arena, left_idx, name, name_err)
            if (len_trim(name_err) == 0) then
                sym = find_symbol(context, name)
                if (sym > 0) then
                    if (context%symbols(sym)%is_array) then
                        mask_sym = sym
                        scalar_index = right_idx
                        array_on_left = .true.
                        return
                    end if
                end if
            end if
        end if
        if (is_identifier(arena, right_idx)) then
            call get_identifier_name(arena, right_idx, name, name_err)
            if (len_trim(name_err) == 0) then
                sym = find_symbol(context, name)
                if (sym > 0) then
                    if (context%symbols(sym)%is_array) then
                        mask_sym = sym
                        scalar_index = left_idx
                        array_on_left = .false.
                        return
                    end if
                end if
            end if
        end if
    end subroutine resolve_comparison_mask

    ! Emit the i-th element of a comparison mask as an i32 0/1 operand.
    subroutine lower_comparison_mask_element(context, mask_sym, scalar_index, &
                                             array_on_left, source_op, arena, &
                                             linear_index, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: mask_sym
        integer, intent(in) :: scalar_index
        logical, intent(in) :: array_on_left
        character(len=*), intent(in) :: source_op
        type(ast_arena_t), intent(in) :: arena
        integer(c_int64_t), intent(in) :: linear_index
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: elem, scalar, lhs, rhs
        integer(c_int) :: predicate
        integer :: vk

        call load_array_linear_element(context, mask_sym, linear_index, elem, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        vk = context%symbols(mask_sym)%value_kind
        select case (vk)
        case (VALUE_F32)
            call lower_f32_expression(arena, scalar_index, context, scalar, &
                                      error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, scalar_index, context, scalar, &
                                      error_msg)
        case default
            call lower_i32_expression(arena, scalar_index, context, scalar, &
                                      error_msg)
        end select
        if (len_trim(error_msg) > 0) return

        if (array_on_left) then
            lhs = elem
            rhs = scalar
        else
            lhs = scalar
            rhs = elem
        end if

        select case (vk)
        case (VALUE_F32)
            call real_compare_predicate(source_op, predicate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_fcmp(context%session, predicate, lhs, rhs, &
                                          value, error_msg)) return
        case (VALUE_F64)
            call real_compare_predicate(source_op, predicate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_fcmp(context%session, predicate, lhs, rhs, &
                                          value, error_msg)) return
        case default
            call integer_compare_predicate(source_op, predicate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, predicate, lhs, rhs, &
                                          value, error_msg)) return
        end select
    end subroutine lower_comparison_mask_element

    ! Float comparison predicate mirroring integer_compare_predicate. Uses the
    ! ordered float predicates so NaN compares false, matching Fortran.
    subroutine real_compare_predicate(source_op, predicate, error_msg)
        character(len=*), intent(in) :: source_op
        integer(c_int), intent(out) :: predicate
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (lowercase_text(trim(adjustl(source_op))))
        case ('==', '=', '.eq.')
            predicate = LR_FCMP_OEQ
        case ('/=', '!=', '.ne.')
            predicate = LR_FCMP_ONE
        case ('>', '.gt.')
            predicate = LR_FCMP_OGT
        case ('>=', '.ge.')
            predicate = LR_FCMP_OGE
        case ('<', '.lt.')
            predicate = LR_FCMP_OLT
        case ('<=', '.le.')
            predicate = LR_FCMP_OLE
        case default
            error_msg = 'unsupported real comparison: '//trim(source_op)
            predicate = LR_FCMP_OEQ
        end select
    end subroutine real_compare_predicate

    ! True when node is an any()/all() call, which yield a LOGICAL scalar and so
    ! must print as T/F rather than the i32 1/0 of their integer lowering.
    logical function is_logical_reduction_call(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_logical_reduction_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%name)) return
            if (node%is_array_access) return
            if (is_contained_i32_function(context, node%name)) return
            is_logical_reduction_call = same_name(node%name, 'any') .or. &
                                        same_name(node%name, 'all')
        end select
    end function is_logical_reduction_call

    ! count(mask): number of .true. (nonzero) elements in a logical array.
    subroutine lower_logical_array_count_intrinsic(arena, node, context, value, &
                                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mask_name, source_op
        integer :: sym, array_size, i, scalar_index
        logical :: array_on_left
        type(lr_operand_desc_t) :: elem, total, next_total

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 1) then
            error_msg = 'count requires one logical array argument'
            return
        end if

        ! A comparison-expression mask (count(a > 2)) reduces over the compared
        ! array's elements; an identifier mask names a declared logical array.
        call resolve_comparison_mask(arena, node%arg_indices(1), context, sym, &
                                     scalar_index, array_on_left, source_op, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        if (sym > 0) then
            array_size = context%symbols(sym)%array_size
            total = i32_immediate(context%session, 0_c_int64_t)
            do i = 0, array_size - 1
                call lower_comparison_mask_element(context, sym, scalar_index, &
                    array_on_left, source_op, arena, int(i, c_int64_t), elem, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, total, &
                                          elem, next_total, error_msg)) return
                total = next_total
            end do
            value = total
            call set_empty(error_msg)
            return
        end if

        call get_identifier_name(arena, node%arg_indices(1), mask_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, mask_name)
        if (sym <= 0) then
            error_msg = 'count: array not declared: '//trim(mask_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = 'count requires an array argument: '//trim(mask_name)
            return
        end if
        array_size = context%symbols(sym)%array_size
        total = i32_immediate(context%session, 0_c_int64_t)
        do i = 0, array_size - 1
            call load_array_linear_element(context, sym, int(i, c_int64_t), &
                                           elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, total, elem, &
                                      next_total, error_msg)) return
            total = next_total
        end do
        value = total
        call set_empty(error_msg)
    end subroutine lower_logical_array_count_intrinsic

    ! any(mask): 1 if any element nonzero, else 0.
    subroutine lower_logical_array_any_intrinsic(arena, node, context, value, &
                                                 error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mask_name, source_op
        integer :: sym, array_size, i, scalar_index
        logical :: array_on_left
        type(lr_operand_desc_t) :: elem, acc, next_acc

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 1) then
            error_msg = 'any requires one logical array argument'
            return
        end if

        block
            logical :: handled
            call try_array_mask_reduction(arena, node, context, .false., &
                                          handled, value, error_msg)
            if (handled .or. len_trim(error_msg) > 0) return
        end block

        call resolve_comparison_mask(arena, node%arg_indices(1), context, sym, &
                                     scalar_index, array_on_left, source_op, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        ! OR all 0/1 mask elements: nonzero iff any is .true.
        if (sym > 0) then
            array_size = context%symbols(sym)%array_size
            acc = i32_immediate(context%session, 0_c_int64_t)
            do i = 0, array_size - 1
                call lower_comparison_mask_element(context, sym, scalar_index, &
                    array_on_left, source_op, arena, int(i, c_int64_t), elem, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, elem, &
                                          next_acc, error_msg)) return
                acc = next_acc
            end do
            value = acc
            call set_empty(error_msg)
            return
        end if

        call get_identifier_name(arena, node%arg_indices(1), mask_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, mask_name)
        if (sym <= 0) then
            error_msg = 'any: array not declared: '//trim(mask_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = 'any requires an array argument: '//trim(mask_name)
            return
        end if
        array_size = context%symbols(sym)%array_size
        acc = i32_immediate(context%session, 0_c_int64_t)
        do i = 0, array_size - 1
            call load_array_linear_element(context, sym, int(i, c_int64_t), &
                                           elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, elem, &
                                      next_acc, error_msg)) return
            acc = next_acc
        end do
        value = acc
        call set_empty(error_msg)
    end subroutine lower_logical_array_any_intrinsic

    ! all(mask): 1 if all elements nonzero, else 0.
    subroutine lower_logical_array_all_intrinsic(arena, node, context, value, &
                                                 error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mask_name, source_op
        integer :: sym, array_size, i, scalar_index
        logical :: array_on_left
        type(lr_operand_desc_t) :: elem, acc, next_acc, cond

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 1) then
            error_msg = 'all requires one logical array argument'
            return
        end if

        block
            logical :: handled
            call try_array_mask_reduction(arena, node, context, .true., &
                                          handled, value, error_msg)
            if (handled .or. len_trim(error_msg) > 0) return
        end block

        call resolve_comparison_mask(arena, node%arg_indices(1), context, sym, &
                                     scalar_index, array_on_left, source_op, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        ! Normalise each mask element to 0/1, AND them: 1 iff all are .true.
        if (sym > 0) then
            array_size = context%symbols(sym)%array_size
            acc = i32_immediate(context%session, 1_c_int64_t)
            do i = 0, array_size - 1
                call lower_comparison_mask_element(context, sym, scalar_index, &
                    array_on_left, source_op, arena, int(i, c_int64_t), elem, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, elem, &
                        i32_immediate(context%session, 0_c_int64_t), cond, &
                        error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, cond, &
                                          next_acc, error_msg)) return
                acc = next_acc
            end do
            value = acc
            call set_empty(error_msg)
            return
        end if

        call get_identifier_name(arena, node%arg_indices(1), mask_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, mask_name)
        if (sym <= 0) then
            error_msg = 'all: array not declared: '//trim(mask_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = 'all requires an array argument: '//trim(mask_name)
            return
        end if
        ! Normalise each element to 0/1 via (elem /= 0), AND them.
        array_size = context%symbols(sym)%array_size
        acc = i32_immediate(context%session, 1_c_int64_t)
        do i = 0, array_size - 1
            call load_array_linear_element(context, sym, int(i, c_int64_t), &
                                           elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, elem, &
                    i32_immediate(context%session, 0_c_int64_t), cond, &
                    error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, cond, &
                                      next_acc, error_msg)) return
            acc = next_acc
        end do
        value = acc
        call set_empty(error_msg)
    end subroutine lower_logical_array_all_intrinsic

    subroutine try_alloc_component_mask_reduction(arena, node, context, is_all, &
                                                  handled, value, error_msg)
        ! all/any of an elementwise integer comparison where one operand is a
        ! whole rank-1 allocatable array component and the other a compile-time-
        ! sized integer array. Unrolls over the array's static extent (the
        ! component must conform). Leaves handled .false. for any other shape.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_all
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: arg_index, left_idx, right_idx, comp_access, arr_sym, n, i
        logical :: comp_on_left
        character(len=:), allocatable :: op
        type(lr_operand_desc_t) :: desc_addr, comp_elem, arr_elem, lhs, rhs
        type(lr_operand_desc_t) :: cond, acc, next_acc
        integer(c_int) :: predicate

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        arg_index = node%arg_indices(1)
        if (.not. node_exists(arena, arg_index)) return
        left_idx = 0
        right_idx = 0
        select type (b => arena%entries(arg_index)%node)
        type is (binary_op_node)
            if (.not. is_relational_operator(b%operator)) return
            op = b%operator
            left_idx = b%left_index
            right_idx = b%right_index
        class default
            return
        end select
        call classify_mask_operands(arena, left_idx, right_idx, context, &
                                    comp_access, arr_sym, comp_on_left)
        if (comp_access <= 0 .or. arr_sym <= 0) return
        if (derived_component_access_kind_at(arena, comp_access, context) &
            /= VALUE_I32) return
        if (context%symbols(arr_sym)%value_kind /= VALUE_I32) return
        if (context%symbols(arr_sym)%is_allocatable) then
            n = context%symbols(arr_sym)%allocatable_static_size
        else
            n = context%symbols(arr_sym)%array_size
        end if
        if (n <= 0) return
        call integer_compare_predicate(op, predicate, error_msg)
        if (len_trim(error_msg) > 0) return
        call alloc_array_component_descriptor(arena, comp_access, context, &
                                              desc_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (is_all) then
            acc = i32_immediate(context%session, 1_c_int64_t)
        else
            acc = i32_immediate(context%session, 0_c_int64_t)
        end if
        do i = 0, n - 1
            call load_alloc_component_element(context, desc_addr, VALUE_I32, &
                i32_immediate(context%session, int(i, c_int64_t)), comp_elem, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, arr_sym, int(i, c_int64_t), &
                                           arr_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (comp_on_left) then
                lhs = comp_elem
                rhs = arr_elem
            else
                lhs = arr_elem
                rhs = comp_elem
            end if
            if (.not. emit_liric_i32_icmp(context%session, predicate, lhs, rhs, &
                    cond, error_msg)) return
            if (is_all) then
                if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, cond, &
                        next_acc, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, cond, &
                        next_acc, error_msg)) return
            end if
            acc = next_acc
        end do
        value = acc
        handled = .true.
        call set_empty(error_msg)
    end subroutine try_alloc_component_mask_reduction

    subroutine classify_mask_operands(arena, left_idx, right_idx, context, &
                                      comp_access, arr_sym, comp_on_left)
        ! comp_access = arena index of a whole alloc-array-component operand (or
        ! 0); arr_sym = symbol of a whole intrinsic array operand (or 0).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: left_idx, right_idx
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: comp_access, arr_sym
        logical, intent(out) :: comp_on_left

        comp_access = 0
        arr_sym = 0
        comp_on_left = .true.
        if (operand_is_alloc_component(arena, left_idx, context)) then
            comp_access = left_idx
            comp_on_left = .true.
            arr_sym = whole_array_symbol(arena, right_idx, context)
        else if (operand_is_alloc_component(arena, right_idx, context)) then
            comp_access = right_idx
            comp_on_left = .false.
            arr_sym = whole_array_symbol(arena, left_idx, context)
        end if
    end subroutine classify_mask_operands

    logical function operand_is_alloc_component(arena, idx, context) result(is_comp)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context

        is_comp = .false.
        if (.not. node_exists(arena, idx)) return
        select type (acc => arena%entries(idx)%node)
        type is (component_access_node)
            is_comp = component_access_is_alloc_array(arena, acc, context)
        end select
    end function operand_is_alloc_component

    integer function whole_array_symbol(arena, idx, context) result(sym)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err

        sym = 0
        if (.not. is_identifier(arena, idx)) return
        call get_identifier_name(arena, idx, name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, name)
        if (sym <= 0) return
        if (.not. is_elementwise_array_operand(context, sym)) sym = 0
    end function whole_array_symbol

    ! maxloc(a [,dim] [,mask]) / minloc(a [,dim] [,mask]) for a rank-1
    ! integer or real array, returning the scalar 1-based index of the first
    ! extreme element. With a logical-array mask, only masked elements are
    ! considered; if none qualify the result is 0 (F2018 semantics). A scalar
    ! dim argument, when present, must select the single rank-1 dimension.
    subroutine lower_array_locate_intrinsic(arena, node, context, value, &
                                            error_msg, want_max)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in) :: want_max
        character(len=:), allocatable :: which
        integer :: sym, mask_sym

        if (want_max) then
            which = 'maxloc'
        else
            which = 'minloc'
        end if

        call resolve_locate_arguments(arena, node, context, which, sym, mask_sym, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        if (context%symbols(sym)%array_size <= 0) then
            error_msg = trim(which)//' requires a non-empty array'
            return
        end if
        call emit_locate_scan(context, want_max, sym, mask_sym, value, error_msg)
    end subroutine lower_array_locate_intrinsic

    ! Resolve the array argument and an optional logical-array mask. A scalar
    ! extra argument is the dim selector, which for a rank-1 array must be 1.
    ! Keyword arguments (dim=, mask=) arrive as keyword assignment nodes that
    ! are unwrapped to their value node before classifying.
    subroutine resolve_locate_arguments(arena, node, context, which, sym, &
                                        mask_sym, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: which
        integer, intent(out) :: sym, mask_sym
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: array_name, name
        integer :: i, ai, value_node
        integer(c_int64_t) :: dim_val

        sym = 0
        mask_sym = 0
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 1) then
            error_msg = trim(which)//' requires an array argument'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(1), array_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, array_name)
        if (sym <= 0) then
            error_msg = trim(which)//': array not declared: '//trim(array_name)
            return
        end if
        if (.not. context%symbols(sym)%is_array) then
            error_msg = trim(which)//' requires an array argument: '//trim(array_name)
            return
        end if
        if (context%symbols(sym)%array_rank /= 1) then
            error_msg = trim(which)//' supports only rank-1 arrays: '//trim(array_name)
            return
        end if

        do ai = 2, size(node%arg_indices)
            value_node = locate_arg_value_node(arena, node%arg_indices(ai))
            if (is_identifier(arena, value_node)) then
                call get_identifier_name(arena, value_node, name, error_msg)
                if (len_trim(error_msg) > 0) return
                i = find_symbol(context, name)
                if (i > 0) then
                    if (context%symbols(i)%is_array) then
                        mask_sym = i
                        cycle
                    end if
                end if
            end if
            call eval_i32_constant(arena, value_node, context, dim_val, error_msg)
            if (len_trim(error_msg) > 0) return
            if (dim_val /= 1_c_int64_t) then
                error_msg = trim(which)//' dim must be 1 for a rank-1 array'
                return
            end if
        end do

        if (mask_sym > 0) then
            if (context%symbols(mask_sym)%array_size /= &
                context%symbols(sym)%array_size) then
                error_msg = trim(which)//' mask must match the array extent'
                return
            end if
        end if
        call set_empty(error_msg)
    end subroutine resolve_locate_arguments

    ! Linear scan over the rank-1 array tracking the best value and its 1-based
    ! index. best_idx starts at 0 (no element chosen): the first qualifying
    ! element seeds best_val and later strictly-better elements replace it,
    ! preserving the first-occurrence rule (no replace on ties). A mask gates
    ! whether each element is eligible to be chosen.
    subroutine emit_locate_scan(context, want_max, sym, mask_sym, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: want_max
        integer, intent(in) :: sym, mask_sym
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: array_size, i
        type(lr_operand_desc_t) :: best_val, best_idx, cand, cand_idx
        type(lr_operand_desc_t) :: better, take, mask_elem, masked
        type(lr_operand_desc_t) :: sel_val, sel_idx

        array_size = context%symbols(sym)%array_size
        best_idx = i32_immediate(context%session, 0_c_int64_t)
        best_val = i32_immediate(context%session, 0_c_int64_t)
        do i = 0, array_size - 1
            call load_array_linear_element(context, sym, int(i, c_int64_t), &
                                           cand, error_msg)
            if (len_trim(error_msg) > 0) return
            cand_idx = i32_immediate(context%session, int(i + 1, c_int64_t))

            call locate_is_better(context, want_max, &
                                  context%symbols(sym)%value_kind, cand, best_val, &
                                  best_idx, better, error_msg)
            if (len_trim(error_msg) > 0) return
            take = better
            if (mask_sym > 0) then
                call load_array_linear_element(context, mask_sym, &
                                               int(i, c_int64_t), mask_elem, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, &
                        mask_elem, i32_immediate(context%session, 0_c_int64_t), &
                        masked, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_AND, take, &
                                          masked, sel_idx, error_msg)) return
                take = sel_idx
            end if

            call select_value(context, take, cand, best_val, sel_val, error_msg)
            if (len_trim(error_msg) > 0) return
            best_val = sel_val
            call select_value(context, take, cand_idx, best_idx, sel_idx, error_msg)
            if (len_trim(error_msg) > 0) return
            best_idx = sel_idx
        end do

        value = best_idx
        call set_empty(error_msg)
    end subroutine emit_locate_scan

    ! better = (best_idx == 0) .or. (want_max ? cand > best_val : cand < best_val)
    subroutine locate_is_better(context, want_max, vk, cand, best_val, best_idx, &
                                better, error_msg)
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: want_max
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: cand, best_val, best_idx
        type(lr_operand_desc_t), intent(out) :: better
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cmp, unset
        integer(c_int) :: predicate

        if (vk == VALUE_F32) then
            if (want_max) then
                predicate = LR_FCMP_OGT
            else
                predicate = LR_FCMP_OLT
            end if
            if (.not. emit_liric_f32_fcmp(context%session, predicate, cand, &
                                          best_val, cmp, error_msg)) return
        else if (vk == VALUE_F64) then
            if (want_max) then
                predicate = LR_FCMP_OGT
            else
                predicate = LR_FCMP_OLT
            end if
            if (.not. emit_liric_f64_fcmp(context%session, predicate, cand, &
                                          best_val, cmp, error_msg)) return
        else
            if (want_max) then
                predicate = LR_CMP_SGT
            else
                predicate = LR_CMP_SLT
            end if
            if (.not. emit_liric_i32_icmp(context%session, predicate, cand, &
                                          best_val, cmp, error_msg)) return
        end if

        ! No element chosen yet (best_idx == 0) forces the candidate in.
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, best_idx, &
                i32_immediate(context%session, 0_c_int64_t), unset, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_OR, cmp, unset, better, &
                                  error_msg)) return
        call set_empty(error_msg)
    end subroutine locate_is_better

    ! A keyword actual argument (dim=1, mask=m) is stored as a keyword
    ! assignment node; return its value node. A positional argument is
    ! returned unchanged.
    function locate_arg_value_node(arena, arg_index) result(value_node)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        integer :: value_node

        value_node = arg_index
        if (.not. node_exists(arena, arg_index)) return
        select type (n => arena%entries(arg_index)%node)
        type is (assignment_node)
            if (n%value_index > 0) value_node = n%value_index
        end select
    end function locate_arg_value_node

    ! scan(s, set [, back]): 1-based position of first char of s found in set; 0 if none.
    subroutine lower_scan_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, set_data, set_len
        type(lr_operand_desc_t) :: i_addr, j_addr, res_addr
        type(lr_operand_desc_t) :: i_val, j_val, cond, sb, tb, tmp
        integer(c_int32_t) :: outer_hdr, outer_body, inner_hdr, inner_body
        integer(c_int32_t) :: matched, advance_j, advance_i, done

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 2) then
            error_msg = 'scan requires two character arguments'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_expr_operands(arena, node%arg_indices(2), context, set_data, &
                                set_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_alloca(context%session, i_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, j_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, res_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), i_addr, &
                error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), res_addr, &
                error_msg)) return

        outer_hdr = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        inner_hdr = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        matched = create_liric_block(context%session)
        advance_j = create_liric_block(context%session)
        advance_i = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! outer_hdr: if i < s_len, scan this char; else done.
        if (.not. set_liric_block(context%session, outer_hdr, error_msg)) return
        context%current_block_id = outer_hdr
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, &
                s_len, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, outer_body, done, &
                error_msg)) return

        ! outer_body: reset j=0 and enter inner loop.
        if (.not. set_liric_block(context%session, outer_body, error_msg)) return
        context%current_block_id = outer_body
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), j_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return

        ! inner_hdr: if j < set_len, compare; else advance_i.
        if (.not. set_liric_block(context%session, inner_hdr, error_msg)) return
        context%current_block_id = inner_hdr
        if (.not. emit_i32_load(context%session, j_addr, j_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, j_val, &
                set_len, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, inner_body, advance_i, &
                error_msg)) return

        ! inner_body: if s[i] == set[j] goto matched else advance_j.
        if (.not. set_liric_block(context%session, inner_body, error_msg)) return
        context%current_block_id = inner_body
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, s_data, i_val, sb, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, set_data, j_val, &
                tb, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, sb, tb, cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, matched, advance_j, &
                error_msg)) return

        ! matched: result = i + 1; goto done.
        if (.not. set_liric_block(context%session, matched, error_msg)) return
        context%current_block_id = matched
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, res_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, done, error_msg)) return

        ! advance_j: j = j + 1; goto inner_hdr.
        if (.not. set_liric_block(context%session, advance_j, error_msg)) return
        context%current_block_id = advance_j
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, j_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, j_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return

        ! advance_i: i = i + 1; goto outer_hdr.
        if (.not. set_liric_block(context%session, advance_i, error_msg)) return
        context%current_block_id = advance_i
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, i_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! done: load result.
        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, res_addr, value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine lower_scan_intrinsic

    ! verify(s, set [, back]): 1-based position of first char of s NOT in set; 0 if all in set.
    subroutine lower_verify_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, set_data, set_len
        type(lr_operand_desc_t) :: i_addr, j_addr, res_addr, found_addr
        type(lr_operand_desc_t) :: i_val, j_val, cond, sb, tb, found_val, tmp
        integer(c_int32_t) :: outer_hdr, outer_body, inner_hdr, inner_body
        integer(c_int32_t) :: set_found, advance_j, check_found
        integer(c_int32_t) :: store_result, advance_i, done

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 2) then
            error_msg = 'verify requires two character arguments'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call char_expr_operands(arena, node%arg_indices(2), context, set_data, &
                                set_len, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_alloca(context%session, i_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, j_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, res_addr, error_msg)) return
        if (.not. emit_i32_alloca(context%session, found_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), i_addr, &
                error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), res_addr, &
                error_msg)) return

        outer_hdr = create_liric_block(context%session)
        outer_body = create_liric_block(context%session)
        inner_hdr = create_liric_block(context%session)
        inner_body = create_liric_block(context%session)
        set_found = create_liric_block(context%session)
        advance_j = create_liric_block(context%session)
        check_found = create_liric_block(context%session)
        store_result = create_liric_block(context%session)
        advance_i = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! outer_hdr: if i < s_len, check this char; else done.
        if (.not. set_liric_block(context%session, outer_hdr, error_msg)) return
        context%current_block_id = outer_hdr
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, &
                s_len, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, outer_body, done, &
                error_msg)) return

        ! outer_body: found=0; j=0; enter inner loop.
        if (.not. set_liric_block(context%session, outer_body, error_msg)) return
        context%current_block_id = outer_body
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), found_addr, &
                error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), j_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return

        ! inner_hdr: if j < set_len, compare; else check_found.
        if (.not. set_liric_block(context%session, inner_hdr, error_msg)) return
        context%current_block_id = inner_hdr
        if (.not. emit_i32_load(context%session, j_addr, j_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, j_val, &
                set_len, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, inner_body, check_found, &
                error_msg)) return

        ! inner_body: if s[i] == set[j] goto set_found else advance_j.
        if (.not. set_liric_block(context%session, inner_body, error_msg)) return
        context%current_block_id = inner_body
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, s_data, i_val, sb, &
                error_msg)) return
        if (.not. emit_liric_char_byte_zext(context%session, set_data, j_val, &
                tb, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, sb, tb, cond, &
                error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, set_found, advance_j, &
                error_msg)) return

        ! set_found: found=1; skip remainder of inner loop.
        if (.not. set_liric_block(context%session, set_found, error_msg)) return
        context%current_block_id = set_found
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 1_c_int64_t), found_addr, &
                error_msg)) return
        if (.not. emit_liric_br(context%session, check_found, error_msg)) return

        ! advance_j: j = j + 1; continue inner loop.
        if (.not. set_liric_block(context%session, advance_j, error_msg)) return
        context%current_block_id = advance_j
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, j_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, j_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, inner_hdr, error_msg)) return

        ! check_found: if found==0 then this char is not in set.
        if (.not. set_liric_block(context%session, check_found, error_msg)) return
        context%current_block_id = check_found
        if (.not. emit_i32_load(context%session, found_addr, found_val, &
                error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, found_val, &
                i32_immediate(context%session, 0_c_int64_t), cond, error_msg)) &
            return
        if (.not. emit_liric_condbr(context%session, cond, store_result, &
                advance_i, error_msg)) return

        ! store_result: result = i + 1; goto done.
        if (.not. set_liric_block(context%session, store_result, error_msg)) return
        context%current_block_id = store_result
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, res_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, done, error_msg)) return

        ! advance_i: i = i + 1; go back to outer loop.
        if (.not. set_liric_block(context%session, advance_i, error_msg)) return
        context%current_block_id = advance_i
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, i_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, outer_hdr, error_msg)) return

        ! done: load result.
        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i32_load(context%session, res_addr, value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine lower_verify_intrinsic

    ! repeat(s, n): character result of s repeated n times.
    subroutine lower_repeat_deferred(arena, node, context, out_data, out_length, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: out_data
        type(lr_operand_desc_t), intent(out) :: out_length
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: s_data, s_len, n_val
        type(lr_operand_desc_t) :: total_len, total_len64, buf, off_i32, off64
        type(lr_operand_desc_t) :: s_len64, dst, null_pos
        type(lr_operand_desc_t) :: i_addr, i_val, cond, tmp
        integer(c_int32_t) :: hdr, body, done

        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) < 2) then
            error_msg = 'repeat requires two arguments: string and count'
            return
        end if
        call char_expr_operands(arena, node%arg_indices(1), context, s_data, &
                                s_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, n_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        ! total_len = s_len * n_val; allocate buf[total_len + 1].
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, s_len, n_val, &
                total_len, error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, total_len, total_len64, &
                error_msg)) return
        if (.not. emit_alloca_bytes(context%session, &
                emit_or_add1(context, total_len64), buf, error_msg)) return

        if (.not. emit_liric_i32_to_i64(context%session, s_len, s_len64, &
                error_msg)) return

        if (.not. emit_i32_alloca(context%session, i_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, 0_c_int64_t), i_addr, &
                error_msg)) return

        hdr = create_liric_block(context%session)
        body = create_liric_block(context%session)
        done = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        ! hdr: if i < n_val, copy; else done.
        if (.not. set_liric_block(context%session, hdr, error_msg)) return
        context%current_block_id = hdr
        if (.not. emit_i32_load(context%session, i_addr, i_val, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, i_val, n_val, &
                cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, body, done, &
                error_msg)) return

        ! body: dst = buf + i*s_len; memcpy(dst, s_data, s_len); i++.
        if (.not. set_liric_block(context%session, body, error_msg)) return
        context%current_block_id = body
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, i_val, s_len, &
                off_i32, error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, off_i32, off64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, off64, &
                dst, error_msg)) return
        if (.not. emit_memcpy(context%session, dst, s_data, s_len64, &
                error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, i_val, &
                i32_immediate(context%session, 1_c_int64_t), tmp, error_msg)) &
            return
        if (.not. emit_i32_store(context%session, tmp, i_addr, error_msg)) return
        if (.not. emit_liric_br(context%session, hdr, error_msg)) return

        ! done: null-terminate at buf[total_len].
        if (.not. set_liric_block(context%session, done, error_msg)) return
        context%current_block_id = done
        context%current_block_terminated = .false.
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, buf, total_len64, &
                null_pos, error_msg)) return
        if (.not. emit_liric_store_char_byte(context%session, null_pos, &
                i32_immediate(context%session, 0_c_int64_t), &
                i32_immediate(context%session, 0_c_int64_t), error_msg)) return

        out_data = buf
        out_length = total_len
        call set_empty(error_msg)
    end subroutine lower_repeat_deferred
