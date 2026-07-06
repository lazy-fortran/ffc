    ! Reduce a comparison mask to a single accumulated i32 value: for all() the
    ! accumulator starts at 1 and ANDs each element; for any() it starts at 0 and
    ! ORs; for count (accumulate=='count') it starts at 0 and adds each 0/1
    ! element. Returns handled=.false. (with no error) for any form it cannot
    ! resolve at compile time so the caller keeps its existing behaviour.
    !
    ! Each comparison operand may be a whole stored/allocatable array, a rank-1
    ! array section (integer/real), an array constructor of scalars, or a scalar
    ! that broadcasts. Relational (==, /=, <, ...) and logical-equivalence
    ! (.eqv., .neqv.) operators are supported.
    subroutine try_general_mask_reduction(arena, node, context, accumulate, &
                                          handled, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: accumulate
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(reduction_operand_t) :: left, right
        character(len=:), allocatable :: op
        integer :: arg_index, left_idx, right_idx, line, col
        integer :: extent, compare_vk, i
        logical :: lok, rok, is_equiv
        integer(c_int) :: predicate
        type(lr_operand_desc_t) :: lhs, rhs, cond, acc, next_acc

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        arg_index = node%arg_indices(1)
        if (.not. is_binary_op(arena, arg_index)) then
            call try_bare_mask_reduction(arena, node, context, accumulate, &
                                         handled, value, error_msg)
            return
        end if
        call get_binary_op_info(arena, arg_index, op, left_idx, right_idx, &
                                line, col, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        is_equiv = is_equivalence_operator(op)
        if (.not. is_relational_operator(op) .and. .not. is_equiv) return

        call classify_mask_side(arena, context, left_idx, left, lok)
        call classify_mask_side(arena, context, right_idx, right, rok)
        if (.not. lok .or. .not. rok) return
        if (left%extent < 0 .and. right%extent < 0) return

        call mask_reduction_extent(left, right, extent)
        if (extent <= 0) return
        call mask_reduction_compare_vk(left, right, is_equiv, compare_vk, handled)
        if (.not. handled) return
        handled = .false.

        select case (compare_vk)
        case (VALUE_F32, VALUE_F64)
            call real_compare_predicate(op, predicate, error_msg)
        case default
            if (is_equiv) then
                call equivalence_compare_predicate(op, predicate)
            else
                call integer_compare_predicate(op, predicate, error_msg)
            end if
        end select
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if

        call mask_accumulator_seed(context, accumulate, acc)
        do i = 0, extent - 1
            call mask_side_element(arena, context, left, i, compare_vk, &
                                   is_equiv, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call mask_side_element(arena, context, right, i, compare_vk, &
                                   is_equiv, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (compare_vk)
            case (VALUE_F32)
                if (.not. emit_liric_f32_fcmp(context%session, predicate, lhs, &
                        rhs, cond, error_msg)) return
            case (VALUE_F64)
                if (.not. emit_liric_f64_fcmp(context%session, predicate, lhs, &
                        rhs, cond, error_msg)) return
            case default
                if (.not. emit_liric_i32_icmp(context%session, predicate, lhs, &
                        rhs, cond, error_msg)) return
            end select
            call mask_accumulator_step(context, accumulate, acc, cond, next_acc, &
                                       error_msg)
            if (len_trim(error_msg) > 0) return
            acc = next_acc
        end do
        value = acc
        handled = .true.
        call set_empty(error_msg)
    end subroutine try_general_mask_reduction

    ! all()/any()/count() of a bare mask that is not a comparison: a whole
    ! logical array, a rank-1 array section, or an array constructor of logical
    ! (or comparison) scalars, e.g. all([.true., .false.]), any(l2(:, 1)),
    ! count([a == b, c > 0]). Each element is normalised to 0/1 (elem /= 0).
    subroutine try_bare_mask_reduction(arena, node, context, accumulate, &
                                       handled, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: accumulate
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(reduction_operand_t) :: side
        logical :: ok
        integer :: i
        type(lr_operand_desc_t) :: raw, cond, acc, next_acc

        handled = .false.
        call set_empty(error_msg)
        call classify_mask_side(arena, context, node%arg_indices(1), side, ok)
        if (.not. ok) return
        ! Only an array constructor of logical/comparison scalars is handled
        ! here; a whole array stays on the existing dedicated handler, and a
        ! bare section (all/any masks are always logical) is left to decline.
        if (side%mode /= 3) return
        if (side%extent <= 0) return

        call mask_accumulator_seed(context, accumulate, acc)
        do i = 0, side%extent - 1
            call mask_side_element(arena, context, side, i, VALUE_I32, .true., &
                                   raw, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, raw, &
                    i32_immediate(context%session, 0_c_int64_t), cond, &
                    error_msg)) return
            call mask_accumulator_step(context, accumulate, acc, cond, next_acc, &
                                       error_msg)
            if (len_trim(error_msg) > 0) return
            acc = next_acc
        end do
        value = acc
        handled = .true.
        call set_empty(error_msg)
    end subroutine try_bare_mask_reduction

    logical function is_equivalence_operator(op) result(is_eq)
        character(len=*), intent(in) :: op

        select case (trim(adjustl(lowercase_text(op))))
        case ('.eqv.', '.neqv.', 'eqv', 'neqv')
            is_eq = .true.
        case default
            is_eq = .false.
        end select
    end function is_equivalence_operator

    subroutine equivalence_compare_predicate(op, predicate)
        character(len=*), intent(in) :: op
        integer(c_int), intent(out) :: predicate

        select case (trim(adjustl(lowercase_text(op))))
        case ('.neqv.', 'neqv')
            predicate = LR_CMP_NE
        case default
            predicate = LR_CMP_EQ
        end select
    end subroutine equivalence_compare_predicate

    subroutine mask_accumulator_seed(context, accumulate, acc)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: accumulate
        type(lr_operand_desc_t), intent(out) :: acc

        if (trim(accumulate) == 'all') then
            acc = i32_immediate(context%session, 1_c_int64_t)
        else
            acc = i32_immediate(context%session, 0_c_int64_t)
        end if
    end subroutine mask_accumulator_seed

    subroutine mask_accumulator_step(context, accumulate, acc, cond, next_acc, &
                                     error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: accumulate
        type(lr_operand_desc_t), intent(in) :: acc, cond
        type(lr_operand_desc_t), intent(out) :: next_acc
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: norm

        call set_empty(error_msg)
        select case (trim(accumulate))
        case ('all')
            if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, cond, &
                                      next_acc, error_msg)) return
        case ('count')
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, cond, &
                    i32_immediate(context%session, 0_c_int64_t), norm, &
                    error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc, norm, &
                                      next_acc, error_msg)) return
        case default
            if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, cond, &
                                      next_acc, error_msg)) return
        end select
    end subroutine mask_accumulator_step

    ! Classify one comparison operand into a reduction_operand_t. ok=.false.
    ! (with no error) rejects forms the general path does not handle.
    subroutine classify_mask_side(arena, context, idx, side, ok)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: idx
        type(reduction_operand_t), intent(out) :: side
        logical, intent(out) :: ok
        character(len=:), allocatable :: name, name_err, sec_err, flat_err
        integer :: s

        ok = .false.
        side%mode = 0
        side%sym = 0
        side%scalar_idx = idx
        side%vk = VALUE_I32
        side%extent = -1
        if (.not. node_exists(arena, idx)) return

        select type (n => arena%entries(idx)%node)
        type is (array_slice_node)
            call describe_array_section(arena, n, context, side%info, sec_err)
            if (len_trim(sec_err) > 0) return
            if (side%info%result_rank /= 1) return
            side%mode = 2
            side%sym = side%info%source_index
            side%vk = context%symbols(side%info%source_index)%value_kind
            side%extent = int(side%info%section_extents(side%info%kept_dims(1)))
            ok = side%extent > 0
            return
        type is (array_literal_node)
            if (allocated(n%syntax_style)) then
                if (n%syntax_style == 'implied_do') return
            end if
            if (.not. allocated(n%element_indices)) return
            allocate (side%flat(0))
            call flatten_constructor_elements(arena, n%element_indices, &
                                              side%flat, flat_err)
            if (len_trim(flat_err) > 0) then
                deallocate (side%flat)
                return
            end if
            side%mode = 3
            side%extent = size(side%flat)
            ok = .true.
            return
        end select

        if (is_identifier(arena, idx)) then
            call get_identifier_name(arena, idx, name, name_err)
            if (len_trim(name_err) == 0) then
                s = find_symbol(context, name)
                if (s > 0) then
                    if (context%symbols(s)%is_array .or. &
                        (context%symbols(s)%is_allocatable .and. &
                         context%symbols(s)%array_rank >= 1)) then
                        side%mode = 1
                        side%sym = s
                        side%vk = context%symbols(s)%value_kind
                        side%extent = array_symbol_extent(context, s)
                        ok = side%extent > 0
                        return
                    end if
                end if
            end if
        end if

        ! A bare function/intrinsic call may return an array (index(a, c),
        ! abs(arr), ...) that this path cannot evaluate elementwise. Decline it
        ! rather than misread it as a broadcast scalar, which would compare a
        ! garbage scalar against every element of the other operand.
        if (is_unhandled_call_operand(arena, idx)) return

        side%mode = 0
        side%scalar_idx = idx
        side%extent = -1
        ok = .true.
    end subroutine classify_mask_side

    logical function is_unhandled_call_operand(arena, idx) result(is_call)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx

        is_call = .false.
        if (.not. node_exists(arena, idx)) return
        select type (n => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            is_call = .not. n%is_array_access
        end select
    end function is_unhandled_call_operand

    subroutine mask_reduction_extent(left, right, extent)
        type(reduction_operand_t), intent(in) :: left, right
        integer, intent(out) :: extent

        extent = max(left%extent, right%extent)
        if (extent <= 0) return
        if (left%extent > 0 .and. right%extent > 0 .and. &
            left%extent /= right%extent) extent = 0
    end subroutine mask_reduction_extent

    ! Comparison kind for the general mask path. Equivalence operators compare
    ! logicals as i32. Otherwise promote among array-shaped operand kinds:
    ! real(8) beats real(4) beats integer; a logical array compares as integer.
    subroutine mask_reduction_compare_vk(left, right, is_equiv, compare_vk, ok)
        type(reduction_operand_t), intent(in) :: left, right
        logical, intent(in) :: is_equiv
        integer, intent(out) :: compare_vk
        logical, intent(out) :: ok

        ok = .true.
        compare_vk = VALUE_I32
        if (is_equiv) return
        if (mask_side_is(left, VALUE_F32) .or. mask_side_is(right, VALUE_F32)) &
            compare_vk = VALUE_F32
        if (mask_side_is(left, VALUE_F64) .or. mask_side_is(right, VALUE_F64)) &
            compare_vk = VALUE_F64
        if (mask_side_unsupported_kind(left) .or. &
            mask_side_unsupported_kind(right)) ok = .false.
    end subroutine mask_reduction_compare_vk

    logical function mask_side_is(side, vk) result(match)
        type(reduction_operand_t), intent(in) :: side
        integer, intent(in) :: vk

        match = .false.
        if (side%mode == 1 .or. side%mode == 2) match = side%vk == vk
    end function mask_side_is

    logical function mask_side_unsupported_kind(side) result(bad)
        type(reduction_operand_t), intent(in) :: side

        bad = .false.
        if (side%mode == 1 .or. side%mode == 2) then
            select case (side%vk)
            case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            case default
                bad = .true.
            end select
        end if
    end function mask_side_unsupported_kind

    ! Element i of one operand, coerced to compare_vk. is_equiv marks a logical
    ! equivalence comparison, whose scalar operands are logical (0/1) rather
    ! than numeric.
    subroutine mask_side_element(arena, context, side, i, compare_vk, is_equiv, &
                                 value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        type(reduction_operand_t), intent(in) :: side
        integer, intent(in) :: i, compare_vk
        logical, intent(in) :: is_equiv
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: raw
        integer :: native_vk, expr_idx

        call set_empty(error_msg)
        select case (side%mode)
        case (1)
            call load_array_linear_element(context, side%sym, &
                                           int(i, c_int64_t), raw, error_msg)
            if (len_trim(error_msg) > 0) return
            native_vk = side%vk
            if (native_vk == VALUE_LOGICAL) native_vk = VALUE_I32
            call coerce_reduction_operand(context, raw, native_vk, compare_vk, &
                                          value, error_msg)
        case (2)
            call lower_array_section_element_from_info(side%info, &
                    int(i, c_int64_t), context, raw, error_msg)
            if (len_trim(error_msg) > 0) return
            native_vk = side%vk
            if (native_vk == VALUE_LOGICAL) native_vk = VALUE_I32
            call coerce_reduction_operand(context, raw, native_vk, compare_vk, &
                                          value, error_msg)
        case (3)
            expr_idx = side%flat(i + 1)
            call mask_scalar_element(arena, expr_idx, context, compare_vk, &
                                     is_equiv, value, error_msg)
        case default
            call mask_scalar_element(arena, side%scalar_idx, context, &
                                     compare_vk, is_equiv, value, error_msg)
        end select
    end subroutine mask_side_element

    subroutine mask_scalar_element(arena, expr_idx, context, compare_vk, &
                                   is_equiv, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: expr_idx, compare_vk
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_equiv
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (is_equiv) then
            call lower_logical_expression(arena, expr_idx, context, value, &
                                          error_msg)
            return
        end if
        select case (compare_vk)
        case (VALUE_F32)
            call lower_f32_operand(arena, expr_idx, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_operand(arena, expr_idx, context, value, error_msg)
        case default
            call lower_i32_expression(arena, expr_idx, context, value, error_msg)
        end select
    end subroutine mask_scalar_element

    ! any()/all() over a whole-array relational comparison whose operands are
    ! declared/allocatable arrays, array constructors, or scalars, e.g.
    ! any(a /= b), all(p == [1, 2, 3]), any(arr /= 10). Unrolls over the static
    ! extent. Declines (handled=.false.) for any shape it cannot resolve at
    ! compile time so the caller keeps its existing behaviour.
    subroutine try_array_mask_reduction(arena, node, context, is_all, handled, &
                                        value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_all
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: accumulate
        integer :: arg_index, left_idx, right_idx, line, col
        integer :: lsym, rsym, lscalar, rscalar, extent, i, compare_vk
        integer, allocatable :: lflat(:), rflat(:)
        logical :: lok, rok
        character(len=:), allocatable :: op
        type(lr_operand_desc_t) :: lhs, rhs, cond, acc, next_acc
        integer(c_int) :: predicate

        handled = .false.
        call set_empty(error_msg)
        if (is_all) then
            accumulate = 'all'
        else
            accumulate = 'any'
        end if
        call try_general_mask_reduction(arena, node, context, accumulate, &
                                        handled, value, error_msg)
        if (handled .or. len_trim(error_msg) > 0) return

        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        arg_index = node%arg_indices(1)
        if (.not. is_binary_op(arena, arg_index)) return
        call get_binary_op_info(arena, arg_index, op, left_idx, right_idx, &
                                line, col, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        if (.not. is_relational_operator(op)) return

        call classify_reduction_side(arena, context, left_idx, lsym, lflat, &
                                     lscalar, lok)
        call classify_reduction_side(arena, context, right_idx, rsym, rflat, &
                                     rscalar, rok)
        if (.not. lok .or. .not. rok) return
        if (lsym <= 0 .and. rsym <= 0) return

        call reduction_extent(context, lsym, rsym, lflat, rflat, extent)
        if (extent <= 0) return
        call reduction_compare_kind(context, lsym, rsym, compare_vk, handled)
        if (.not. handled) then
            handled = .false.
            return
        end if
        handled = .false.

        select case (compare_vk)
        case (VALUE_F32, VALUE_F64)
            call real_compare_predicate(op, predicate, error_msg)
        case default
            call integer_compare_predicate(op, predicate, error_msg)
        end select
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if

        if (is_all) then
            acc = i32_immediate(context%session, 1_c_int64_t)
        else
            acc = i32_immediate(context%session, 0_c_int64_t)
        end if
        do i = 0, extent - 1
            call reduction_side_element(arena, context, lsym, lflat, lscalar, &
                                        i, compare_vk, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call reduction_side_element(arena, context, rsym, rflat, rscalar, &
                                        i, compare_vk, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (compare_vk)
            case (VALUE_F32)
                if (.not. emit_liric_f32_fcmp(context%session, predicate, lhs, &
                        rhs, cond, error_msg)) return
            case (VALUE_F64)
                if (.not. emit_liric_f64_fcmp(context%session, predicate, lhs, &
                        rhs, cond, error_msg)) return
            case default
                if (.not. emit_liric_i32_icmp(context%session, predicate, lhs, &
                        rhs, cond, error_msg)) return
            end select
            if (is_all) then
                if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, &
                        cond, next_acc, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, &
                        cond, next_acc, error_msg)) return
            end if
            acc = next_acc
        end do
        value = acc
        handled = .true.
        call set_empty(error_msg)
    end subroutine try_array_mask_reduction

    ! Classify one comparison operand: sym>0 for a whole declared/allocatable
    ! array, flat allocated for an array constructor of scalars, scalar_idx>0
    ! for any other scalar expression. ok=.false. rejects unsupported forms.
    subroutine classify_reduction_side(arena, context, idx, sym, flat, &
                                       scalar_idx, ok)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: idx
        integer, intent(out) :: sym, scalar_idx
        integer, allocatable, intent(out) :: flat(:)
        logical, intent(out) :: ok
        character(len=:), allocatable :: name, name_err, flat_err
        integer :: s

        sym = 0
        scalar_idx = 0
        ok = .false.
        if (.not. node_exists(arena, idx)) return

        if (is_identifier(arena, idx)) then
            call get_identifier_name(arena, idx, name, name_err)
            if (len_trim(name_err) == 0) then
                s = find_symbol(context, name)
                if (s > 0) then
                    if (context%symbols(s)%is_array) then
                        sym = s
                        ok = .true.
                        return
                    end if
                    if (context%symbols(s)%is_allocatable .and. &
                        context%symbols(s)%array_rank >= 1) then
                        sym = s
                        ok = .true.
                        return
                    end if
                end if
            end if
            scalar_idx = idx
            ok = .true.
            return
        end if

        select type (n => arena%entries(idx)%node)
        type is (array_literal_node)
            if (allocated(n%syntax_style)) then
                if (n%syntax_style == 'implied_do') return
            end if
            if (.not. allocated(n%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, n%element_indices, flat, &
                                              flat_err)
            if (len_trim(flat_err) > 0) then
                deallocate (flat)
                return
            end if
            ok = .true.
            return
        class default
            scalar_idx = idx
            ok = .true.
        end select
    end subroutine classify_reduction_side

    ! Static extent of the reduction from whichever operand is an array. Array
    ! constructors must match a present array extent; a lone constructor gives
    ! its own length. Returns 0 when extents conflict or none is known.
    subroutine reduction_extent(context, lsym, rsym, lflat, rflat, extent)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: lsym, rsym
        integer, allocatable, intent(in) :: lflat(:), rflat(:)
        integer, intent(out) :: extent
        integer :: le, re

        le = -1
        re = -1
        if (lsym > 0) le = array_symbol_extent(context, lsym)
        if (rsym > 0) re = array_symbol_extent(context, rsym)
        if (le > 0 .and. re > 0 .and. le /= re) then
            extent = 0
            return
        end if
        extent = max(le, re)
        if (extent <= 0) return
        if (allocated(lflat)) then
            if (size(lflat) /= extent) extent = 0
        end if
        if (extent <= 0) return
        if (allocated(rflat)) then
            if (size(rflat) /= extent) extent = 0
        end if
    end subroutine reduction_extent

    integer function array_symbol_extent(context, sym) result(extent)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: sym

        if (context%symbols(sym)%is_allocatable) then
            extent = context%symbols(sym)%allocatable_static_size
        else
            extent = context%symbols(sym)%array_size
        end if
    end function array_symbol_extent

    ! Comparison kind: real(8) if any array operand is f64, else real(4) if any
    ! is f32, else integer. ok=.false. for array kinds outside {i32, f32, f64}.
    subroutine reduction_compare_kind(context, lsym, rsym, compare_vk, ok)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: lsym, rsym
        integer, intent(out) :: compare_vk
        logical, intent(out) :: ok

        ok = .true.
        if (lsym > 0) then
            if (.not. supported_reduction_kind(context%symbols(lsym)%value_kind)) &
                ok = .false.
        end if
        if (rsym > 0) then
            if (.not. supported_reduction_kind(context%symbols(rsym)%value_kind)) &
                ok = .false.
        end if
        if (.not. ok) then
            compare_vk = VALUE_I32
            return
        end if
        compare_vk = VALUE_I32
        if (array_kind_is(context, lsym, VALUE_F32) .or. &
            array_kind_is(context, rsym, VALUE_F32)) compare_vk = VALUE_F32
        if (array_kind_is(context, lsym, VALUE_F64) .or. &
            array_kind_is(context, rsym, VALUE_F64)) compare_vk = VALUE_F64
    end subroutine reduction_compare_kind

    logical function supported_reduction_kind(vk) result(ok)
        integer, intent(in) :: vk

        ok = vk == VALUE_I32 .or. vk == VALUE_F32 .or. vk == VALUE_F64
    end function supported_reduction_kind

    logical function array_kind_is(context, sym, vk) result(match)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: sym, vk

        match = .false.
        if (sym > 0) match = context%symbols(sym)%value_kind == vk
    end function array_kind_is

    ! The i-th element of one comparison operand, coerced to compare_vk.
    subroutine reduction_side_element(arena, context, sym, flat, scalar_idx, i, &
                                      compare_vk, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym, scalar_idx, i, compare_vk
        integer, allocatable, intent(in) :: flat(:)
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: raw
        integer :: native_vk, expr_idx

        call set_empty(error_msg)
        if (sym > 0) then
            call load_array_linear_element(context, sym, int(i, c_int64_t), &
                                           raw, error_msg)
            if (len_trim(error_msg) > 0) return
            native_vk = context%symbols(sym)%value_kind
            call coerce_reduction_operand(context, raw, native_vk, compare_vk, &
                                          value, error_msg)
            return
        end if

        if (allocated(flat)) then
            expr_idx = flat(i + 1)
        else
            expr_idx = scalar_idx
        end if
        select case (compare_vk)
        case (VALUE_F32)
            call lower_f32_operand(arena, expr_idx, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_operand(arena, expr_idx, context, value, error_msg)
        case default
            call lower_i32_expression(arena, expr_idx, context, value, error_msg)
        end select
    end subroutine reduction_side_element

    subroutine coerce_reduction_operand(context, raw, native_vk, compare_vk, &
                                        value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: raw
        integer, intent(in) :: native_vk, compare_vk
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (native_vk == compare_vk) then
            value = raw
            return
        end if
        select case (compare_vk)
        case (VALUE_F64)
            if (native_vk == VALUE_I32) then
                if (.not. emit_liric_i32_to_f64(context%session, raw, value, &
                                                error_msg)) return
            else
                if (.not. emit_liric_f32_to_f64(context%session, raw, value, &
                                                error_msg)) return
            end if
        case (VALUE_F32)
            if (.not. emit_liric_i32_to_f32(context%session, raw, value, &
                                            error_msg)) return
        case default
            error_msg = 'unsupported reduction operand coercion'
        end select
    end subroutine coerce_reduction_operand
