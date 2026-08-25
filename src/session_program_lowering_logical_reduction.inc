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
        if (trim(adjustl(lowercase_text(op))) == '.not.' .or. &
            trim(adjustl(lowercase_text(op))) == 'not') then
            call try_bare_mask_reduction(arena, node, context, accumulate, &
                                         handled, value, error_msg)
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
        if (left%mode == 4) then
            call create_array_expression_temp(context, VALUE_LOGICAL, &
                                              left%extent, 0, left%sym, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (right%mode == 4) then
            call create_array_expression_temp(context, VALUE_LOGICAL, &
                                              right%extent, 0, right%sym, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
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
        ! Array constructors and elementwise logical expressions are handled
        ! here; a bare identifier stays on the dedicated whole-array path.
        if (side%mode /= 3 .and. side%mode /= 4) return
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
    recursive subroutine classify_mask_side(arena, context, idx, side, ok)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: idx
        type(reduction_operand_t), intent(out) :: side
        logical, intent(out) :: ok
        character(len=:), allocatable :: name, name_err, sec_err, flat_err
        character(len=:), allocatable :: bin_op
        integer :: s, bin_left, bin_right, bin_line, bin_col
        type(reduction_operand_t) :: left_side, right_side
        logical :: lok, rok, array_real_call

        ok = .false.
        side%mode = 0
        side%sym = 0
        side%scalar_idx = idx
        side%vk = VALUE_I32
        side%extent = -1
        side%apply_abs = .false.
        side%apply_complex_abs = .false.
        side%component = 0
        if (.not. node_exists(arena, idx)) return

        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, bin_op, bin_left, bin_right, &
                                    bin_line, bin_col, name_err)
            if (len_trim(name_err) > 0) return
            if (trim(adjustl(lowercase_text(bin_op))) == '.not.') then
                call classify_mask_side(arena, context, bin_right, right_side, ok)
                if (.not. ok) return
                if (right_side%extent > 0) then
                    side%mode = 4
                    side%sym = right_side%sym
                    side%vk = VALUE_LOGICAL
                    side%extent = right_side%extent
                else
                    side%mode = 0
                    side%scalar_idx = idx
                    side%vk = VALUE_LOGICAL
                    side%extent = -1
                end if
                ok = .true.
                return
            end if
            if (is_logical_connective(bin_op) .or. &
                is_relational_operator(bin_op)) then
                call classify_mask_side(arena, context, bin_left, left_side, lok)
                call classify_mask_side(arena, context, bin_right, right_side, rok)
                if (.not. lok .or. .not. rok) return
                if (left_side%extent > 0 .and. right_side%extent > 0 .and. &
                    left_side%extent /= right_side%extent) return
                if (max(left_side%extent, right_side%extent) > 0) then
                    side%mode = 4
                    side%sym = left_side%sym
                    if (side%sym <= 0) side%sym = right_side%sym
                    side%vk = VALUE_LOGICAL
                    side%extent = max(left_side%extent, right_side%extent)
                else
                    side%mode = 0
                    side%scalar_idx = idx
                    side%vk = VALUE_LOGICAL
                    side%extent = -1
                end if
                ok = .true.
                return
            end if
            if (trim(bin_op) == '+' .or. trim(bin_op) == '-' .or. &
                trim(bin_op) == '*' .or. trim(bin_op) == '/' .or. &
                trim(bin_op) == '**') then
                call classify_mask_side(arena, context, bin_left, left_side, lok)
                call classify_mask_side(arena, context, bin_right, right_side, rok)
                if (.not. lok .or. .not. rok) return
                if (left_side%extent > 0 .or. right_side%extent > 0) then
                    side%mode = 4
                    side%scalar_idx = idx
                    side%sym = left_side%sym
                    if (side%sym <= 0) side%sym = right_side%sym
                    side%vk = merge(VALUE_F64, VALUE_F32, &
                        left_side%vk == VALUE_F64 .or. right_side%vk == VALUE_F64)
                    side%extent = max(left_side%extent, right_side%extent)
                    ok = side%extent > 0
                    return
                end if
            end if
        end if

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
        type is (call_or_subscript_node)
            array_real_call = is_elemental_real_call(arena, idx)
            ! Intrinsic ABS has dedicated scalar/complex element handling
            ! below.  Treating it as a generic elemental real call first
            ! loses the complex source kind for ABS(ABS(z)) and routes the
            ! whole-array operand to the identifier-only fallback.
            if (array_real_call .and. allocated(n%name) .and. &
                same_name(n%name, 'abs') .and. n%is_intrinsic) then
                array_real_call = .false.
            end if
            if (.not. array_real_call) then
                if (allocated(n%name)) then
                    if (same_name(n%name, 'aint') .or. same_name(n%name, 'anint')) then
                        if (allocated(n%arg_indices)) then
                            array_real_call = size(n%arg_indices) == 1
                        end if
                    end if
                end if
            end if
            if (array_real_call) then
                call classify_mask_side(arena, context, n%arg_indices(1), &
                                        right_side, ok)
                if (.not. ok .or. right_side%extent <= 0) then
                    ok = .false.
                    return
                end if
                side%mode = 4
                side%sym = right_side%sym
                side%vk = right_side%vk
                side%extent = right_side%extent
                ok = side%vk == VALUE_F32 .or. side%vk == VALUE_F64
                return
            end if
            if (is_contained_function_reference(n, context)) then
                if (contained_function_kind(context, n%name) /= VALUE_LOGICAL) return
                side%mode = 0
                side%scalar_idx = idx
                side%vk = VALUE_LOGICAL
                side%extent = -1
                ok = .true.
                return
            end if
            if (n%is_intrinsic .and. allocated(n%name) .and. &
                same_name(n%name, 'abs') .and. allocated(n%arg_indices) .and. &
                size(n%arg_indices) == 1) then
                call classify_mask_side(arena, context, n%arg_indices(1), side, ok)
                if (.not. ok) return
                if (side%extent <= 0) then
                    ok = .false.
                    return
                end if
                if (side%vk /= VALUE_F32 .and. side%vk /= VALUE_F64) then
                    if (side%vk == VALUE_C4 .or. side%vk == VALUE_C8) then
                        side%vk = merge(VALUE_F32, VALUE_F64, side%vk == VALUE_C4)
                        side%apply_complex_abs = .true.
                    else
                        ok = .false.
                        return
                    end if
                end if
                side%apply_abs = .true.
                return
            end if
        end select

        select type (n => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            if (.not. n%is_array_access .and. allocated(n%name) .and. &
                allocated(n%arg_indices) .and. size(n%arg_indices) == 1 .and. &
                same_name(n%name, 'aimag')) then
                call classify_mask_side(arena, context, n%arg_indices(1), side, ok)
                if (.not. ok .or. side%extent <= 0) then
                    ok = .false.
                    return
                end if
                if (side%vk == VALUE_C4 .or. side%vk == VALUE_C8) then
                    side%vk = merge(VALUE_F32, VALUE_F64, side%vk == VALUE_C4)
                    side%component = 1
                    ok = .true.
                    return
                end if
                ok = .false.
                return
            end if
        end select

        if (is_identifier(arena, idx)) then
            call get_identifier_name(arena, idx, name, name_err)
            if (len_trim(name_err) == 0) then
                s = find_symbol_compat(context, name)
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

        ! Another bare function/intrinsic call may return an array that this
        ! path cannot evaluate elementwise. Decline it
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
        if (mask_side_is(left, VALUE_F32) .or. mask_side_is(right, VALUE_F32) .or. &
            mask_side_is_mode4(left, VALUE_F32) .or. &
            mask_side_is_mode4(right, VALUE_F32)) &
            compare_vk = VALUE_F32
        if (mask_side_is(left, VALUE_F64) .or. mask_side_is(right, VALUE_F64) .or. &
            mask_side_is_mode4(left, VALUE_F64) .or. &
            mask_side_is_mode4(right, VALUE_F64)) &
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

    logical function mask_side_is_mode4(side, vk) result(match)
        type(reduction_operand_t), intent(in) :: side
        integer, intent(in) :: vk

        match = side%mode == 4 .and. side%vk == vk
    end function mask_side_is_mode4

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
            if (side%apply_complex_abs .or. side%component == 1) then
                block
                    type(lr_operand_desc_t) :: re_value, im_value
                    type(lr_operand_desc_t) :: call_args(2)
                    integer :: complex_vk
                    complex_vk = merge(VALUE_C4, VALUE_C8, side%vk == VALUE_F32)
                    call load_complex_array_linear_element(context, side%sym, i, &
                        complex_vk, re_value, im_value, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (side%component == 1) then
                        raw = im_value
                    else if (side%vk == VALUE_F32) then
                        call_args(1) = re_value
                        call_args(2) = im_value
                        if (.not. emit_liric_f32_call(context%session, 'hypotf', &
                                call_args, raw, error_msg)) return
                    else
                        call_args(1) = re_value
                        call_args(2) = im_value
                        if (.not. emit_liric_f64_call(context%session, 'hypot', &
                                call_args, raw, error_msg)) return
                    end if
                    value = raw
                    if (side%apply_abs) call mask_real_abs(context, compare_vk, &
                        value, error_msg)
                    return
                end block
            end if
            call load_array_linear_element(context, side%sym, &
                                           int(i, c_int64_t), raw, error_msg)
            if (len_trim(error_msg) > 0) return
            native_vk = side%vk
            if (native_vk == VALUE_LOGICAL) native_vk = VALUE_I32
            call coerce_reduction_operand(context, raw, native_vk, compare_vk, &
                                          value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (side%apply_abs) call mask_real_abs(context, compare_vk, value, &
                                                  error_msg)
        case (2)
            call lower_array_section_element_from_info(side%info, &
                    int(i, c_int64_t), context, raw, error_msg)
            if (len_trim(error_msg) > 0) return
            native_vk = side%vk
            if (native_vk == VALUE_LOGICAL) native_vk = VALUE_I32
            call coerce_reduction_operand(context, raw, native_vk, compare_vk, &
                                          value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (side%apply_abs) call mask_real_abs(context, compare_vk, value, &
                                                  error_msg)
        case (3)
            expr_idx = side%flat(i + 1)
            call mask_scalar_element(arena, expr_idx, context, compare_vk, &
                                     is_equiv, value, error_msg)
        case (4)
            call lower_mask_expression_element(arena, side%scalar_idx, side%vk, &
                                               i, context, value, error_msg)
        case default
            call mask_scalar_element(arena, side%scalar_idx, context, &
                                     compare_vk, is_equiv, value, error_msg)
        end select
    end subroutine mask_side_element

    recursive subroutine lower_mask_expression_element(arena, idx, compare_vk, &
                                                       linear_index, context, &
                                                       value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx, compare_vk, linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, op, err, logical_op
        character(len=:), allocatable :: lit_value, lit_type
        integer :: sym, left, right, line, column, opcode, source_vk
        integer, allocatable :: flat(:)
        logical :: array_real_call
        type(lr_operand_desc_t) :: lhs, rhs, re_value, im_value, promoted
        type(lr_operand_desc_t) :: operand, zero, cond, true_val, false_val
        type(lr_operand_desc_t) :: call_args(2)
        integer :: complex_vk

        call set_empty(error_msg)
        if (is_identifier(arena, idx)) then
            call get_identifier_name(arena, idx, name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol_compat(context, name)
            if (sym > 0 .and. (context%symbols(sym)%is_array .or. &
                (context%symbols(sym)%is_allocatable .and. &
                 context%symbols(sym)%array_rank >= 1))) then
                call load_array_linear_element(context, sym, &
                    int(linear_index, c_int64_t), value, error_msg)
                return
            end if
            if (compare_vk == VALUE_LOGICAL) then
                call lower_logical_expression(arena, idx, context, value, error_msg)
            else if (compare_vk == VALUE_F32) then
                call lower_f32_expression(arena, idx, context, value, error_msg)
            else
                call lower_f64_expression(arena, idx, context, value, error_msg)
            end if
            return
        end if

        select type (node => arena%entries(idx)%node)
        type is (call_or_subscript_node)
            array_real_call = is_elemental_real_call(arena, idx)
            if (.not. array_real_call) then
                if (allocated(node%name) .and. allocated(node%arg_indices)) then
                    array_real_call = size(node%arg_indices) == 1 .and. &
                        (same_name(node%name, 'aint') .or. &
                         same_name(node%name, 'anint'))
                end if
            end if
            if (array_real_call) then
                sym = whole_array_expr_shape_symbol(arena, idx, context)
                if (sym <= 0) then
                    error_msg = 'elemental array mask has no shape symbol'
                    return
                end if
                source_vk = context%symbols(sym)%value_kind
                call lower_array_elementwise_value(arena, idx, sym, linear_index, &
                                                   context, value, error_msg)
                if (len_trim(error_msg) > 0) return
                call coerce_reduction_operand(context, value, source_vk, &
                                              compare_vk, promoted, error_msg)
                value = promoted
                return
            end if
            if (allocated(node%name) .and. allocated(node%arg_indices) .and. &
                size(node%arg_indices) == 1 .and. .not. node%is_array_access) then
                if (same_name(node%name, 'abs')) then
                    if (is_complex_valued(arena, node%arg_indices(1), context, &
                            VALUE_C4)) then
                        complex_vk = VALUE_C4
                    else if (is_complex_valued(arena, node%arg_indices(1), context, &
                            VALUE_C8)) then
                        complex_vk = VALUE_C8
                    else
                        call lower_mask_expression_element(arena, node%arg_indices(1), &
                            compare_vk, linear_index, context, value, error_msg)
                        if (len_trim(error_msg) > 0) return
                        call mask_real_abs(context, compare_vk, value, error_msg)
                        return
                    end if
                    call mask_complex_array_element(arena, node%arg_indices(1), &
                        linear_index, context, complex_vk, re_value, im_value, &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                    call_args(1) = re_value
                    call_args(2) = im_value
                    if (complex_vk == VALUE_C4) then
                        if (.not. emit_liric_f32_call(context%session, 'hypotf', &
                                call_args, value, error_msg)) return
                        if (compare_vk == VALUE_F64) then
                            if (.not. emit_liric_f32_to_f64(context%session, value, &
                                    promoted, error_msg)) return
                            value = promoted
                        end if
                    else
                        if (.not. emit_liric_f64_call(context%session, 'hypot', &
                                call_args, value, error_msg)) return
                    end if
                    return
                else if (same_name(node%name, 'aimag')) then
                    if (is_complex_valued(arena, node%arg_indices(1), context, &
                            VALUE_C4)) then
                        complex_vk = VALUE_C4
                    else
                        complex_vk = VALUE_C8
                    end if
                    call mask_complex_array_element(arena, node%arg_indices(1), &
                        linear_index, context, complex_vk, re_value, im_value, &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                    value = im_value
                    if (complex_vk == VALUE_C4 .and. compare_vk == VALUE_F64) then
                        if (.not. emit_liric_f32_to_f64(context%session, value, &
                                promoted, error_msg)) return
                        value = promoted
                    end if
                    return
                end if
            end if
        type is (array_literal_node)
            if (allocated(node%syntax_style)) then
                if (node%syntax_style == 'implied_do') then
                    error_msg = 'nested implied-do array constructors are '// &
                        'not supported'
                    return
                end if
            end if
            if (.not. allocated(node%element_indices)) then
                error_msg = 'array constructor has no elements'
                return
            end if
            allocate (flat(0))
            call flatten_constructor_elements(arena, node%element_indices, flat, err)
            if (len_trim(err) > 0) then
                error_msg = err
                return
            end if
            if (linear_index < 0 .or. linear_index >= size(flat)) then
                error_msg = 'array constructor element index is out of bounds'
                return
            end if
            if (is_logical_literal(arena, flat(linear_index + 1))) then
                call get_literal_info(arena, flat(linear_index + 1), lit_value, &
                                      lit_type, error_msg)
                if (len_trim(error_msg) > 0) return
                value = i32_immediate(context%session, logical_i32_value(lit_value))
                call set_empty(error_msg)
            else
                call mask_scalar_element(arena, flat(linear_index + 1), context, &
                                         compare_vk, .true., value, error_msg)
            end if
            return
        end select

        if (is_literal(arena, idx) .and. compare_vk == VALUE_LOGICAL) then
            call lower_logical_expression(arena, idx, context, value, error_msg)
            return
        end if

        if (is_binary_op(arena, idx)) then
            call get_binary_op_info(arena, idx, op, left, right, line, column, err)
            if (len_trim(err) > 0) then
                error_msg = err
                return
            end if
            if (trim(adjustl(lowercase_text(op))) == '.not.' .or. &
                trim(adjustl(lowercase_text(op))) == 'not') then
                ! Unary .NOT. is represented as a binary AST node with an
                ! invalid left index.  Lower its real operand and materialize
                ! the inverse as a normalized i32 logical value.
                call lower_mask_expression_element(arena, right, compare_vk, &
                    linear_index, context, operand, error_msg)
                if (len_trim(error_msg) > 0) return
                zero = i32_immediate(context%session, 0_c_int64_t)
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, &
                        operand, zero, cond, error_msg)) return
                true_val = i32_immediate(context%session, 0_c_int64_t)
                false_val = i32_immediate(context%session, 1_c_int64_t)
                call select_value(context, cond, true_val, false_val, value, &
                                  error_msg)
                return
            end if
            logical_op = trim(adjustl(lowercase_text(op)))
            if (is_logical_connective(logical_op)) then
                ! Fold a constructor-only connective before emitting a nested
                ! logical instruction.  The result is then a regular i32
                ! operand when this expression is itself an array-mask element.
                block
                    integer, allocatable :: left_flat(:), right_flat(:)
                    integer :: left_value, right_value
                    character(len=:), allocatable :: left_text, right_text
                    character(len=:), allocatable :: left_type, right_type
                    character(len=:), allocatable :: constructor_error
                    logical :: left_constructor, right_constructor

                    left_constructor = .false.
                    right_constructor = .false.
                    if (node_exists(arena, left) .and. node_exists(arena, right)) then
                        select type (left_node => arena%entries(left)%node)
                        type is (array_literal_node)
                            left_constructor = .true.
                        end select
                        select type (right_node => arena%entries(right)%node)
                        type is (array_literal_node)
                            right_constructor = .true.
                        end select
                    end if
                    if (left_constructor .and. right_constructor) then
                        select type (left_node => arena%entries(left)%node)
                        type is (array_literal_node)
                            allocate (left_flat(0))
                            call flatten_constructor_elements(arena, &
                                left_node%element_indices, left_flat, constructor_error)
                        end select
                        if (len_trim(constructor_error) == 0) then
                            select type (right_node => arena%entries(right)%node)
                            type is (array_literal_node)
                                allocate (right_flat(0))
                                call flatten_constructor_elements(arena, &
                                    right_node%element_indices, right_flat, &
                                    constructor_error)
                            end select
                        end if
                        if (len_trim(constructor_error) == 0 .and. &
                            size(left_flat) == size(right_flat) .and. &
                            linear_index >= 0 .and. &
                            linear_index < size(left_flat)) then
                            if (is_logical_literal(arena, left_flat(linear_index + 1)) &
                                .and. is_logical_literal(arena, right_flat(linear_index + 1))) then
                                call get_literal_info(arena, left_flat(linear_index + 1), &
                                    left_text, left_type, error_msg)
                                if (len_trim(error_msg) > 0) return
                                call get_literal_info(arena, right_flat(linear_index + 1), &
                                    right_text, right_type, error_msg)
                                if (len_trim(error_msg) > 0) return
                                left_value = logical_i32_value(left_text)
                                right_value = logical_i32_value(right_text)
                                select case (logical_op)
                                case ('.and.', 'and')
                                    left_value = merge(1, 0, left_value /= 0 .and. &
                                        right_value /= 0)
                                case ('.or.', 'or')
                                    left_value = merge(1, 0, left_value /= 0 .or. &
                                        right_value /= 0)
                                case ('.eqv.', 'eqv')
                                    left_value = merge(1, 0, (left_value /= 0) .eqv. &
                                        (right_value /= 0))
                                case default
                                    left_value = merge(1, 0, (left_value /= 0) .neqv. &
                                        (right_value /= 0))
                                end select
                                value = i32_immediate(context%session, &
                                    int(left_value, c_int64_t))
                                call set_empty(error_msg)
                                return
                            end if
                        end if
                    end if
                end block
                call lower_mask_expression_element(arena, left, VALUE_LOGICAL, &
                    linear_index, context, lhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_mask_expression_element(arena, right, VALUE_LOGICAL, &
                    linear_index, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                select case (logical_op)
                case ('.and.', 'and')
                    if (.not. emit_i32_binary(context%session, LR_OP_AND, lhs, &
                            rhs, value, error_msg)) return
                case ('.or.', 'or')
                    if (.not. emit_i32_binary(context%session, LR_OP_OR, lhs, &
                            rhs, value, error_msg)) return
                case ('.eqv.', 'eqv', '.neqv.', 'neqv')
                    if (.not. emit_i32_binary(context%session, LR_OP_XOR, lhs, &
                            rhs, value, error_msg)) return
                    if (logical_op == '.eqv.' .or. logical_op == 'eqv') then
                        if (.not. emit_i32_binary(context%session, LR_OP_XOR, value, &
                                i32_immediate(context%session, 1_c_int64_t), &
                                value, error_msg)) return
                    end if
                end select
                return
            end if
            if (compare_vk == VALUE_LOGICAL) then
                call lower_logical_expression(arena, idx, context, value, error_msg)
                return
            end if
            call lower_mask_expression_element(arena, left, compare_vk, linear_index, &
                context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_mask_expression_element(arena, right, compare_vk, linear_index, &
                context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(op) == '**') then
                call_args(1) = lhs
                call_args(2) = rhs
                if (compare_vk == VALUE_F32) then
                    if (.not. emit_liric_f32_call(context%session, 'powf', &
                            call_args, value, error_msg)) return
                else
                    if (.not. emit_liric_f64_call(context%session, 'pow', &
                            call_args, value, error_msg)) return
                end if
                return
            end if
            call real_opcode(op, line, column, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (compare_vk == VALUE_F32) then
                if (.not. emit_liric_f32_binary(context%session, opcode, lhs, rhs, &
                        value, error_msg)) return
            else
                if (.not. emit_liric_f64_binary(context%session, opcode, lhs, rhs, &
                        value, error_msg)) return
            end if
            return
        end if
        if (compare_vk == VALUE_LOGICAL) then
            call lower_logical_expression(arena, idx, context, value, error_msg)
            return
        end if
        error_msg = 'array mask expression is not supported'
    end subroutine lower_mask_expression_element

    subroutine mask_complex_array_element(arena, idx, linear_index, context, vk, &
                                          re_value, im_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: idx, linear_index, vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: re_value, im_value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: sym

        if (.not. is_identifier(arena, idx)) then
            error_msg = 'complex array mask operand is not an identifier'
            return
        end if
        call get_identifier_name(arena, idx, name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol_compat(context, name)
        if (sym <= 0 .or. .not. context%symbols(sym)%is_array .and. &
            .not. context%symbols(sym)%is_allocatable) then
            error_msg = 'complex array mask operand was not declared: '//trim(name)
            return
        end if
        call load_complex_array_linear_element(context, sym, linear_index, vk, &
                                               re_value, im_value, error_msg)
    end subroutine mask_complex_array_element

    subroutine mask_real_abs(context, value_kind, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(inout) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: zero, nonnegative, negative, absolute

        call set_empty(error_msg)
        if (value_kind == VALUE_F64) then
            zero = liric_f64_immediate(context%session, 0.0_c_double)
            if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGE, value, &
                    zero, nonnegative, error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FSUB, zero, &
                    value, negative, error_msg)) return
        else
            zero = liric_f32_immediate(context%session, 0.0_c_float)
            if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGE, value, &
                    zero, nonnegative, error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FSUB, zero, &
                    value, negative, error_msg)) return
        end if
        call select_value(context, nonnegative, value, negative, absolute, &
                          error_msg)
        if (len_trim(error_msg) == 0) value = absolute
    end subroutine mask_real_abs

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

    subroutine try_alloc_component_constructor_mask_reduction(arena, node, context, is_all, &
                                                  handled, value, error_msg)
        ! Handle any()/all() of an integer/real allocatable component against
        ! an explicit constructor. The component extent is runtime-owned, but
        ! the constructor gives this conformance path a fixed, independently
        ! checkable number of elements to compare.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_all
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: op
        integer :: arg_index, left_idx, right_idx, line, col
        integer :: component_idx, constructor_idx, comp_kind, i, n
        integer, allocatable :: flat(:)
        logical :: component_on_left
        type(lr_operand_desc_t) :: desc_addr, component_value, constructor_value
        type(lr_operand_desc_t) :: condition, accumulator, next_acc
        integer(c_int) :: predicate

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices) .or. size(node%arg_indices) /= 1) return
        arg_index = node%arg_indices(1)
        if (.not. is_binary_op(arena, arg_index)) return
        call get_binary_op_info(arena, arg_index, op, left_idx, right_idx, line, col, &
                                error_msg)
        if (len_trim(error_msg) > 0 .or. .not. is_relational_operator(op)) then
            call set_empty(error_msg)
            return
        end if

        component_idx = 0
        constructor_idx = 0
        component_on_left = .false.
        if (node_exists(arena, left_idx)) then
            select type (left => arena%entries(left_idx)%node)
            type is (component_access_node)
                if (component_access_is_alloc_array(arena, left, context)) then
                    component_idx = left_idx
                    constructor_idx = right_idx
                    component_on_left = .true.
                    comp_kind = derived_component_access_kind(arena, left, context)
                end if
            class default
            end select
        end if
        if (component_idx == 0 .and. node_exists(arena, right_idx)) then
            select type (right => arena%entries(right_idx)%node)
            type is (component_access_node)
                if (component_access_is_alloc_array(arena, right, context)) then
                    component_idx = right_idx
                    constructor_idx = left_idx
                    component_on_left = .false.
                    comp_kind = derived_component_access_kind(arena, right, context)
                end if
            class default
            end select
        end if
        if (component_idx == 0 .or. .not. node_exists(arena, constructor_idx)) return
        select type (constructor => arena%entries(constructor_idx)%node)
        type is (array_literal_node)
            if (is_implied_do_constructor(arena, constructor) .or. &
                .not. allocated(constructor%element_indices)) return
            allocate (flat(0))
            call flatten_constructor_elements(arena, constructor%element_indices, flat, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            return
        end select
        n = size(flat)
        if (n <= 0) return

        call alloc_array_component_descriptor(arena, component_idx, context, desc_addr, &
                                              error_msg)
        if (len_trim(error_msg) > 0) return
        if (comp_kind /= VALUE_I32 .and. comp_kind /= VALUE_F32 .and. &
            comp_kind /= VALUE_F64) then
            return
        end if
        if (comp_kind == VALUE_F32 .or. comp_kind == VALUE_F64) then
            call real_compare_predicate(op, predicate, error_msg)
        else
            call integer_compare_predicate(op, predicate, error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        if (is_all) then
            accumulator = i32_immediate(context%session, 1_c_int64_t)
        else
            accumulator = i32_immediate(context%session, 0_c_int64_t)
        end if
        do i = 0, n - 1
            call load_alloc_component_element(context, desc_addr, comp_kind, &
                i32_immediate(context%session, int(i, c_int64_t)), component_value, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_component_rhs_value(arena, flat(i + 1), comp_kind, context, &
                                           constructor_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (component_on_left) then
                if (comp_kind == VALUE_F32) then
                    if (.not. emit_liric_f32_fcmp(context%session, predicate, &
                            component_value, constructor_value, condition, error_msg)) return
                else if (comp_kind == VALUE_F64) then
                    if (.not. emit_liric_f64_fcmp(context%session, predicate, &
                            component_value, constructor_value, condition, error_msg)) return
                else
                    if (.not. emit_liric_i32_icmp(context%session, predicate, &
                            component_value, constructor_value, condition, error_msg)) return
                end if
            else
                if (comp_kind == VALUE_F32) then
                    if (.not. emit_liric_f32_fcmp(context%session, predicate, &
                            constructor_value, component_value, condition, error_msg)) return
                else if (comp_kind == VALUE_F64) then
                    if (.not. emit_liric_f64_fcmp(context%session, predicate, &
                            constructor_value, component_value, condition, error_msg)) return
                else
                    if (.not. emit_liric_i32_icmp(context%session, predicate, &
                            constructor_value, component_value, condition, error_msg)) return
                end if
            end if
            if (is_all) then
                if (.not. emit_i32_binary(context%session, LR_OP_AND, accumulator, &
                        condition, next_acc, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_OR, accumulator, &
                        condition, next_acc, error_msg)) return
            end if
            accumulator = next_acc
        end do
        value = accumulator
        handled = .true.
        call set_empty(error_msg)
    end subroutine try_alloc_component_constructor_mask_reduction

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
        call try_alloc_component_constructor_mask_reduction(arena, node, context, is_all, &
                                                  handled, value, error_msg)
        if (handled .or. len_trim(error_msg) > 0) return
        call try_derived_array_component_mask_reduction(arena, node, context, is_all, &
                                                        handled, value, error_msg)
        if (handled .or. len_trim(error_msg) > 0) return
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

        if (compare_vk == VALUE_C4 .or. compare_vk == VALUE_C8) then
            call try_complex_array_mask_reduction(arena, context, op, lsym, &
                rsym, lflat, rflat, lscalar, rscalar, extent, is_all, handled, &
                value, error_msg)
            return
        end if

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

    subroutine try_complex_array_mask_reduction(arena, context, op, lsym, &
                                                rsym, lflat, rflat, lscalar, &
                                                rscalar, extent, is_all, &
                                                handled, value, error_msg)
        ! ANY/ALL over complex equality is defined componentwise: equality is
        ! (re==re .and. im==im), while inequality is the negation. Fortran does
        ! not define ordering comparisons for complex values.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: op
        integer, intent(in) :: lsym, rsym, lscalar, rscalar, extent
        integer, allocatable, intent(in) :: lflat(:), rflat(:)
        logical, intent(in) :: is_all
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lre, lim, rre, rim, re_cond, im_cond
        type(lr_operand_desc_t) :: condition, accumulator, next_acc
        integer(c_int) :: predicate
        integer :: i
        logical :: is_equal

        handled = .false.
        call set_empty(error_msg)
        is_equal = trim(op) == '==' .or. trim(op) == '=' .or. &
                   trim(op) == '.eq.'
        if (.not. is_equal .and. trim(op) /= '/=' .and. trim(op) /= '!=' .and. &
            trim(op) /= '.ne.') return
        if (is_equal) then
            call real_compare_predicate('==', predicate, error_msg)
        else
            call real_compare_predicate('/=', predicate, error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        if (is_all) then
            accumulator = i32_immediate(context%session, 1_c_int64_t)
        else
            accumulator = i32_immediate(context%session, 0_c_int64_t)
        end if
        do i = 0, extent - 1
            call reduction_complex_side_element(arena, context, lsym, lflat, &
                lscalar, i, context_complex_compare_kind(context, lsym, rsym), &
                lre, lim, error_msg)
            if (len_trim(error_msg) > 0) return
            call reduction_complex_side_element(arena, context, rsym, rflat, &
                rscalar, i, context_complex_compare_kind(context, lsym, rsym), &
                rre, rim, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context_complex_compare_kind(context, lsym, rsym) == VALUE_C4) then
                if (.not. emit_liric_f32_fcmp(context%session, predicate, lre, rre, &
                                              re_cond, error_msg)) return
                if (.not. emit_liric_f32_fcmp(context%session, predicate, lim, rim, &
                                              im_cond, error_msg)) return
            else
                if (.not. emit_liric_f64_fcmp(context%session, predicate, lre, rre, &
                                              re_cond, error_msg)) return
                if (.not. emit_liric_f64_fcmp(context%session, predicate, lim, rim, &
                                              im_cond, error_msg)) return
            end if
            if (is_equal) then
                if (.not. emit_i32_binary(context%session, LR_OP_AND, re_cond, &
                                          im_cond, condition, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_OR, re_cond, &
                                          im_cond, condition, error_msg)) return
            end if
            if (is_all) then
                if (.not. emit_i32_binary(context%session, LR_OP_AND, accumulator, &
                                          condition, next_acc, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_OR, accumulator, &
                                          condition, next_acc, error_msg)) return
            end if
            accumulator = next_acc
        end do
        value = accumulator
        handled = .true.
    end subroutine try_complex_array_mask_reduction

    integer function context_complex_compare_kind(context, lsym, rsym) result(vk)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: lsym, rsym

        vk = VALUE_C4
        if ((lsym > 0 .and. context%symbols(lsym)%value_kind == VALUE_C8) .or. &
            (rsym > 0 .and. context%symbols(rsym)%value_kind == VALUE_C8)) then
            vk = VALUE_C8
        end if
    end function context_complex_compare_kind

    subroutine reduction_complex_side_element(arena, context, sym, flat, &
                                              scalar_idx, i, compare_vk, re, im, &
                                              error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym, scalar_idx, i, compare_vk
        integer, allocatable, intent(in) :: flat(:)
        type(lr_operand_desc_t), intent(out) :: re, im
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_addr, im_addr, narrow_re, narrow_im
        integer :: expr_idx, native_vk

        call set_empty(error_msg)
        if (sym > 0) then
            native_vk = context%symbols(sym)%value_kind
            call complex_array_linear_addresses(context, sym, i, native_vk, &
                                                re_addr, im_addr, error_msg)
            if (len_trim(error_msg) > 0) return
            if (native_vk == VALUE_C4) then
                if (.not. emit_liric_f32_load(context%session, re_addr, narrow_re, &
                                              error_msg)) return
                if (.not. emit_liric_f32_load(context%session, im_addr, narrow_im, &
                                              error_msg)) return
                if (compare_vk == VALUE_C8) then
                    if (.not. emit_liric_f32_to_f64(context%session, narrow_re, re, &
                                                   error_msg)) return
                    if (.not. emit_liric_f32_to_f64(context%session, narrow_im, im, &
                                                   error_msg)) return
                else
                    re = narrow_re
                    im = narrow_im
                end if
            else
                if (.not. emit_liric_f64_load(context%session, re_addr, re, &
                                              error_msg)) return
                if (.not. emit_liric_f64_load(context%session, im_addr, im, &
                                              error_msg)) return
            end if
            return
        end if
        if (allocated(flat)) then
            expr_idx = flat(i + 1)
        else
            expr_idx = scalar_idx
        end if
        if (compare_vk == VALUE_C4) then
            call eval_c4_rhs(arena, expr_idx, context, re, im, error_msg)
        else
            call eval_c8_rhs(arena, expr_idx, context, re, im, error_msg)
        end if
    end subroutine reduction_complex_side_element

    subroutine try_derived_array_component_mask_reduction(arena, node, context, &
            is_all, handled, value, error_msg)
        ! Handle any()/all() of a relational comparison involving a scalar
        ! component projected across a fixed-size derived array, for example
        ! any(items%i /= [1, 2, 3]) or any(items%x /= source).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_all
        logical, intent(out) :: handled
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: op, resolve_error, name
        integer :: arg_index, left_index, right_index, line, column
        integer :: component_index, component_type, component_symbol
        integer :: other_index, other_symbol, other_kind
        integer :: component_kind_value, extent, i
        integer, allocatable :: flat(:)
        logical :: component_on_left
        type(lr_operand_desc_t) :: component_value, other_value, condition
        type(lr_operand_desc_t) :: accumulator, next_acc
        integer(c_int) :: predicate

        handled = .false.
        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        arg_index = node%arg_indices(1)
        if (.not. is_binary_op(arena, arg_index)) return
        call get_binary_op_info(arena, arg_index, op, left_index, right_index, &
                                line, column, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        if (.not. is_relational_operator(op)) return

        component_index = 0
        component_type = 0
        component_symbol = 0
        component_on_left = .false.
        if (node_exists(arena, left_index)) then
            select type (left => arena%entries(left_index)%node)
            type is (component_access_node)
                call resolve_derived_array_component(arena, left_index, context, &
                    component_symbol, component_type, component_index, &
                    resolve_error)
                if (len_trim(resolve_error) == 0) component_on_left = .true.
            class default
            end select
        end if
        if (component_index == 0 .and. node_exists(arena, right_index)) then
            select type (right => arena%entries(right_index)%node)
            type is (component_access_node)
                call resolve_derived_array_component(arena, right_index, context, &
                    component_symbol, component_type, component_index, &
                    resolve_error)
                if (len_trim(resolve_error) == 0) component_on_left = .false.
            class default
            end select
        end if
        if (component_index == 0) return

        component_kind_value = component_kind(context, component_type, &
                                               component_index)
        if (component_kind_value /= VALUE_I32 .and. &
            component_kind_value /= VALUE_F32 .and. &
            component_kind_value /= VALUE_F64) return
        if (component_on_left) then
            other_index = right_index
        else
            other_index = left_index
        end if

        other_symbol = 0
        other_kind = component_kind_value
        if (node_exists(arena, other_index)) then
            select type (other => arena%entries(other_index)%node)
            type is (array_literal_node)
                if (is_implied_do_constructor(arena, other) .or. &
                    .not. allocated(other%element_indices)) return
                allocate (flat(0))
                call flatten_constructor_elements(arena, other%element_indices, flat, &
                                                  error_msg)
                if (len_trim(error_msg) > 0) return
                extent = size(flat)
            class default
                if (.not. is_identifier(arena, other_index)) return
                call get_identifier_name(arena, other_index, name, error_msg)
                if (len_trim(error_msg) > 0) return
                other_symbol = find_symbol_compat(context, name)
                if (other_symbol <= 0) return
                if (.not. is_elementwise_array_operand(context, other_symbol)) return
                if (context%symbols(other_symbol)%value_kind /= &
                    component_kind_value) return
                extent = array_symbol_extent(context, other_symbol)
            end select
        else
            return
        end if
        if (extent /= derived_array_component_extent(context, component_symbol)) return

        if (component_kind_value == VALUE_F32 .or. &
            component_kind_value == VALUE_F64) then
            call real_compare_predicate(op, predicate, error_msg)
        else
            call integer_compare_predicate(op, predicate, error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        if (is_all) then
            accumulator = i32_immediate(context%session, 1_c_int64_t)
        else
            accumulator = i32_immediate(context%session, 0_c_int64_t)
        end if
        do i = 0, extent - 1
            call load_derived_array_component_linear(context, component_symbol, &
                component_type, component_index, i, component_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (allocated(flat)) then
                call lower_component_rhs_value(arena, flat(i + 1), &
                    component_kind_value, context, other_value, error_msg)
            else
                call load_array_linear_element(context, other_symbol, &
                    int(i, c_int64_t), other_value, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            if (component_on_left) then
                call emit_reduction_comparison(context, component_kind_value, &
                    predicate, component_value, other_value, condition, error_msg)
            else
                call emit_reduction_comparison(context, component_kind_value, &
                    predicate, other_value, component_value, condition, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            if (is_all) then
                if (.not. emit_i32_binary(context%session, LR_OP_AND, accumulator, &
                        condition, next_acc, error_msg)) return
            else
                if (.not. emit_i32_binary(context%session, LR_OP_OR, accumulator, &
                        condition, next_acc, error_msg)) return
            end if
            accumulator = next_acc
        end do
        value = accumulator
        handled = .true.
        call set_empty(error_msg)
    end subroutine try_derived_array_component_mask_reduction

    subroutine emit_reduction_comparison(context, value_kind, predicate, lhs, rhs, &
                                         result, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        integer(c_int), intent(in) :: predicate
        type(lr_operand_desc_t), intent(in) :: lhs, rhs
        type(lr_operand_desc_t), intent(out) :: result
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: ok

        select case (value_kind)
        case (VALUE_F32)
            ok = emit_liric_f32_fcmp(context%session, predicate, lhs, rhs, result, &
                                     error_msg)
        case (VALUE_F64)
            ok = emit_liric_f64_fcmp(context%session, predicate, lhs, rhs, result, &
                                     error_msg)
        case default
            ok = emit_liric_i32_icmp(context%session, predicate, lhs, rhs, result, &
                                     error_msg)
        end select
        if (ok) call set_empty(error_msg)
    end subroutine emit_reduction_comparison

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
                s = find_symbol_compat(context, name)
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

    ! Comparison kind: complex(8) if any array operand is c8, then c4, real(8),
    ! real(4), or integer. ok=.false. for unsupported array kinds.
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
        if (array_kind_is(context, lsym, VALUE_C4) .or. &
            array_kind_is(context, rsym, VALUE_C4)) compare_vk = VALUE_C4
        if (array_kind_is(context, lsym, VALUE_C8) .or. &
            array_kind_is(context, rsym, VALUE_C8)) compare_vk = VALUE_C8
    end subroutine reduction_compare_kind

    logical function supported_reduction_kind(vk) result(ok)
        integer, intent(in) :: vk

        ok = vk == VALUE_I32 .or. vk == VALUE_F32 .or. vk == VALUE_F64 .or. &
             vk == VALUE_C4 .or. vk == VALUE_C8
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
