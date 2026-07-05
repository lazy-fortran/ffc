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
        integer :: arg_index, left_idx, right_idx, line, col
        integer :: lsym, rsym, lscalar, rscalar, extent, i, compare_vk
        integer, allocatable :: lflat(:), rflat(:)
        logical :: lok, rok
        character(len=:), allocatable :: op
        type(lr_operand_desc_t) :: lhs, rhs, cond, acc, next_acc
        integer(c_int) :: predicate

        handled = .false.
        call set_empty(error_msg)
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
