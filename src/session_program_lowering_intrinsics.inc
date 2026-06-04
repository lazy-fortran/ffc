    subroutine lower_i32_intrinsic_call(arena, node, intrinsic_id, context, &
                                        value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: intrinsic_id
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (intrinsic_id)
        case (I32_INTRINSIC_ABS)
            call lower_i32_abs_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_MIN)
            call lower_i32_minmax_intrinsic(arena, node, LR_CMP_SLE, context, &
                                            value, error_msg)
        case (I32_INTRINSIC_MAX)
            call lower_i32_minmax_intrinsic(arena, node, LR_CMP_SGE, context, &
                                            value, error_msg)
        case (I32_INTRINSIC_MOD)
            call lower_i32_mod_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_IAND)
            call lower_i32_bitwise_intrinsic(arena, node, LR_OP_AND, context, &
                                             value, error_msg)
        case (I32_INTRINSIC_IOR)
            call lower_i32_bitwise_intrinsic(arena, node, LR_OP_OR, context, &
                                             value, error_msg)
        case (I32_INTRINSIC_IEOR)
            call lower_i32_bitwise_intrinsic(arena, node, LR_OP_XOR, context, &
                                             value, error_msg)
        case (I32_INTRINSIC_NOT)
            call lower_i32_not_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_ISHFT)
            call lower_i32_ishft_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_ISHFTC)
            call lower_i32_ishftc_intrinsic(arena, node, context, value, &
                                            error_msg)
        case (I32_INTRINSIC_SIGN)
            call lower_i32_sign_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_INT)
            call lower_i32_from_f64_intrinsic(arena, node, '', context, value, &
                                              error_msg)
        case (I32_INTRINSIC_NINT)
            call lower_i32_from_f64_intrinsic(arena, node, 'round', context, &
                                              value, error_msg)
        case (I32_INTRINSIC_FLOOR)
            call lower_i32_from_f64_intrinsic(arena, node, 'floor', context, &
                                              value, error_msg)
        case (I32_INTRINSIC_CEILING)
            call lower_i32_from_f64_intrinsic(arena, node, 'ceil', context, &
                                              value, error_msg)
        case (I32_INTRINSIC_DOT_PRODUCT)
            call lower_i32_dot_product_intrinsic(arena, node, context, value, &
                                                 error_msg)
        case default
            call unsupported_intrinsic_error(node, error_msg)
        end select
    end subroutine lower_i32_intrinsic_call

    subroutine lower_i32_bitwise_intrinsic(arena, node, opcode, context, value, &
                                           error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer(c_int), intent(in) :: opcode
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return

        call lower_i32_expression(arena, node%arg_indices(1), context, lhs, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, rhs, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, value, &
                                  error_msg)) return
    end subroutine lower_i32_bitwise_intrinsic

    subroutine lower_i32_not_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: all_ones

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return

        call lower_i32_expression(arena, node%arg_indices(1), context, arg, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        ! not(x) == x xor -1 (bitwise complement)
        all_ones = i32_immediate(context%session, -1_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_XOR, arg, all_ones, &
                                  value, error_msg)) return
    end subroutine lower_i32_not_intrinsic

    subroutine lower_i32_ishft_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: shift_amount
        integer(c_int64_t) :: shift_count

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call literal_shift_count(arena, node%arg_indices(2), shift_count, &
                                 'ishft', error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_i32_expression(arena, node%arg_indices(1), context, arg, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (shift_count == 0_c_int64_t) then
            value = arg
            call set_empty(error_msg)
            return
        end if
        ! Positive shift -> left (shl); negative -> logical right (lshr).
        if (shift_count > 0_c_int64_t) then
            shift_amount = i32_immediate(context%session, shift_count)
            if (.not. emit_i32_binary(context%session, LR_OP_SHL, arg, &
                                      shift_amount, value, error_msg)) return
        else
            shift_amount = i32_immediate(context%session, -shift_count)
            if (.not. emit_i32_binary(context%session, LR_OP_LSHR, arg, &
                                      shift_amount, value, error_msg)) return
        end if
    end subroutine lower_i32_ishft_intrinsic

    subroutine lower_i32_ishftc_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: left_part
        type(lr_operand_desc_t) :: right_part
        integer(c_int64_t) :: shift_count
        integer(c_int64_t) :: rotate

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call literal_shift_count(arena, node%arg_indices(2), shift_count, &
                                 'ishftc', error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_i32_expression(arena, node%arg_indices(1), context, arg, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        ! Default 32-bit circular shift: normalise the count into [0, 31].
        rotate = modulo(shift_count, 32_c_int64_t)
        if (rotate == 0_c_int64_t) then
            value = arg
            call set_empty(error_msg)
            return
        end if
        ! ishftc(i, n) = (i << n) | (i >> (32 - n)) using a logical right shift.
        if (.not. emit_i32_binary(context%session, LR_OP_SHL, arg, &
                                  i32_immediate(context%session, rotate), &
                                  left_part, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_LSHR, arg, &
                                  i32_immediate(context%session, 32_c_int64_t - &
                                                rotate), &
                                  right_part, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_OR, left_part, &
                                  right_part, value, error_msg)) return
    end subroutine lower_i32_ishftc_intrinsic

    subroutine literal_shift_count(arena, shift_index, shift_count, name, &
                                   error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: shift_index
        integer(c_int64_t), intent(out) :: shift_count
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: parse_error

        call parse_i32_constant(arena, shift_index, shift_count, name, &
                                parse_error)
        if (len_trim(parse_error) > 0) then
            call unsupported_feature_error(name//' shift count', &
                                       get_node_line(arena, shift_index), &
                                       get_node_column(arena, shift_index), &
                                       'direct LIRIC session only supports a '// &
                                       'literal integer shift count', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine literal_shift_count

    subroutine lower_f64_intrinsic_call(arena, node, intrinsic_id, context, &
                                        value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: intrinsic_id
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (intrinsic_id)
        case (F64_INTRINSIC_ABS)
            call lower_f64_abs_intrinsic(arena, node, context, value, error_msg)
        case (F64_INTRINSIC_MIN)
            call lower_f64_minmax_intrinsic(arena, node, LR_FCMP_OLE, &
                                            context, value, error_msg)
        case (F64_INTRINSIC_MAX)
            call lower_f64_minmax_intrinsic(arena, node, LR_FCMP_OGE, &
                                            context, value, error_msg)
        case (F64_INTRINSIC_REAL)
            call lower_f64_real_intrinsic(arena, node, context, value, &
                                          error_msg)
        case (F64_INTRINSIC_SIGN)
            call lower_f64_sign_intrinsic(arena, node, context, value, error_msg)
        case (F64_INTRINSIC_SQRT)
            call lower_f64_libm_call(arena, node, 'sqrt', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_EXP)
            call lower_f64_libm_call(arena, node, 'exp', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_LOG)
            call lower_f64_libm_call(arena, node, 'log', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_SIN)
            call lower_f64_libm_call(arena, node, 'sin', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_COS)
            call lower_f64_libm_call(arena, node, 'cos', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_TAN)
            call lower_f64_libm_call(arena, node, 'tan', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ATAN)
            call lower_f64_libm_call(arena, node, 'atan', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ATAN2)
            call lower_f64_libm_call(arena, node, 'atan2', 2, context, value, &
                                     error_msg)
        case default
            call unsupported_intrinsic_error(node, error_msg)
        end select
    end subroutine lower_f64_intrinsic_call

    subroutine lower_i32_abs_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: negative
        type(lr_operand_desc_t) :: zero

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return

        call lower_i32_expression(arena, node%arg_indices(1), context, arg, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGE, arg, zero, &
                                      condition, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, zero, arg, &
                                                  negative, error_msg)) return
        call select_value(context, condition, arg, negative, value, error_msg)
    end subroutine lower_i32_abs_intrinsic

    subroutine lower_i32_sign_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, b, zero
        type(lr_operand_desc_t) :: a_nonneg, neg_a, magnitude
        type(lr_operand_desc_t) :: b_nonneg, neg_magnitude

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, a, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, b, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        zero = i32_immediate(context%session, 0_c_int64_t)
        ! magnitude = abs(a)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGE, a, zero, &
                                      a_nonneg, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, zero, a, neg_a, &
                                  error_msg)) return
        call select_value(context, a_nonneg, a, neg_a, magnitude, error_msg)
        if (len_trim(error_msg) > 0) return
        ! result = (b >= 0) ? magnitude : -magnitude  (sign(7,0) == 7)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGE, b, zero, &
                                      b_nonneg, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, zero, magnitude, &
                                  neg_magnitude, error_msg)) return
        call select_value(context, b_nonneg, magnitude, neg_magnitude, value, &
                          error_msg)
    end subroutine lower_i32_sign_intrinsic

    subroutine lower_i32_from_f64_intrinsic(arena, node, libm_name, context, &
                                            value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: libm_name
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: rounded

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        ! int(x) truncates toward zero, which fptosi already does directly.
        ! nint/floor/ceiling apply the libm rounding first, then convert.
        if (len_trim(libm_name) > 0) then
            if (.not. emit_liric_f64_call(context%session, libm_name, [arg], &
                                          rounded, error_msg)) return
        else
            rounded = arg
        end if
        if (.not. emit_liric_f64_to_i32(context%session, rounded, value, &
                                        error_msg)) return
    end subroutine lower_i32_from_f64_intrinsic

    subroutine lower_f64_abs_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: negative
        type(lr_operand_desc_t) :: zero

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return

        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        zero = liric_f64_immediate(context%session, 0.0_c_double)
        if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGE, arg, zero, &
                                      condition, error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, LR_OP_FSUB, zero, &
                                        arg, negative, error_msg)) return
        call select_value(context, condition, arg, negative, value, error_msg)
    end subroutine lower_f64_abs_intrinsic

    subroutine lower_f64_sign_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, b, zero
        type(lr_operand_desc_t) :: a_nonneg, neg_a, magnitude
        type(lr_operand_desc_t) :: b_nonneg, neg_magnitude

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, a, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(2), context, b, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        zero = liric_f64_immediate(context%session, 0.0_c_double)
        ! magnitude = abs(a)
        if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGE, a, zero, &
                                      a_nonneg, error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, LR_OP_FSUB, zero, a, &
                                        neg_a, error_msg)) return
        call select_value(context, a_nonneg, a, neg_a, magnitude, error_msg)
        if (len_trim(error_msg) > 0) return
        ! result = (b >= 0) ? magnitude : -magnitude
        if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGE, b, zero, &
                                      b_nonneg, error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, LR_OP_FSUB, zero, &
                                        magnitude, neg_magnitude, error_msg)) &
            return
        call select_value(context, b_nonneg, magnitude, neg_magnitude, value, &
                          error_msg)
    end subroutine lower_f64_sign_intrinsic

    subroutine lower_i32_mod_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs
        type(lr_operand_desc_t) :: rhs

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return

        call lower_i32_expression(arena, node%arg_indices(1), context, lhs, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, rhs, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_binary(context%session, LR_OP_SREM, lhs, rhs, value, &
                                                  error_msg)) return
    end subroutine lower_i32_mod_intrinsic

    subroutine lower_i32_minmax_intrinsic(arena, node, predicate, context, &
                                          value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer(c_int), intent(in) :: predicate
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: candidate
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: selected
        integer :: i

    if (.not. require_intrinsic_arg_count(node, 2, 64, &
                                               error_msg)) return

        call lower_i32_expression(arena, node%arg_indices(1), context, value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 2, size(node%arg_indices)
            call lower_i32_expression(arena, node%arg_indices(i), context, &
                                      candidate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, predicate, value, &
                                          candidate, condition, error_msg)) return
            call select_value(context, condition, value, candidate, selected, &
                              error_msg)
            if (len_trim(error_msg) > 0) return
            value = selected
        end do
    end subroutine lower_i32_minmax_intrinsic

    subroutine lower_f64_minmax_intrinsic(arena, node, predicate, context, &
                                          value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer(c_int), intent(in) :: predicate
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: candidate
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: selected
        integer :: i

   if (.not. require_intrinsic_arg_count(node, 2, 64, &
                                               error_msg)) return

        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, &
                                     value, error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 2, size(node%arg_indices)
            call lower_f64_intrinsic_arg(arena, node%arg_indices(i), context, &
                                         candidate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_fcmp(context%session, predicate, value, &
                                          candidate, condition, error_msg)) return
            call select_value(context, condition, value, candidate, selected, &
                              error_msg)
            if (len_trim(error_msg) > 0) return
            value = selected
        end do
    end subroutine lower_f64_minmax_intrinsic

    subroutine lower_f64_real_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return

        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, value, &
                                     error_msg)
    end subroutine lower_f64_real_intrinsic

    subroutine lower_f64_libm_call(arena, node, libm_name, arg_count, context, &
                                   value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: libm_name
        integer, intent(in) :: arg_count
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: args(arg_count)
        integer :: i

        if (.not. require_intrinsic_arg_count(node, arg_count, arg_count, &
                                              error_msg)) return
        do i = 1, arg_count
            call lower_f64_intrinsic_arg(arena, node%arg_indices(i), context, &
                                         args(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (.not. emit_liric_f64_call(context%session, libm_name, args, value, &
                                      error_msg)) return
    end subroutine lower_f64_libm_call

    subroutine lower_f64_intrinsic_arg(arena, node_index, context, value, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: integer_value

        if (is_f64_expression(arena, node_index, context)) then
            call lower_f64_expression(arena, node_index, context, value, &
                                      error_msg)
            return
        end if

        call lower_i32_expression(arena, node_index, context, integer_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_f64(context%session, integer_value, value, &
                                        error_msg)) return
    end subroutine lower_f64_intrinsic_arg

    subroutine select_value(context, condition, true_value, false_value, value, &
                            error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: condition
        type(lr_operand_desc_t), intent(in) :: true_value
        type(lr_operand_desc_t), intent(in) :: false_value
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: true_block
        integer(c_int32_t) :: false_block
        integer(c_int32_t) :: merge_block

        true_block = create_liric_block(context%session)
        false_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)

        if (.not. emit_liric_condbr(context%session, condition, true_block, &
                                    false_block, error_msg)) return

        if (.not. set_liric_block(context%session, true_block, error_msg)) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, false_block, error_msg)) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        if (.not. emit_liric_phi(context%session, true_value, true_block, &
                                 false_value, false_block, value, &
                                 error_msg)) return
    end subroutine select_value

    logical function require_intrinsic_arg_count(node, min_count, max_count, &
                                                 error_msg)
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: min_count
        integer, intent(in) :: max_count
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: arg_count

        require_intrinsic_arg_count = .false.
        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
        if (arg_count < min_count .or. arg_count > max_count) then
            error_msg = 'invalid argument count for scalar intrinsic: '// &
                        trim(node%name)
            return
        end if

        call set_empty(error_msg)
        require_intrinsic_arg_count = .true.
    end function require_intrinsic_arg_count

    integer function i32_intrinsic_id(name) result(intrinsic_id)
        character(len=*), intent(in) :: name
        integer :: i

        intrinsic_id = I32_INTRINSIC_NONE
        do i = 1, size(I32_INTRINSIC_NAMES)
            if (same_name(name, I32_INTRINSIC_NAMES(i))) then
                intrinsic_id = I32_INTRINSIC_IDS(i)
                return
            end if
        end do
    end function i32_intrinsic_id

    integer function f64_intrinsic_id(name) result(intrinsic_id)
        character(len=*), intent(in) :: name
        integer :: i

        intrinsic_id = F64_INTRINSIC_NONE
        do i = 1, size(F64_INTRINSIC_NAMES)
            if (same_name(name, F64_INTRINSIC_NAMES(i))) then
                intrinsic_id = F64_INTRINSIC_IDS(i)
                return
            end if
        end do
    end function f64_intrinsic_id

    subroutine lower_i32_dot_product_intrinsic(arena, node, context, value, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: a_name, b_name
        integer :: a_index, b_index
        integer :: n
        type(lr_operand_desc_t) :: a_elem, b_elem
        type(lr_operand_desc_t) :: product, sum
        integer :: i
        integer(c_int64_t) :: addr_offset

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return

        if (allocated(node%arg_indices) .and. size(node%arg_indices) == 2) then
            call get_identifier_name(arena, node%arg_indices(1), a_name, error_msg)
            if (len_trim(error_msg) > 0) return
            call get_identifier_name(arena, node%arg_indices(2), b_name, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            error_msg = 'dot_product requires exactly two array arguments'
            return
        end if

        a_index = find_symbol(context, a_name)
        if (a_index <= 0) then
            error_msg = 'dot_product first argument was not declared: '//trim(a_name)
            return
        end if
        b_index = find_symbol(context, b_name)
        if (b_index <= 0) then
            error_msg = 'dot_product second argument was not declared: '//trim(b_name)
            return
        end if
        if (.not. context%symbols(a_index)%is_array) then
            error_msg = 'dot_product requires array arguments: '//trim(a_name)
            return
        end if
        if (.not. context%symbols(b_index)%is_array) then
            error_msg = 'dot_product requires array arguments: '//trim(b_name)
            return
        end if
        if (context%symbols(a_index)%array_rank /= 1) then
            error_msg = 'dot_product requires rank-1 array arguments'
            return
        end if
        if (context%symbols(b_index)%array_rank /= 1) then
            error_msg = 'dot_product requires rank-1 array arguments'
            return
        end if
        if (context%symbols(a_index)%array_size /= &
            context%symbols(b_index)%array_size) then
            error_msg = 'dot_product requires conforming array arguments'
            return
        end if

        n = context%symbols(a_index)%array_size
        if (n <= 0) then
            error_msg = 'dot_product requires non-empty arrays'
            return
        end if

        sum = i32_immediate(context%session, 0_c_int64_t)
        do i = 0, n - 1
            addr_offset = int(i, c_int64_t)
            call load_array_linear_element(context, a_index, addr_offset, &
                                           a_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, b_index, addr_offset, &
                                           b_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, a_elem, &
                                      b_elem, product, error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_ADD, sum, &
                                      product, value, error_msg)) return
            sum = value
        end do
        call set_empty(error_msg)
    end subroutine lower_i32_dot_product_intrinsic

    subroutine unsupported_intrinsic_error(node, error_msg)
        type(call_or_subscript_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        if (allocated(node%name)) then
            call unsupported_feature_error('scalar intrinsic: '//trim(node%name), &
                                           node%line, node%column, &
                                           'direct LIRIC session supports only '// &
                                           'abs, min, max, mod, iand, ior, '// &
                                           'ieor, not, ishft, ishftc, sign, '// &
                                           'real, and the libm reals (sqrt, '// &
                                           'exp, log, sin, cos, tan, atan, '// &
                                           'atan2)', error_msg)
        else
            call unsupported_feature_error('scalar intrinsic', node%line, &
                                           node%column, &
                                           'direct LIRIC session supports only '// &
                                           'abs, min, max, mod, iand, ior, '// &
                                           'ieor, not, ishft, ishftc, sign, '// &
                                           'real, and the libm reals (sqrt, '// &
                                           'exp, log, sin, cos, tan, atan, '// &
                                           'atan2)', error_msg)
        end if
    end subroutine unsupported_intrinsic_error

    subroutine lower_command_argument_count(node, context, value, error_msg)
        ! command_argument_count() = argc - 1 (argv(0) is the program name).
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. context%has_command_args) then
            call unsupported_feature_error('command_argument_count', node%line, &
                node%column, 'command-line argument intrinsics are only '// &
                'supported in the main program', error_msg)
            return
        end if
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, context%argc_value, &
                i32_immediate(context%session, 1_c_int64_t), value, error_msg)) &
            return
        call set_empty(error_msg)
    end subroutine lower_command_argument_count

    subroutine lower_get_command_argument(arena, arg_indices, context, error_msg)
        ! get_command_argument(i, value): copy argv(i) into the character
        ! variable value, blank-padded to its declared length. The optional
        ! length and status arguments are not supported.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: index_value, buffer
        character(len=:), allocatable :: var_name
        integer :: symbol_index, length

        if (.not. context%has_command_args) then
            error_msg = 'get_command_argument is only supported in the main '// &
                        'program'
            return
        end if
        if (size(arg_indices) /= 2) then
            error_msg = 'get_command_argument requires exactly two arguments; '// &
                        'the optional length and status arguments are not '// &
                        'supported'
            return
        end if

        call lower_i32_expression(arena, arg_indices(1), context, index_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. node_exists(arena, arg_indices(2))) then
            error_msg = 'get_command_argument value argument is not an AST node'
            return
        end if
        if (is_identifier(arena, arg_indices(2))) then
            call get_identifier_name(arena, arg_indices(2), var_name, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            error_msg = 'get_command_argument value must be a character variable'
            return
        end if

        symbol_index = find_symbol(context, var_name)
        if (symbol_index <= 0) then
            error_msg = 'get_command_argument value variable is not declared: '// &
                        trim(var_name)
            return
        end if
        if (context%symbols(symbol_index)%value_kind /= VALUE_CHARACTER) then
            error_msg = 'get_command_argument value must be a character variable'
            return
        end if
        length = context%symbols(symbol_index)%character_length
        if (length <= 0) then
            error_msg = 'get_command_argument requires a fixed-length character '// &
                        'variable'
            return
        end if

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, int(length + 1, c_int64_t)), &
                buffer, error_msg)) return
        if (.not. emit_get_arg_call(context%session, index_value, &
                context%argv_value, buffer, &
                i64_immediate(context%session, int(length, c_int64_t)), &
                error_msg)) return

        context%symbols(symbol_index)%value = buffer
        context%symbols(symbol_index)%has_character_value = .true.
        call set_empty(error_msg)
    end subroutine lower_get_command_argument
