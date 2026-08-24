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
        case (I32_INTRINSIC_SELECTED_INT_KIND)
            call lower_selected_int_kind_intrinsic(arena, node, context, value, &
                                                   error_msg)
        case (I32_INTRINSIC_SELECTED_REAL_KIND)
            call lower_selected_real_kind_intrinsic(arena, node, context, value, &
                                                    error_msg)
        case (I32_INTRINSIC_MODULO)
            call lower_i32_modulo_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_DIM)
            call lower_i32_dim_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_IABS)
            call lower_i32_abs_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_IBITS)
            call lower_i32_ibits_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_IBSET)
            call lower_i32_ibset_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_IBCLR)
            call lower_i32_ibclr_intrinsic(arena, node, context, value, error_msg)
        case (I32_INTRINSIC_BIT_SIZE)
            call lower_i32_bit_size_intrinsic(arena, node, context, value, &
                                              error_msg)
        case default
            call unsupported_intrinsic_error(node, error_msg)
        end select
    end subroutine lower_i32_intrinsic_call

    ! selected_int_kind(r): smallest integer kind holding values up to 10**r-1.
    ! gfortran kinds and their decimal ranges: 1->2, 2->4, 4->9, 8->18, 16->38.
    ! Returns -1 when no available kind satisfies the request.
    integer(c_int64_t) function selected_int_kind_value(r) result(kind_value)
        integer(c_int64_t), intent(in) :: r

        if (r <= 2_c_int64_t) then
            kind_value = 1_c_int64_t
        else if (r <= 4_c_int64_t) then
            kind_value = 2_c_int64_t
        else if (r <= 9_c_int64_t) then
            kind_value = 4_c_int64_t
        else if (r <= 18_c_int64_t) then
            kind_value = 8_c_int64_t
        else if (r <= 38_c_int64_t) then
            kind_value = 16_c_int64_t
        else
            kind_value = -1_c_int64_t
        end if
    end function selected_int_kind_value

    ! selected_real_kind(p, r): smallest real kind with at least p decimal
    ! digits of precision and decimal exponent range r, restricted to the
    ! kinds direct LIRIC session actually backs (f32, f64); real(10) and
    ! real(16) fold as unavailable here even where gfortran offers extended
    ! or quad precision, since a real declaration with such a kind silently
    ! narrows to f32 instead of being honored. Error codes: -1 precision
    ! unmet, -2 range unmet, -3 both unmet.
    integer(c_int64_t) function selected_real_kind_value(p, r) result(kind_value)
        integer(c_int64_t), intent(in) :: p, r
        integer, parameter :: nk = 2
        integer(c_int64_t), parameter :: kinds(nk) = [4_c_int64_t, 8_c_int64_t]
        integer(c_int64_t), parameter :: precs(nk) = [6_c_int64_t, 15_c_int64_t]
        integer(c_int64_t), parameter :: ranges(nk) = [37_c_int64_t, 307_c_int64_t]
        logical :: prec_ok, range_ok
        integer :: i

        prec_ok = .false.
        range_ok = .false.
        do i = 1, nk
            if (precs(i) >= p) prec_ok = .true.
            if (ranges(i) >= r) range_ok = .true.
            if (precs(i) >= p .and. ranges(i) >= r) then
                kind_value = kinds(i)
                return
            end if
        end do
        if (.not. prec_ok .and. .not. range_ok) then
            kind_value = -3_c_int64_t
        else if (.not. prec_ok) then
            kind_value = -1_c_int64_t
        else
            kind_value = -2_c_int64_t
        end if
    end function selected_real_kind_value

    subroutine lower_selected_int_kind_intrinsic(arena, node, context, value, &
                                                 error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: r

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return
        call eval_i32_constant(arena, node%arg_indices(1), context, r, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'selected_int_kind requires a compile-time integer '// &
                        'argument: '//error_msg
            return
        end if
        value = i32_immediate(context%session, selected_int_kind_value(r))
        call set_empty(error_msg)
    end subroutine lower_selected_int_kind_intrinsic

    subroutine lower_selected_real_kind_intrinsic(arena, node, context, value, &
                                                  error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: p, r

        if (.not. require_intrinsic_arg_count(node, 1, 2, error_msg)) return
        p = 0_c_int64_t
        r = 0_c_int64_t
        call eval_i32_constant(arena, node%arg_indices(1), context, p, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = 'selected_real_kind requires compile-time integer '// &
                        'arguments: '//error_msg
            return
        end if
        if (size(node%arg_indices) >= 2) then
            call eval_i32_constant(arena, node%arg_indices(2), context, r, error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'selected_real_kind requires compile-time integer '// &
                            'arguments: '//error_msg
                return
            end if
        end if
        value = i32_immediate(context%session, selected_real_kind_value(p, r))
        call set_empty(error_msg)
    end subroutine lower_selected_real_kind_intrinsic

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

    subroutine lower_f32_intrinsic_call(arena, node, intrinsic_id, context, &
                                        value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: intrinsic_id
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (intrinsic_id)
        case (F64_INTRINSIC_ABS)
            call lower_f32_abs_intrinsic(arena, node, context, value, error_msg)
        case (F64_INTRINSIC_MIN)
            call lower_f32_minmax_intrinsic(arena, node, LR_FCMP_OLE, context, &
                                            value, error_msg)
        case (F64_INTRINSIC_MAX)
            call lower_f32_minmax_intrinsic(arena, node, LR_FCMP_OGE, context, &
                                            value, error_msg)
        case (F64_INTRINSIC_REAL)
            call lower_f32_real_intrinsic(arena, node, context, value, error_msg)
        case (F64_INTRINSIC_SIGN)
            call lower_f32_sign_intrinsic(arena, node, context, value, error_msg)
        case (F64_INTRINSIC_SQRT)
            call lower_f32_libm_call(arena, node, 'sqrtf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_EXP)
            call lower_f32_libm_call(arena, node, 'expf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_LOG)
            call lower_f32_libm_call(arena, node, 'logf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_SIN)
            call lower_f32_libm_call(arena, node, 'sinf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_COS)
            call lower_f32_libm_call(arena, node, 'cosf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_TAN)
            call lower_f32_libm_call(arena, node, 'tanf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ATAN)
            call lower_f32_libm_call(arena, node, 'atanf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ATAN2)
            call lower_f32_libm_call(arena, node, 'atan2f', 2, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ASIN)
            call lower_f32_libm_call(arena, node, 'asinf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ACOS)
            call lower_f32_libm_call(arena, node, 'acosf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_SINH)
            call lower_f32_libm_call(arena, node, 'sinhf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_COSH)
            call lower_f32_libm_call(arena, node, 'coshf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_TANH)
            call lower_f32_libm_call(arena, node, 'tanhf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ASINH)
            call lower_f32_libm_call(arena, node, 'asinhf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ACOSH)
            call lower_f32_libm_call(arena, node, 'acoshf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ATANH)
            call lower_f32_libm_call(arena, node, 'atanhf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_LOG10)
            call lower_f32_libm_call(arena, node, 'log10f', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ERF)
            call lower_f32_libm_call(arena, node, 'erff', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ERFC)
            call lower_f32_libm_call(arena, node, 'erfcf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_GAMMA)
            call lower_f32_libm_call(arena, node, 'tgammaf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_LOG_GAMMA)
            call lower_f32_libm_call(arena, node, 'lgammaf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_HYPOT)
            call lower_f32_libm_call(arena, node, 'hypotf', 2, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_MOD)
            call lower_f32_remainder_intrinsic(arena, node, 'truncf', context, &
                                               value, error_msg)
        case (F64_INTRINSIC_MODULO)
            call lower_f32_modulo_intrinsic(arena, node, context, value, &
                                            error_msg)
        case (F64_INTRINSIC_AINT)
            call lower_f32_real_conversion(arena, node, 'truncf', context, &
                                           value, error_msg)
        case (F64_INTRINSIC_ANINT)
            ! ANINT rounds half away from zero, matching C round().
            call lower_f32_real_conversion(arena, node, 'roundf', context, &
                                           value, error_msg)
        case default
            call unsupported_intrinsic_error(node, error_msg)
        end select
    end subroutine lower_f32_intrinsic_call

    subroutine lower_real_inquiry_intrinsic(arena, node, context, vk, value, &
                                            error_msg)
        !! Lower tiny/huge/epsilon to a constant f32/f64 immediate. The result
        !! kind follows the argument kind, so the value is computed with the
        !! compiler's own intrinsics and emitted into the requested value path.
        use, intrinsic :: iso_c_binding, only: c_float, c_double
        use, intrinsic :: iso_fortran_env, only: real32, real64
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: kind_num
        real(c_double) :: val

        call set_empty(error_msg)
        kind_num = inquiry_arg_real_kind(arena, node, context)
        if (kind_num /= 4 .and. kind_num /= 8) then
            call unsupported_intrinsic_error(node, error_msg)
            return
        end if
        select case (trim(node%name))
        case ('tiny')
            if (kind_num == 4) then
                val = real(tiny(1.0_real32), c_double)
            else
                val = tiny(1.0_real64)
            end if
        case ('huge')
            if (kind_num == 4) then
                val = real(huge(1.0_real32), c_double)
            else
                val = huge(1.0_real64)
            end if
        case ('epsilon')
            if (kind_num == 4) then
                val = real(epsilon(1.0_real32), c_double)
            else
                val = epsilon(1.0_real64)
            end if
        case default
            call unsupported_intrinsic_error(node, error_msg)
            return
        end select
        if (vk == VALUE_F32) then
            value = liric_f32_immediate(context%session, real(val, c_float))
        else
            value = liric_f64_immediate(context%session, val)
        end if
    end subroutine lower_real_inquiry_intrinsic

    subroutine lower_f32_libm_call(arena, node, libm_name, arg_count, context, &
                                   value, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
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
            call lower_f32_intrinsic_arg(arena, node%arg_indices(i), context, &
                                         args(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (.not. emit_liric_f32_call(context%session, libm_name, args, value, &
                                      error_msg)) return
    end subroutine lower_f32_libm_call

    ! modulo(a, p) = a - floor(a / p) * p for default-real arguments.
    subroutine lower_f32_modulo_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, p, quotient, floored, product
        type(lr_operand_desc_t) :: args(1)

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_f32_intrinsic_arg(arena, node%arg_indices(1), context, a, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_f32_intrinsic_arg(arena, node%arg_indices(2), context, p, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_f32_binary(context%session, LR_OP_FDIV, a, p, &
                                        quotient, error_msg)) return
        args(1) = quotient
        if (.not. emit_liric_f32_call(context%session, 'floorf', args, floored, &
                                      error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, floored, p, &
                                        product, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FSUB, a, product, &
                                        value, error_msg)) return
    end subroutine lower_f32_modulo_intrinsic

    ! MOD(a, p) = a - trunc(a / p) * p for default-real arguments. Unlike
    ! MODULO, the result follows the sign of a because the quotient truncates
    ! toward zero.
    subroutine lower_f32_remainder_intrinsic(arena, node, rounding_name, &
                                             context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: rounding_name
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, p, quotient, rounded, product
        type(lr_operand_desc_t) :: args(1)

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_f32_intrinsic_arg(arena, node%arg_indices(1), context, a, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_f32_intrinsic_arg(arena, node%arg_indices(2), context, p, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_f32_binary(context%session, LR_OP_FDIV, a, p, &
                                        quotient, error_msg)) return
        args(1) = quotient
        if (.not. emit_liric_f32_call(context%session, rounding_name, args, &
                                      rounded, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, rounded, &
                                        p, product, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FSUB, a, product, &
                                        value, error_msg)) return
    end subroutine lower_f32_remainder_intrinsic

    subroutine lower_f32_sign_intrinsic(arena, node, context, value, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, b, zero
        type(lr_operand_desc_t) :: a_nonneg, neg_a, magnitude
        type(lr_operand_desc_t) :: b_nonneg, neg_magnitude

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_f32_intrinsic_arg(arena, node%arg_indices(1), context, a, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_f32_intrinsic_arg(arena, node%arg_indices(2), context, b, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        zero = liric_f32_immediate(context%session, 0.0_c_float)
        if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGE, a, zero, &
                                      a_nonneg, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FSUB, zero, a, &
                                        neg_a, error_msg)) return
        call select_value(context, a_nonneg, a, neg_a, magnitude, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGE, b, zero, &
                                      b_nonneg, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FSUB, zero, &
                                        magnitude, neg_magnitude, error_msg)) &
            return
        call select_value(context, b_nonneg, magnitude, neg_magnitude, value, &
                          error_msg)
    end subroutine lower_f32_sign_intrinsic

    subroutine lower_f32_abs_intrinsic(arena, node, context, value, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg, re_val, im_val
        type(lr_operand_desc_t) :: call_args(2)
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: negative
        type(lr_operand_desc_t) :: zero

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return

        if (is_complex_valued(arena, node%arg_indices(1), context, VALUE_C4)) then
            call eval_c4_rhs(arena, node%arg_indices(1), context, re_val, &
                             im_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call_args(1) = re_val
            call_args(2) = im_val
            if (.not. emit_liric_f32_call(context%session, 'hypotf', &
                    call_args, value, error_msg)) return
            return
        end if

        call lower_f32_intrinsic_arg(arena, node%arg_indices(1), context, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        zero = liric_f32_immediate(context%session, 0.0_c_float)
        if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGE, arg, zero, &
                                      condition, error_msg)) return
        if (.not. emit_liric_f32_binary(context%session, LR_OP_FSUB, zero, &
                                        arg, negative, error_msg)) return
        call select_value(context, condition, arg, negative, value, error_msg)
    end subroutine lower_f32_abs_intrinsic

    subroutine lower_ieee_value_intrinsic(arena, node, context, value_kind, &
                                          value, error_msg)
        use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
        use, intrinsic :: iso_c_binding, only: c_double, c_float
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: class_name

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        if (.not. is_identifier(arena, node%arg_indices(2))) then
            error_msg = 'ieee_value currently supports IEEE_QUIET_NAN only'
            return
        end if
        call get_identifier_name(arena, node%arg_indices(2), class_name, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. same_name(class_name, 'ieee_quiet_nan')) then
            error_msg = 'ieee_value currently supports IEEE_QUIET_NAN only'
            return
        end if

        select case (value_kind)
        case (VALUE_F32)
            value = liric_f32_immediate(context%session, &
                ieee_value(0.0_c_float, ieee_quiet_nan))
        case (VALUE_F64)
            value = liric_f64_immediate(context%session, &
                ieee_value(0.0_c_double, ieee_quiet_nan))
        case default
            error_msg = 'ieee_value requires a real(4) or real(8) argument'
        end select
    end subroutine lower_ieee_value_intrinsic

    subroutine lower_ieee_is_nan_intrinsic(arena, node, context, value, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        integer :: value_kind

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return
        value_kind = scalar_real_expr_kind(arena, node%arg_indices(1), context)
        select case (value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, node%arg_indices(1), context, arg, &
                                       error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_UNO, arg, arg, &
                                          value, error_msg)) return
        case (VALUE_F64)
            call lower_f64_expression(arena, node%arg_indices(1), context, arg, &
                                       error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_UNO, arg, arg, &
                                          value, error_msg)) return
        case default
            error_msg = 'ieee_is_nan requires a real(4) or real(8) argument'
        end select
    end subroutine lower_ieee_is_nan_intrinsic

    subroutine lower_f32_minmax_intrinsic(arena, node, predicate, context, &
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

        if (.not. require_intrinsic_arg_count(node, 2, 64, error_msg)) return

        call lower_f32_intrinsic_arg(arena, node%arg_indices(1), context, &
                                     value, error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 2, size(node%arg_indices)
            call lower_f32_intrinsic_arg(arena, node%arg_indices(i), context, &
                                         candidate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_fcmp(context%session, predicate, value, &
                                          candidate, condition, error_msg)) return
            call select_value(context, condition, value, candidate, selected, &
                              error_msg)
            if (len_trim(error_msg) > 0) return
            value = selected
        end do
    end subroutine lower_f32_minmax_intrinsic

    subroutine lower_f32_real_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: handled
        integer :: argument_index, kind_index

        ! real(x) and real(x, 4)/real(x, kind=sp) convert x to f32. The optional
        ! second argument is the result kind; ignore it here.
        if (.not. require_intrinsic_arg_count(node, 1, 2, error_msg)) return
        call intrinsic_real_conversion_args(arena, node, argument_index, &
                                            kind_index)
        if (argument_index <= 0) then
            error_msg = 'real requires a value argument'
            return
        end if

        call lower_boz_real_bits(arena, argument_index, context, .false., &
                                 value, handled, error_msg)
        if (handled .or. len_trim(error_msg) > 0) return

        call lower_f32_intrinsic_arg(arena, argument_index, context, value, &
                                     error_msg)
    end subroutine lower_f32_real_intrinsic

    recursive subroutine lower_legacy_integer_conversion(arena, node, context, &
                                                         want_f64, value, error_msg)
        !! Lower GNU FLOAT/DFLOAT, whose argument is integer-only.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: want_f64
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: integer_kind

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return
        integer_kind = expression_value_kind(arena, node%arg_indices(1), context, &
                                             VALUE_I32)
        if (integer_kind /= VALUE_I8 .and. integer_kind /= VALUE_I16 .and. &
            integer_kind /= VALUE_I32 .and. integer_kind /= VALUE_I64) then
            error_msg = trim(node%name)//' argument must be INTEGER'
            return
        end if

        if (want_f64) then
            call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, &
                                         value, error_msg)
        else
            call lower_f32_intrinsic_arg(arena, node%arg_indices(1), context, &
                                         value, error_msg)
        end if
    end subroutine lower_legacy_integer_conversion

    recursive subroutine lower_f32_intrinsic_arg(arena, node_index, context, value, &
                                                 error_msg)
        ! Lower an argument to an f32 intrinsic. Real literals and f32 variables
        ! lower as f32; integer literals/variables are promoted to f32.
        use, intrinsic :: iso_c_binding, only: c_float
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: integer_value
        real(c_double) :: lit_f64
        character(len=:), allocatable :: lit_value, lit_type
        integer :: actual_index
        integer :: integer_kind
        character(len=:), allocatable :: keyword

        call unwrap_intrinsic_arg(arena, node_index, keyword, actual_index)
        if (actual_index /= node_index) then
            call lower_f32_intrinsic_arg(arena, actual_index, context, value, &
                                         error_msg)
            return
        end if

        if (is_literal(arena, node_index)) then
            if (is_real_literal(arena, node_index)) then
                call get_literal_info(arena, node_index, lit_value, lit_type, &
                                      error_msg)
                if (len_trim(error_msg) > 0) return
                call parse_f64_literal(lit_value, context, lit_f64, error_msg, &
                                       node_index)
                if (len_trim(error_msg) > 0) return
                value = liric_f32_immediate(context%session, real(lit_f64, c_float))
                return
            end if
        end if

        if (scalar_real_expr_kind(arena, node_index, context) == VALUE_F32) then
            call lower_f32_expression(arena, node_index, context, value, &
                                      error_msg)
            return
        end if

        ! An f64 argument to a single-precision conversion (real(d), real(d, 4))
        ! narrows to f32.
        if (scalar_real_expr_kind(arena, node_index, context) == VALUE_F64) then
            block
                type(lr_operand_desc_t) :: wide_value
                call lower_f64_expression(arena, node_index, context, &
                                          wide_value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_f64_to_f32(context%session, wide_value, &
                                                value, error_msg)) return
            end block
            return
        end if

        ! Integer arguments retain their source width until SITOFP.  In
        ! particular, FLOAT(integer(8)) must not truncate through i32.
        integer_kind = expression_value_kind(arena, node_index, context, VALUE_I32)
        select case (integer_kind)
        case (VALUE_I64)
            call lower_i64_expression(arena, node_index, context, integer_value, &
                                      error_msg)
        case (VALUE_I16)
            call lower_i16_expression(arena, node_index, context, integer_value, &
                                      error_msg)
        case (VALUE_I8)
            call lower_i8_expression(arena, node_index, context, integer_value, &
                                     error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, integer_value, &
                                      error_msg)
        end select
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_integer_to_f32(context%session, integer_value, value, &
                                        error_msg)) return
    end subroutine lower_f32_intrinsic_arg

    recursive subroutine lower_f64_intrinsic_call(arena, node, intrinsic_id, context, &
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
        case (F64_INTRINSIC_ASIN)
            call lower_f64_libm_call(arena, node, 'asin', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ACOS)
            call lower_f64_libm_call(arena, node, 'acos', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_SINH)
            call lower_f64_libm_call(arena, node, 'sinh', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_COSH)
            call lower_f64_libm_call(arena, node, 'cosh', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_TANH)
            call lower_f64_libm_call(arena, node, 'tanh', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ASINH)
            call lower_f64_libm_call(arena, node, 'asinh', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ACOSH)
            call lower_f64_libm_call(arena, node, 'acosh', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ATANH)
            call lower_f64_libm_call(arena, node, 'atanh', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_LOG10)
            call lower_f64_libm_call(arena, node, 'log10', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ERF)
            call lower_f64_libm_call(arena, node, 'erf', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_ERFC)
            call lower_f64_libm_call(arena, node, 'erfc', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_GAMMA)
            call lower_f64_libm_call(arena, node, 'tgamma', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_LOG_GAMMA)
            call lower_f64_libm_call(arena, node, 'lgamma', 1, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_HYPOT)
            call lower_f64_libm_call(arena, node, 'hypot', 2, context, value, &
                                     error_msg)
        case (F64_INTRINSIC_MOD)
            call lower_f64_remainder_intrinsic(arena, node, 'trunc', context, &
                                               value, error_msg)
        case (F64_INTRINSIC_MODULO)
            call lower_f64_remainder_intrinsic(arena, node, 'floor', context, &
                                               value, error_msg)
        case (F64_INTRINSIC_AINT)
            call lower_f64_real_conversion(arena, node, 'trunc', context, &
                                           value, error_msg)
        case (F64_INTRINSIC_ANINT)
            ! ANINT rounds half away from zero, matching C round().
            call lower_f64_real_conversion(arena, node, 'round', context, &
                                           value, error_msg)
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
        type(lr_operand_desc_t) :: call_args(1)

        if (.not. require_intrinsic_arg_count(node, 1, 2, error_msg)) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        ! int(x) truncates toward zero, which fptosi already does directly.
        ! nint/floor/ceiling apply the libm rounding first, then convert.
        if (len_trim(libm_name) > 0) then
            call_args(1) = arg
            if (.not. emit_liric_f64_call(context%session, libm_name, call_args, &
                                          rounded, error_msg)) return
        else
            rounded = arg
        end if
        if (.not. emit_liric_f64_to_i32(context%session, rounded, value, &
                                        error_msg)) return
    end subroutine lower_i32_from_f64_intrinsic

    subroutine lower_i64_from_f64_intrinsic(arena, node, libm_name, context, &
                                            value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: libm_name
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg
        type(lr_operand_desc_t) :: rounded
        type(lr_operand_desc_t) :: call_args(1)

        if (.not. require_intrinsic_arg_count(node, 1, 2, error_msg)) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        if (len_trim(libm_name) > 0) then
            call_args(1) = arg
            if (.not. emit_liric_f64_call(context%session, libm_name, call_args, &
                                          rounded, error_msg)) return
        else
            rounded = arg
        end if
        if (.not. emit_liric_f64_to_i64(context%session, rounded, value, &
                                        error_msg)) return
    end subroutine lower_i64_from_f64_intrinsic

    recursive subroutine lower_f64_abs_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg, re_val, im_val
        type(lr_operand_desc_t) :: call_args(2)
        type(lr_operand_desc_t) :: condition
        type(lr_operand_desc_t) :: negative
        type(lr_operand_desc_t) :: zero

        if (.not. require_intrinsic_arg_count(node, 1, 1, error_msg)) return

        if (is_complex_valued(arena, node%arg_indices(1), context, VALUE_C8)) then
            call eval_c8_rhs(arena, node%arg_indices(1), context, re_val, &
                             im_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call_args(1) = re_val
            call_args(2) = im_val
            if (.not. emit_liric_f64_call(context%session, 'hypot', &
                    call_args, value, error_msg)) return
            return
        end if

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

    ! modulo(a, p): remainder with the sign of p.
    ! For integers: r = a srem p; if r /= 0 and sign(r) /= sign(p): r = r + p.
    subroutine lower_i32_modulo_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, p, r, zero, r_plus_p
        type(lr_operand_desc_t) :: r_nonzero, signs_differ, adjust

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, a, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, p, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_SREM, a, p, r, &
                                  error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, r, p, r_plus_p, &
                                  error_msg)) return
        ! r_nonzero = (r /= 0)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, r, zero, &
                                      r_nonzero, error_msg)) return
        ! signs_differ: (r XOR p) < 0 -- MSB set iff signs differ
        if (.not. emit_i32_binary(context%session, LR_OP_XOR, r, p, adjust, &
                                  error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, adjust, zero, &
                                      signs_differ, error_msg)) return
        ! need_adjust = r_nonzero AND signs_differ
        if (.not. emit_i32_binary(context%session, LR_OP_AND, r_nonzero, &
                                  signs_differ, adjust, error_msg)) return
        ! result = need_adjust ? r + p : r
        call select_value(context, adjust, r_plus_p, r, value, error_msg)
    end subroutine lower_i32_modulo_intrinsic

    ! dim(x, y): max(x - y, 0) -- positive difference or zero.
    subroutine lower_i32_dim_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: x, y, diff, zero, is_pos

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, x, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, y, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, x, y, diff, &
                                  error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGT, diff, zero, &
                                      is_pos, error_msg)) return
        call select_value(context, is_pos, diff, zero, value, error_msg)
    end subroutine lower_i32_dim_intrinsic

    ! ibits(i, pos, len): the len-bit field of i starting at bit pos, shifted
    ! down to bit 0. shifted = i >> pos (logical); masked keeps its low len
    ! bits via a shift-left/shift-right-by-(32-len) round trip, which also
    ! handles len == 32 (shift by 0, keep everything) without an overflowing
    ! shift count. len == 0 shifts by 32, which is out of range, so that case
    ! is selected separately and forced to zero.
    subroutine lower_i32_ibits_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: i_val, pos_val, len_val
        type(lr_operand_desc_t) :: shifted, complement, tmp, masked
        type(lr_operand_desc_t) :: thirty_two, zero, is_zero_len

        if (.not. require_intrinsic_arg_count(node, 3, 3, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, i_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, pos_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(3), context, len_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_binary(context%session, LR_OP_LSHR, i_val, pos_val, &
                                  shifted, error_msg)) return
        thirty_two = i32_immediate(context%session, 32_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, thirty_two, &
                                  len_val, complement, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_SHL, shifted, &
                                  complement, tmp, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_LSHR, tmp, complement, &
                                  masked, error_msg)) return

        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, len_val, &
                                      zero, is_zero_len, error_msg)) return
        call select_value(context, is_zero_len, zero, masked, value, error_msg)
    end subroutine lower_i32_ibits_intrinsic

    subroutine lower_i32_ibset_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: i_val, pos_val, one, bit_mask

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, i_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, pos_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        one = i32_immediate(context%session, 1_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_SHL, one, pos_val, &
                                  bit_mask, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_OR, i_val, bit_mask, &
                                  value, error_msg)) return
    end subroutine lower_i32_ibset_intrinsic

    subroutine lower_i32_ibclr_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: i_val, pos_val, one, bit_mask
        type(lr_operand_desc_t) :: all_ones, inverted_mask

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, i_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, pos_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        one = i32_immediate(context%session, 1_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_SHL, one, pos_val, &
                                  bit_mask, error_msg)) return
        all_ones = i32_immediate(context%session, -1_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_XOR, bit_mask, &
                                  all_ones, inverted_mask, error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_AND, i_val, &
                                  inverted_mask, value, error_msg)) return
    end subroutine lower_i32_ibclr_intrinsic

    ! bit_size(x): storage width in bits of x's declared integer kind. Folds
    ! from x's declared kind alone, matching the compile-time bit_size used
    ! in a PARAMETER initializer, so it also lowers a runtime x whose kind
    ! is not VALUE_I32 (e.g. bit_size(int8_var) assigned to a default
    ! integer result).
    subroutine lower_i32_bit_size_intrinsic(arena, node, context, value, &
                                            error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: constant_value

        call eval_bit_size_constant(arena, node, context, constant_value, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        value = i32_immediate(context%session, constant_value)
    end subroutine lower_i32_bit_size_intrinsic

    ! btest(i, pos): .true. when bit pos of i is set.
    subroutine lower_btest_intrinsic(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: i_val, pos_val, shifted, one, bit, zero

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_i32_expression(arena, node%arg_indices(1), context, i_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(2), context, pos_val, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_binary(context%session, LR_OP_LSHR, i_val, pos_val, &
                                  shifted, error_msg)) return
        one = i32_immediate(context%session, 1_c_int64_t)
        if (.not. emit_i32_binary(context%session, LR_OP_AND, shifted, one, &
                                  bit, error_msg)) return
        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, bit, zero, &
                                      value, error_msg)) return
    end subroutine lower_btest_intrinsic

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

    subroutine lower_i32_legacy_minmax_call(arena, node, context, real_args, &
                                            is_max, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: real_args
        logical, intent(in) :: is_max
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: real_value
        integer(c_int) :: predicate

        if (real_args) then
            if (is_max) then
                predicate = LR_FCMP_OGE
            else
                predicate = LR_FCMP_OLE
            end if
            call lower_f32_minmax_intrinsic(arena, node, predicate, context, &
                                            real_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_to_i32(context%session, real_value, value, &
                                            error_msg)) return
        else
            if (is_max) then
                predicate = LR_CMP_SGE
            else
                predicate = LR_CMP_SLE
            end if
            call lower_i32_minmax_intrinsic(arena, node, predicate, context, &
                                            value, error_msg)
        end if
    end subroutine lower_i32_legacy_minmax_call

    subroutine lower_i64_minmax_intrinsic(arena, node, predicate, context, &
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

        if (.not. require_intrinsic_arg_count(node, 2, 64, error_msg)) return

        call lower_i64_expression(arena, node%arg_indices(1), context, value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 2, size(node%arg_indices)
            call lower_i64_expression(arena, node%arg_indices(i), context, &
                                      candidate, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i64_icmp(context%session, predicate, value, &
                                          candidate, condition, error_msg)) return
            call select_value(context, condition, value, candidate, selected, &
                              error_msg)
            if (len_trim(error_msg) > 0) return
            value = selected
        end do
    end subroutine lower_i64_minmax_intrinsic

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
        logical :: handled
        integer :: argument_index, kind_index

        ! real(x) and real(x, kind=dp)/real(x, 8): the optional second argument
        ! is the result kind, already resolved to f64 by the dispatch. Lower the
        ! value argument and ignore the kind selector.
        if (.not. require_intrinsic_arg_count(node, 1, 2, error_msg)) return
        call intrinsic_real_conversion_args(arena, node, argument_index, &
                                            kind_index)
        if (argument_index <= 0) then
            error_msg = 'real requires a value argument'
            return
        end if

        call lower_boz_real_bits(arena, argument_index, context, .true., &
                                 value, handled, error_msg)
        if (handled .or. len_trim(error_msg) > 0) return

        call lower_f64_intrinsic_arg(arena, argument_index, context, value, &
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

    ! MOD uses truncation toward zero; MODULO uses floor so the result has
    ! the divisor's sign.
    subroutine lower_f64_remainder_intrinsic(arena, node, rounding_name, &
                                             context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: rounding_name
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: a, p, quotient, rounded, product
        type(lr_operand_desc_t) :: args(1)

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(1), context, a, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_f64_intrinsic_arg(arena, node%arg_indices(2), context, p, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_liric_f64_binary(context%session, LR_OP_FDIV, a, p, &
                                        quotient, error_msg)) return
        args(1) = quotient
        if (.not. emit_liric_f64_call(context%session, rounding_name, args, &
                                      rounded, error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, rounded, &
                                        p, product, error_msg)) return
        if (.not. emit_liric_f64_binary(context%session, LR_OP_FSUB, a, product, &
                                        value, error_msg)) return
    end subroutine lower_f64_remainder_intrinsic

    recursive subroutine lower_f64_intrinsic_arg(arena, node_index, context, value, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: integer_value
        type(lr_operand_desc_t) :: f32_value
        integer :: actual_index
        integer :: integer_kind
        character(len=:), allocatable :: keyword

        call unwrap_intrinsic_arg(arena, node_index, keyword, actual_index)
        if (actual_index /= node_index) then
            call lower_f64_intrinsic_arg(arena, actual_index, context, value, &
                                         error_msg)
            return
        end if

        if (scalar_real_expr_kind(arena, node_index, context) == VALUE_F64) then
            call lower_f64_expression(arena, node_index, context, value, &
                                      error_msg)
            return
        end if

        ! f32 argument: promote to f64 via FPEXT.
        if (scalar_real_expr_kind(arena, node_index, context) == VALUE_F32) then
            call lower_f32_expression(arena, node_index, context, f32_value, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_to_f64(context%session, f32_value, value, &
                                             error_msg)) return
            return
        end if

        integer_kind = expression_value_kind(arena, node_index, context, VALUE_I32)
        select case (integer_kind)
        case (VALUE_I64)
            call lower_i64_expression(arena, node_index, context, integer_value, &
                                      error_msg)
        case (VALUE_I16)
            call lower_i16_expression(arena, node_index, context, integer_value, &
                                      error_msg)
        case (VALUE_I8)
            call lower_i8_expression(arena, node_index, context, integer_value, &
                                     error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, integer_value, &
                                      error_msg)
        end select
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_integer_to_f64(context%session, integer_value, value, &
                                        error_msg)) return
    end subroutine lower_f64_intrinsic_arg

    integer function reduction_operation(name) result(op)
        !! Map an intrinsic reduction name onto its DIM_REDUCE_* operation.
        !! Returns 0 for a name that is not a reduction.
        character(len=*), intent(in) :: name

        select case (trim(adjustl(lowercase_text(name))))
        case ('sum'); op = DIM_REDUCE_SUM
        case ('product'); op = DIM_REDUCE_PROD
        case ('count'); op = DIM_REDUCE_COUNT
        case ('any'); op = DIM_REDUCE_ANY
        case ('all'); op = DIM_REDUCE_ALL
        case ('norm2'); op = DIM_REDUCE_NORM2
        case ('minval'); op = DIM_REDUCE_MIN
        case ('maxval'); op = DIM_REDUCE_MAX
        case default; op = 0
        end select
    end function reduction_operation

    logical function reduction_is_mask_valued(op) result(is_mask)
        !! COUNT, ANY, and ALL consume a logical element and accumulate an i32
        !! 0/1 value; the other reductions consume an element of the source's
        !! own type.
        integer, intent(in) :: op

        is_mask = (op == DIM_REDUCE_COUNT .or. op == DIM_REDUCE_ANY .or. &
                   op == DIM_REDUCE_ALL)
    end function reduction_is_mask_valued

    subroutine reduction_identity(context, op, vk, acc, error_msg)
        !! The identity element of one reduction over element kind vk. Folding
        !! starts here, so this is also the value of the reduction over an
        !! empty array, which is what the intrinsic definitions require:
        !! SUM 0, PRODUCT 1, COUNT 0, ANY false, ALL true, MINVAL HUGE, and
        !! MAXVAL -HUGE for the kind involved.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: op
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: acc
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (reduction_is_mask_valued(op)) then
            if (op == DIM_REDUCE_ALL) then
                acc = i32_immediate(context%session, 1_c_int64_t)
            else
                acc = i32_immediate(context%session, 0_c_int64_t)
            end if
            return
        end if
        select case (op)
        case (DIM_REDUCE_SUM)
            select case (vk)
            case (VALUE_F32); acc = liric_f32_immediate(context%session, 0.0_c_float)
            case (VALUE_F64); acc = liric_f64_immediate(context%session, 0.0_c_double)
            case default; acc = i32_immediate(context%session, 0_c_int64_t)
            end select
        case (DIM_REDUCE_PROD)
            select case (vk)
            case (VALUE_F32); acc = liric_f32_immediate(context%session, 1.0_c_float)
            case (VALUE_F64); acc = liric_f64_immediate(context%session, 1.0_c_double)
            case default; acc = i32_immediate(context%session, 1_c_int64_t)
            end select
        case (DIM_REDUCE_MIN)
            select case (vk)
            case (VALUE_F32)
                acc = liric_f32_immediate(context%session, huge(0.0_c_float))
            case (VALUE_F64)
                acc = liric_f64_immediate(context%session, huge(0.0_c_double))
            case default
                acc = i32_immediate(context%session, 2147483647_c_int64_t)
            end select
        case (DIM_REDUCE_MAX)
            select case (vk)
            case (VALUE_F32)
                acc = liric_f32_immediate(context%session, -huge(0.0_c_float))
            case (VALUE_F64)
                acc = liric_f64_immediate(context%session, -huge(0.0_c_double))
            case default
                acc = i32_immediate(context%session, -2147483648_c_int64_t)
            end select
        case default
            error_msg = 'reduction has no identity element for this operation'
        end select
    end subroutine reduction_identity

    subroutine reduction_combine(context, op, vk, acc, elem, error_msg)
        !! Fold one element into the accumulator. Every reduction here is
        !! associative and commutative over its identity, so the standard
        !! leaves the traversal order unspecified and the fold runs in
        !! ascending linear-index order. Integer SUM and PRODUCT wrap exactly
        !! as the source operations do; real SUM and PRODUCT accumulate in
        !! that same order, which fixes the rounding sequence.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: op
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(inout) :: acc
        type(lr_operand_desc_t), intent(in) :: elem
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: next, cond
        type(lr_operand_desc_t) :: call_args(2)

        call set_empty(error_msg)
        if (reduction_is_mask_valued(op)) then
            select case (op)
            case (DIM_REDUCE_ANY)
                if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, elem, &
                                          next, error_msg)) return
            case (DIM_REDUCE_ALL)
                if (.not. emit_i32_binary(context%session, LR_OP_AND, acc, elem, &
                                          next, error_msg)) return
            case default
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, acc, elem, &
                                          next, error_msg)) return
            end select
            acc = next
            return
        end if

        select case (op)
        case (DIM_REDUCE_SUM, DIM_REDUCE_PROD)
            select case (vk)
            case (VALUE_F32)
                if (.not. emit_liric_f32_binary(context%session, &
                        merge(LR_OP_FADD, LR_OP_FMUL, op == DIM_REDUCE_SUM), &
                        acc, elem, next, error_msg)) return
            case (VALUE_F64)
                if (.not. emit_liric_f64_binary(context%session, &
                        merge(LR_OP_FADD, LR_OP_FMUL, op == DIM_REDUCE_SUM), &
                        acc, elem, next, error_msg)) return
            case default
                if (.not. emit_i32_binary(context%session, &
                        merge(LR_OP_ADD, LR_OP_MUL, op == DIM_REDUCE_SUM), &
                        acc, elem, next, error_msg)) return
            end select
        case (DIM_REDUCE_MIN, DIM_REDUCE_MAX)
            select case (vk)
            case (VALUE_F32)
                call_args(1) = acc
                call_args(2) = elem
                if (.not. emit_liric_f32_call(context%session, &
                        merge('fminf', 'fmaxf', op == DIM_REDUCE_MIN), &
                        call_args, next, error_msg)) return
            case (VALUE_F64)
                call_args(1) = acc
                call_args(2) = elem
                if (.not. emit_liric_f64_call(context%session, &
                        merge('fmin', 'fmax', op == DIM_REDUCE_MIN), &
                        call_args, next, error_msg)) return
            case default
                if (.not. emit_liric_i32_icmp(context%session, &
                        merge(LR_CMP_SLT, LR_CMP_SGT, op == DIM_REDUCE_MIN), &
                        elem, acc, cond, error_msg)) return
                call select_value(context, cond, elem, acc, next, error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        case default
            error_msg = 'reduction operation has no element combine rule'
            return
        end select
        acc = next
    end subroutine reduction_combine

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

        a_index = find_symbol_compat(context, a_name)
        if (a_index <= 0) then
            error_msg = 'dot_product first argument was not declared: '//trim(a_name)
            return
        end if
        b_index = find_symbol_compat(context, b_name)
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

    subroutine lower_logical_dot_product_intrinsic(arena, node, context, value, &
                                                   error_msg)
        ! For logical arrays, DOT_PRODUCT is ANY(a .AND. b), not an integer
        ! sum. Keep this path separate from the numeric implementation so the
        ! generic intrinsic dispatcher preserves the result type.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: a_name, b_name
        integer :: a_index, b_index, n, i
        integer(c_int64_t) :: addr_offset
        type(lr_operand_desc_t) :: a_elem, b_elem, both, acc, next_acc

        if (.not. require_intrinsic_arg_count(node, 2, 2, error_msg)) return
        call get_identifier_name(arena, node%arg_indices(1), a_name, error_msg)
        if (len_trim(error_msg) > 0) return
        call get_identifier_name(arena, node%arg_indices(2), b_name, error_msg)
        if (len_trim(error_msg) > 0) return

        a_index = find_symbol_compat(context, a_name)
        b_index = find_symbol_compat(context, b_name)
        if (a_index <= 0 .or. b_index <= 0) then
            error_msg = 'logical dot_product arguments were not declared'
            return
        end if
        if (.not. context%symbols(a_index)%is_array .or. &
            .not. context%symbols(b_index)%is_array) then
            error_msg = 'logical dot_product requires array arguments'
            return
        end if
        if (context%symbols(a_index)%value_kind /= VALUE_LOGICAL .or. &
            context%symbols(b_index)%value_kind /= VALUE_LOGICAL) then
            error_msg = 'logical dot_product requires logical arrays'
            return
        end if
        if (context%symbols(a_index)%array_rank /= 1 .or. &
            context%symbols(b_index)%array_rank /= 1) then
            error_msg = 'logical dot_product requires rank-1 arrays'
            return
        end if
        n = context%symbols(a_index)%array_size
        if (n <= 0 .or. n /= context%symbols(b_index)%array_size) then
            error_msg = 'logical dot_product requires conforming non-empty arrays'
            return
        end if

        acc = i32_immediate(context%session, 0_c_int64_t)
        do i = 0, n - 1
            addr_offset = int(i, c_int64_t)
            call load_array_linear_element(context, a_index, addr_offset, &
                                           a_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, b_index, addr_offset, &
                                           b_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_AND, a_elem, &
                                      b_elem, both, error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_OR, acc, both, &
                                      next_acc, error_msg)) return
            acc = next_acc
        end do
        value = acc
        call set_empty(error_msg)
    end subroutine lower_logical_dot_product_intrinsic

    subroutine lower_f32_dot_product_intrinsic(arena, node, context, value, &
                                               error_msg)
        use, intrinsic :: iso_c_binding, only: c_float
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: a_name, b_name
        integer :: a_index, b_index, n, i
        integer(c_int64_t) :: addr_offset
        type(lr_operand_desc_t) :: a_elem, b_elem, product, sum, next_sum

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
        a_index = find_symbol_compat(context, a_name)
        if (a_index <= 0) then
            error_msg = 'dot_product first argument not declared: '//trim(a_name)
            return
        end if
        b_index = find_symbol_compat(context, b_name)
        if (b_index <= 0) then
            error_msg = 'dot_product second argument not declared: '//trim(b_name)
            return
        end if
        if (.not. context%symbols(a_index)%is_array .or. &
            .not. context%symbols(b_index)%is_array) then
            error_msg = 'dot_product requires array arguments'
            return
        end if
        if (context%symbols(a_index)%array_rank /= 1 .or. &
            context%symbols(b_index)%array_rank /= 1) then
            error_msg = 'dot_product requires rank-1 arrays'
            return
        end if
        n = context%symbols(a_index)%array_size
        if (n /= context%symbols(b_index)%array_size .or. n <= 0) then
            error_msg = 'dot_product requires conforming non-empty arrays'
            return
        end if
        sum = liric_f32_immediate(context%session, 0.0_c_float)
        do i = 0, n - 1
            addr_offset = int(i, c_int64_t)
            call load_array_linear_element(context, a_index, addr_offset, &
                                           a_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, b_index, addr_offset, &
                                           b_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FMUL, &
                                            a_elem, b_elem, product, error_msg)) return
            if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, &
                                            sum, product, next_sum, error_msg)) return
            sum = next_sum
        end do
        value = sum
        call set_empty(error_msg)
    end subroutine lower_f32_dot_product_intrinsic

    subroutine lower_f64_dot_product_intrinsic(arena, node, context, value, &
                                               error_msg)
        use, intrinsic :: iso_c_binding, only: c_double
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: a_name, b_name
        integer :: a_index, b_index, n, i
        integer(c_int64_t) :: addr_offset
        type(lr_operand_desc_t) :: a_elem, b_elem, product, sum, next_sum

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
        a_index = find_symbol_compat(context, a_name)
        if (a_index <= 0) then
            error_msg = 'dot_product first argument not declared: '//trim(a_name)
            return
        end if
        b_index = find_symbol_compat(context, b_name)
        if (b_index <= 0) then
            error_msg = 'dot_product second argument not declared: '//trim(b_name)
            return
        end if
        if (.not. context%symbols(a_index)%is_array .or. &
            .not. context%symbols(b_index)%is_array) then
            error_msg = 'dot_product requires array arguments'
            return
        end if
        if (context%symbols(a_index)%array_rank /= 1 .or. &
            context%symbols(b_index)%array_rank /= 1) then
            error_msg = 'dot_product requires rank-1 arrays'
            return
        end if
        n = context%symbols(a_index)%array_size
        if (n /= context%symbols(b_index)%array_size .or. n <= 0) then
            error_msg = 'dot_product requires conforming non-empty arrays'
            return
        end if
        sum = liric_f64_immediate(context%session, 0.0_c_double)
        do i = 0, n - 1
            addr_offset = int(i, c_int64_t)
            call load_array_linear_element(context, a_index, addr_offset, &
                                           a_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, b_index, addr_offset, &
                                           b_elem, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FMUL, &
                                            a_elem, b_elem, product, error_msg)) return
            if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, &
                                            sum, product, next_sum, error_msg)) return
            sum = next_sum
        end do
        value = sum
        call set_empty(error_msg)
    end subroutine lower_f64_dot_product_intrinsic

    subroutine unsupported_intrinsic_error(node, error_msg)
        type(call_or_subscript_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        if (allocated(node%name)) then
            call unsupported_feature_error('scalar intrinsic: '//trim(node%name), &
                                           node%line, node%column, &
                                           'direct LIRIC session supports only '// &
                                           'abs, iabs, min, max, mod, modulo, '// &
                                           'dim, iand, ior, ieor, not, ishft, '// &
                                           'ishftc, ibits, ibset, ibclr, '// &
                                           'bit_size, btest, sign, real, and '// &
                                           'the libm reals (sqrt, exp, log, '// &
                                           'log10, sin, cos, tan, asin, acos, '// &
                                           'atan, atan2, sinh, cosh, tanh, '// &
                                           'asinh, acosh, atanh, erf, erfc, '// &
                                           'gamma, log_gamma, hypot)', error_msg)
        else
            call unsupported_feature_error('scalar intrinsic', node%line, &
                                           node%column, &
                                           'direct LIRIC session supports only '// &
                                           'abs, iabs, min, max, mod, modulo, '// &
                                           'dim, iand, ior, ieor, not, ishft, '// &
                                           'ishftc, ibits, ibset, ibclr, '// &
                                           'bit_size, btest, sign, real, and '// &
                                           'the libm reals (sqrt, exp, log, '// &
                                           'log10, sin, cos, tan, asin, acos, '// &
                                           'atan, atan2, sinh, cosh, tanh, '// &
                                           'asinh, acosh, atanh, erf, erfc, '// &
                                           'gamma, log_gamma, hypot)', error_msg)
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

        symbol_index = find_symbol_compat(context, var_name)
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
