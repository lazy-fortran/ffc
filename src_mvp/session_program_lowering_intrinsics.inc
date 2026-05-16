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
        case default
            call unsupported_intrinsic_error(node, error_msg)
        end select
    end subroutine lower_i32_intrinsic_call

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

        zero = context%session%i32_immediate(0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGE, arg, zero, &
                                      condition, error_msg)) return
        if (.not. context%session%emit_i32_binary(LR_OP_SUB, zero, arg, &
                                                  negative, error_msg)) return
        call select_value(context, condition, arg, negative, value, error_msg)
    end subroutine lower_i32_abs_intrinsic

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

        if (.not. require_intrinsic_arg_count(node, 2, MAX_SYMBOLS, &
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

        if (.not. require_intrinsic_arg_count(node, 2, MAX_SYMBOLS, &
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

    subroutine unsupported_intrinsic_error(node, error_msg)
        type(call_or_subscript_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        if (allocated(node%name)) then
            error_msg = 'unsupported scalar intrinsic: '//trim(node%name)
        else
            error_msg = 'unsupported scalar intrinsic'
        end if
    end subroutine unsupported_intrinsic_error
