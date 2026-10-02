submodule(session_program_lowering_impl) session_program_lowering_merge
    implicit none
contains

    logical function is_merge_call(arena, node_index, context) result(is_merge)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_merge = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%name)) return
            if (node%is_array_access) return
            if (is_contained_function_reference(node, context)) return
            if (external_procedure_index(context, node%name) > 0) return
            is_merge = same_name(node%name, 'merge')
        end select
    end function is_merge_call

    subroutine resolve_merge_arguments(arena, node, indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(out) :: indices(3)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: keyword
        integer :: i, slot, actual

        indices = 0
        error_msg = 'merge requires tsource, fsource, and mask'
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 3) return
        do i = 1, 3
            call unwrap_intrinsic_arg(arena, node%arg_indices(i), keyword, actual)
            select case (trim(keyword))
            case ('')
                slot = i
            case ('tsource')
                slot = 1
            case ('fsource')
                slot = 2
            case ('mask')
                slot = 3
            case default
                error_msg = 'merge has an unsupported keyword: '//trim(keyword)
                return
            end select
            if (indices(slot) /= 0) then
                error_msg = 'merge has a duplicate argument: '//trim(keyword)
                return
            end if
            indices(slot) = actual
        end do
        if (any(indices <= 0)) return
        call set_empty(error_msg)
    end subroutine resolve_merge_arguments

    recursive integer function merge_value_kind(arena, node, context) result(vk)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: indices(3), source_symbol, integer_kind
        integer(c_int64_t) :: literal_kind
        character(len=:), allocatable :: error_msg, literal_value, literal_type

        vk = VALUE_I32
        call resolve_merge_arguments(arena, node, indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (whole_array_expr_is_logical(arena, indices(1))) then
            vk = VALUE_LOGICAL
            return
        end if
        vk = scalar_real_expr_kind(arena, indices(1), context)
        if (vk /= SCALAR_REAL_NONE) return
        vk = expression_value_kind(arena, indices(1), context, VALUE_I32)
        integer_kind = integer_operand_kind(arena, indices(1), context)
        if (integer_kind > 0) vk = integer_kind
        source_symbol = whole_array_expr_shape_symbol(arena, indices(1), context)
        if (source_symbol > 0) vk = context%symbols(source_symbol)%value_kind
        if (is_literal(arena, indices(1)) .and. vk == VALUE_I32) then
            call get_literal_info(arena, indices(1), literal_value, literal_type, &
                                  error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_literal_kind_number(literal_value, literal_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (literal_kind)
            case (1); vk = VALUE_I8
            case (2); vk = VALUE_I16
            case (8); vk = VALUE_I64
            end select
        end if
    end function merge_value_kind

    subroutine lower_merge_call(arena, node, vk, context, value, error_msg, &
                                target_symbol, linear_index)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: target_symbol, linear_index
        type(lr_operand_desc_t) :: tsource, fsource, mask, condition, selected
        type(lr_operand_desc_t) :: element_index
        integer :: indices(3), mask_symbol, source_vk

        call resolve_merge_arguments(arena, node, indices, error_msg)
        if (len_trim(error_msg) > 0) return
        source_vk = merge_value_kind(arena, node, context)
        if (.not. any(source_vk == &
                      [VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL])) then
            error_msg = 'merge supports default integer, real, real(8), and logical'
            return
        end if
        if (present(target_symbol)) then
            if (source_vk /= vk) then
                error_msg = 'merge array result kind conversion is not supported'
                return
            end if
            if (.not. present(linear_index)) then
                error_msg = 'merge array lowering requires an element index'
                return
            end if
            call lower_array_elementwise_value(arena, indices(1), target_symbol, &
                                               linear_index, context, tsource, &
                                               error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_elementwise_value(arena, indices(2), target_symbol, &
                                               linear_index, context, fsource, &
                                               error_msg)
            if (len_trim(error_msg) > 0) return
            mask_symbol = whole_array_expr_shape_symbol(arena, indices(3), context)
            if (mask_symbol > 0) then
                call ensure_array_shapes_match(context, target_symbol, &
                                               mask_symbol, error_msg)
                if (len_trim(error_msg) > 0) return
                element_index = i32_immediate(context%session, &
                                              int(linear_index, c_int64_t))
                call lower_runtime_reduction_mask(arena, indices(3), &
                                                  element_index, context, mask, &
                                                  error_msg)
            else
                call lower_logical_expression(arena, indices(3), context, mask, &
                                              error_msg)
            end if
        else
            call lower_merge_scalar_source(arena, indices(1), source_vk, context, &
                                           tsource, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_merge_scalar_source(arena, indices(2), source_vk, context, &
                                           fsource, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_logical_expression(arena, indices(3), context, mask, &
                                          error_msg)
        end if
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, mask, &
                                      i32_immediate(context%session, 0_c_int64_t), &
                                      condition, &
                                      error_msg)) return
        call select_value(context, condition, tsource, fsource, selected, error_msg)
        if (len_trim(error_msg) > 0) return
        if (source_vk == vk) then
            value = selected
        else
            call coerce_reduction_operand(context, selected, source_vk, vk, &
                                          value, error_msg)
        end if
    end subroutine lower_merge_call

    subroutine lower_merge_scalar_source(arena, node_index, vk, context, value, &
                                         error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (vk)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
        case default
            call lower_reduction_scalar(arena, node_index, vk, context, value, &
                                        error_msg)
        end select
    end subroutine lower_merge_scalar_source

end submodule session_program_lowering_merge
