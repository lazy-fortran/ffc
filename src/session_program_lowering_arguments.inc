    subroutine store_reference_value(context, symbol_index, value, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (context%symbols(symbol_index)%value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. context%session%emit_i32_store( &
                value, context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_store( &
                context%session, value, context%symbols(symbol_index)%address, &
                error_msg)) return
        case default
            error_msg = 'direct LIRIC session only stores numeric reference arguments'
            return
        end select
        call set_empty(error_msg)
    end subroutine store_reference_value

    subroutine prepare_reference_args(arena, arg_indices, context, default_kind, &
                                      args, copyback_indices, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: default_kind
        type(lr_operand_desc_t), allocatable, intent(out) :: args(:)
        integer, allocatable, intent(out) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        allocate (args(size(arg_indices)))
        allocate (copyback_indices(size(arg_indices)))
        copyback_indices = 0

        do i = 1, size(arg_indices)
            call prepare_reference_arg(arena, arg_indices(i), context, &
                                       default_kind, args(i), &
                                       copyback_indices(i), error_msg)
            if (len_trim(error_msg) > 0) return
        end do

        call set_empty(error_msg)
    end subroutine prepare_reference_args

    subroutine prepare_reference_arg(arena, node_index, context, default_kind, &
                                     arg, copyback_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: default_kind
        type(lr_operand_desc_t), intent(out) :: arg
        integer, intent(out) :: copyback_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: value_kind
        logical :: handled

        copyback_index = 0
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'argument index does not reference an AST node'
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            call prepare_identifier_reference_arg(node, context, arg, &
                                                  copyback_index, handled, &
                                                  error_msg)
            if (len_trim(error_msg) > 0 .or. handled) return
        type is (call_or_subscript_node)
            if (node%is_array_access) then
                call lower_i32_array_element_address(arena, node, context, arg, &
                                                     error_msg)
                return
            end if
        end select

        value_kind = expression_value_kind(arena, node_index, context, default_kind)
        call lower_expression_by_kind(arena, node_index, context, value_kind, value, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return
        call make_reference_argument(context, value_kind, value, arg, error_msg)
    end subroutine prepare_reference_arg

    subroutine prepare_identifier_reference_arg(node, context, arg, copyback_index, &
                                                handled, error_msg)
        type(identifier_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: arg
        integer, intent(out) :: copyback_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index

        copyback_index = 0
        handled = .false.
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            call set_empty(error_msg)
            return
        end if
        handled = .true.
        if (context%symbols(symbol_index)%is_array) then
            call unsupported_feature_error('array argument', node%line, &
                                           node%column, &
                                           'whole-array procedure arguments '// &
                                           'are not supported', error_msg)
            return
        end if
        if (context%symbols(symbol_index)%is_reference) then
            arg = context%symbols(symbol_index)%address
            call set_empty(error_msg)
            return
        end if

        call make_reference_argument(context, &
                                     context%symbols(symbol_index)%value_kind, &
                                     context%symbols(symbol_index)%value, arg, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        copyback_index = symbol_index
        call set_empty(error_msg)
    end subroutine prepare_identifier_reference_arg

    integer function expression_value_kind(arena, node_index, context, &
                                           default_kind) result(value_kind)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: default_kind
        integer :: symbol_index

        value_kind = default_kind
        if (.not. node_exists(arena, node_index)) return

        select type (node => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, node%name)
            if (symbol_index > 0) value_kind = context%symbols(symbol_index)%value_kind
        type is (literal_node)
            if (is_real_literal(node)) then
                value_kind = VALUE_F64
            else if (is_logical_literal(node)) then
                value_kind = VALUE_LOGICAL
            end if
        type is (binary_op_node)
            if (is_f64_expression(arena, node_index, context)) value_kind = VALUE_F64
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                if (is_contained_f64_function(context, node%name)) then
                    value_kind = VALUE_F64
                else if (is_contained_logical_function(context, node%name)) then
                    value_kind = VALUE_LOGICAL
                end if
            end if
        type is (component_access_node)
            value_kind = VALUE_I32
        end select
    end function expression_value_kind

    subroutine lower_expression_by_kind(arena, node_index, context, value_kind, &
                                        value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        end select
    end subroutine lower_expression_by_kind

    subroutine make_reference_argument(context, value_kind, value, address, &
                                       error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(out) :: address
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_F64)
            if (.not. emit_liric_f64_alloca(context%session, address, error_msg)) &
                return
            if (.not. emit_liric_f64_store(context%session, value, address, &
                                           error_msg)) return
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. context%session%emit_i32_alloca(address, error_msg)) return
            if (.not. context%session%emit_i32_store(value, address, error_msg)) &
                return
        case default
            error_msg = 'direct LIRIC session cannot pass this scalar argument'
            return
        end select
        call set_empty(error_msg)
    end subroutine make_reference_argument

    subroutine copy_back_reference_args(context, args, copyback_indices, &
                                        error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: args(:)
        integer, intent(in) :: copyback_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value
        integer :: i

        do i = 1, size(copyback_indices)
            if (copyback_indices(i) <= 0) cycle
            select case (context%symbols(copyback_indices(i))%value_kind)
            case (VALUE_F64)
                if (.not. emit_liric_f64_load(context%session, args(i), value, &
                                              error_msg)) return
            case (VALUE_I32, VALUE_LOGICAL)
                if (.not. context%session%emit_i32_load(args(i), value, &
                                                        error_msg)) return
            case default
                error_msg = 'direct LIRIC session cannot copy back this argument'
                return
            end select
            context%symbols(copyback_indices(i))%value = value
        end do
        call set_empty(error_msg)
    end subroutine copy_back_reference_args
