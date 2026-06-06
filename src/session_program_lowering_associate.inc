    subroutine lower_associate(arena, node, context, error_msg)
        ! Lower an ASSOCIATE construct over scalar selectors. Each associate
        ! name is bound, for the construct scope only, to the value of its
        ! selector expression. The body is then lowered, and the associate-name
        ! bindings are removed at the end of the construct.
        type(ast_arena_t), intent(in) :: arena
        type(associate_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, saved_symbol_count
        logical :: terminated
        type(lr_operand_desc_t) :: value

        call set_empty(error_msg)
        if (.not. allocated(node%associations)) then
            call set_empty(error_msg)
            return
        end if
        saved_symbol_count = context%symbol_count
        do i = 1, size(node%associations)
            call bind_associate_name(arena, node%associations(i), context, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        ! Associate names are scoped to the construct: drop their bindings.
        context%symbol_count = saved_symbol_count
    end subroutine lower_associate

    subroutine bind_associate_name(arena, assoc, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(association_t), intent(in) :: assoc
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: value_kind, idx
        type(lr_operand_desc_t) :: value

        call set_empty(error_msg)
        if (.not. allocated(assoc%name)) then
            error_msg = 'associate name is missing'
            return
        end if
        if (.not. node_exists(arena, assoc%expr_index)) then
            error_msg = 'associate selector expression is missing for '// &
                        trim(assoc%name)
            return
        end if
        value_kind = expression_value_kind(arena, assoc%expr_index, context, &
                                           VALUE_I32)
        select case (value_kind)
        case (VALUE_F64)
            call lower_f64_expression(arena, assoc%expr_index, context, value, &
                                      error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, assoc%expr_index, context, &
                                          value, error_msg)
        case default
            call lower_i32_expression(arena, assoc%expr_index, context, value, &
                                      error_msg)
        end select
        if (len_trim(error_msg) > 0) return

        idx = find_symbol(context, assoc%name)
        if (idx <= 0) then
            call grow_symbols(context)
            idx = context%symbol_count + 1
            context%symbol_count = idx
        end if
        context%symbols(idx)%name = trim(assoc%name)
        context%symbols(idx)%value_kind = value_kind
        context%symbols(idx)%value = value
        context%symbols(idx)%has_address = .false.
        context%symbols(idx)%is_reference = .false.
        context%symbols(idx)%is_derived = .false.
        context%symbols(idx)%is_deferred_character = .false.
        context%symbols(idx)%is_array = .false.
        context%symbols(idx)%is_allocatable = .false.
    end subroutine bind_associate_name
