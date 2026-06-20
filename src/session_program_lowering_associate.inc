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
        integer :: value_kind, idx, src_idx
        character(len=:), allocatable :: sel_name, name_err
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

        ! A bare-variable selector aliases the variable: copy the existing
        ! symbol so its kind, array shape, and derived-type metadata carry
        ! through reads under the associate name (Fortran 2018 11.1.3.3).
        if (is_identifier(arena, assoc%expr_index)) then
            call get_identifier_name(arena, assoc%expr_index, sel_name, name_err)
            src_idx = find_symbol(context, sel_name)
            if (src_idx > 0) then
                idx = associate_symbol_slot(context, assoc%name)
                context%symbols(idx) = context%symbols(src_idx)
                context%symbols(idx)%name = trim(assoc%name)
                return
            end if
        end if

        ! Array-valued expression selectors (e.g. z => x + y) need a temporary
        ! array and shape tracking, which the scalar value path cannot model.
        ! Reject them cleanly instead of silently aliasing only one element.
        if (associate_selector_is_array(arena, assoc%expr_index, context)) then
            call unsupported_feature_error('associate array selector', &
                get_node_line(arena, assoc%expr_index), &
                get_node_column(arena, assoc%expr_index), &
                'ffc direct-session associates only scalar expression '// &
                'selectors; array-expression aliases are not yet supported', &
                error_msg)
            return
        end if

        ! Otherwise the selector is an expression: evaluate it once and bind the
        ! associate name to that value for the construct scope.
        value_kind = expression_value_kind(arena, assoc%expr_index, context, &
                                           VALUE_I32)
        select case (value_kind)
        case (VALUE_F32, VALUE_F64, VALUE_LOGICAL, VALUE_I32)
            call lower_expression_by_kind(arena, assoc%expr_index, context, &
                                          value_kind, value, error_msg)
        case default
            error_msg = 'associate selector for '//trim(assoc%name)// &
                        ' uses an unsupported scalar type'
            return
        end select
        if (len_trim(error_msg) > 0) return

        idx = associate_symbol_slot(context, assoc%name)
        context%symbols(idx)%name = trim(assoc%name)
        context%symbols(idx)%value_kind = value_kind
        context%symbols(idx)%value = value
        context%symbols(idx)%has_address = .false.
        context%symbols(idx)%is_reference = .false.
        context%symbols(idx)%is_parameter = .false.
        context%symbols(idx)%is_derived = .false.
        context%symbols(idx)%is_deferred_character = .false.
        context%symbols(idx)%is_array = .false.
        context%symbols(idx)%is_allocatable = .false.
    end subroutine bind_associate_name

    recursive logical function associate_selector_is_array(arena, node_index, &
                                                           context) result(is_arr)
        ! An expression is array-valued when any identifier leaf names an array
        ! symbol. Binary operators propagate array shape from either operand;
        ! a whole-array reference (no subscript) is array-valued as well.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: id_name, id_err
        integer :: bin_left, bin_right, bin_line, bin_col, sym
        character(len=:), allocatable :: bin_op, bin_err

        is_arr = .false.
        if (.not. node_exists(arena, node_index)) return

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            is_arr = associate_selector_is_array(arena, bin_left, context) .or. &
                     associate_selector_is_array(arena, bin_right, context)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            sym = find_symbol(context, id_name)
            if (sym > 0) is_arr = context%symbols(sym)%is_array
        end if
    end function associate_selector_is_array

    integer function associate_symbol_slot(context, name) result(idx)
        ! Find an existing symbol slot for an associate name, or append a fresh
        ! one. Reuse keeps a same-named outer symbol shadowed for the construct.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name

        idx = find_symbol(context, name)
        if (idx <= 0) then
            call grow_symbols(context)
            idx = context%symbol_count + 1
            context%symbol_count = idx
        end if
    end function associate_symbol_slot
