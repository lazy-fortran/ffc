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

        ! A contiguous rank-1 array-section selector (associate(x => a(lo:hi)))
        ! aliases a view onto the source array's storage, so element access,
        ! size, and writes through the associate name flow to the source.
        if (node_exists(arena, assoc%expr_index)) then
            select type (sel => arena%entries(assoc%expr_index)%node)
            type is (array_slice_node)
                call bind_associate_array_section(arena, sel, assoc%name, &
                                                  context, error_msg)
                return
            type is (component_access_node)
                ! A derived-type component selector (associate(s => a%comp))
                ! aliases the component's storage address, so reads and writes
                ! through the associate name flow to the component.
                call bind_associate_component(arena, sel, assoc%name, context, &
                                              error_msg)
                return
            end select
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

    subroutine bind_associate_array_section(arena, slice_node, assoc_name, &
                                            context, error_msg)
        ! associate(x => a(lo:hi)): bind x to a rank-1, unit-stride view onto
        ! a(lo:hi)'s own storage (the associate name always starts at lower
        ! bound 1, Fortran 2018 11.1.3.3). Element access through x GEPs the
        ! same storage as a, so reads and writes flow through to a. Non-unit
        ! stride and rank-2 sections keep the prior unsupported diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(array_slice_node), intent(in) :: slice_node
        character(len=*), intent(in) :: assoc_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        type(lr_operand_desc_t) :: start_index, view_address
        integer :: idx, value_kind, source_array_size
        integer(c_int64_t) :: extent
        logical :: ok

        call describe_array_section(arena, slice_node, context, info, error_msg)
        if (len_trim(error_msg) > 0) return
        if (info%result_rank /= 1) then
            call unsupported_feature_error('associate array selector', &
                slice_node%line, slice_node%column, &
                'ffc direct-session associate only supports a rank-1 array '// &
                'section selector', error_msg)
            return
        end if
        if (info%section_strides(info%kept_dims(1)) /= 1_c_int64_t) then
            call unsupported_feature_error('associate array selector', &
                slice_node%line, slice_node%column, &
                'ffc direct-session associate only supports a unit-stride '// &
                'array section selector', error_msg)
            return
        end if

        value_kind = context%symbols(info%source_index)%value_kind
        source_array_size = context%symbols(info%source_index)%array_size
        start_index = i32_immediate(context%session, &
            array_section_source_linear_index(info, 0_c_int64_t))
        select case (value_kind)
        case (VALUE_I32)
            ok = emit_i32_array_element_addr(context%session, source_array_size, &
                context%symbols(info%source_index)%element_address, start_index, &
                view_address, error_msg)
        case (VALUE_F32)
            ok = emit_f32_array_element_addr(context%session, source_array_size, &
                context%symbols(info%source_index)%element_address, start_index, &
                view_address, error_msg)
        case (VALUE_F64)
            ok = emit_f64_array_element_addr(context%session, source_array_size, &
                context%symbols(info%source_index)%element_address, start_index, &
                view_address, error_msg)
        case default
            call unsupported_feature_error('associate array selector', &
                slice_node%line, slice_node%column, &
                'ffc direct-session associate array selector supports '// &
                'integer, real(4), and real(8) arrays only', error_msg)
            return
        end select
        if (.not. ok) return

        extent = array_section_total_elements(info)
        idx = associate_symbol_slot(context, assoc_name)
        context%symbols(idx)%name = trim(assoc_name)
        context%symbols(idx)%value_kind = value_kind
        context%symbols(idx)%is_array = .true.
        context%symbols(idx)%array_rank = 1
        context%symbols(idx)%array_size = int(extent)
        context%symbols(idx)%array_lower_bound = 1
        context%symbols(idx)%array_dim_lowers(1) = 1
        context%symbols(idx)%array_dim_sizes(1) = int(extent)
        context%symbols(idx)%element_address = view_address
        context%symbols(idx)%has_address = .true.
        context%symbols(idx)%is_reference = .true.
        context%symbols(idx)%is_pointer = .false.
        context%symbols(idx)%is_parameter = .false.
        context%symbols(idx)%is_derived = .false.
        context%symbols(idx)%is_allocatable = .false.
        call set_empty(error_msg)
    end subroutine bind_associate_array_section

    subroutine bind_associate_component(arena, comp_node, assoc_name, context, &
                                        error_msg)
        ! associate(s => a%comp): bind s to the component's own storage
        ! address, so reads and writes through s flow to a%comp.
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: comp_node
        character(len=*), intent(in) :: assoc_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address
        integer :: comp_kind, idx

        comp_kind = derived_component_access_kind(arena, comp_node, context)
        if (comp_kind == VALUE_CHARACTER) then
            call unsupported_feature_error('associate component selector', &
                comp_node%line, comp_node%column, &
                'ffc direct-session associate does not support a character '// &
                'component selector', error_msg)
            return
        end if
        call lower_derived_component_address(arena, comp_node, context, &
            component_address, error_msg, 'associate component selector')
        if (len_trim(error_msg) > 0) return

        idx = associate_symbol_slot(context, assoc_name)
        context%symbols(idx)%name = trim(assoc_name)
        context%symbols(idx)%value_kind = comp_kind
        context%symbols(idx)%address = component_address
        context%symbols(idx)%has_address = .true.
        context%symbols(idx)%is_reference = .true.
        context%symbols(idx)%is_pointer = .false.
        context%symbols(idx)%is_parameter = .false.
        context%symbols(idx)%is_derived = .false.
        context%symbols(idx)%is_deferred_character = .false.
        context%symbols(idx)%is_array = .false.
        context%symbols(idx)%is_allocatable = .false.
        call set_empty(error_msg)
    end subroutine bind_associate_component
