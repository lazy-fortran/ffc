    subroutine lower_derived_type_declaration(node, context, derived_type_index, &
                                              error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_parameter) then
            call unsupported_feature_error('derived type parameter declaration', &
                                           node%line, node%column, &
                                           'derived type parameters are not '// &
                                           'supported by direct LIRIC session', &
                                           error_msg)
            return
        end if
        if (node%is_array) then
            call unsupported_feature_error('derived type array declaration', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar derived type variables', &
                                           error_msg)
            return
        end if
        if (node%has_initializer) then
            call unsupported_feature_error('derived type constructor', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived type constructors', &
                                           error_msg)
            return
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_derived_symbol(context, node, node%var_names(i), &
                                           derived_type_index, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_derived_symbol(context, node, node%var_name, &
                                       derived_type_index, error_msg)
        else
            error_msg = 'derived type declaration did not expose a variable name'
        end if
    end subroutine lower_derived_type_declaration

    subroutine define_derived_symbol(context, node, name, derived_type_index, &
                                     error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: component_count
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate derived type declaration: '//trim(name)
            return
        end if
      call grow_symbols(context)

        component_count = context%derived_types(derived_type_index)%component_count
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_DERIVED
        context%symbols(index)%is_derived = .true.
        context%symbols(index)%derived_type_index = derived_type_index
        context%symbols(index)%array_size = component_count

        if (.not. emit_i32_array_alloca(context%session,  &
            component_count, context%symbols(index)%element_address, &
            error_msg)) then
            error_msg = 'derived_alloca['//trim(name)//']: '//error_msg
            return
        end if

        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_derived_symbol

    subroutine lower_derived_component_assignment(arena, node, target, context, &
                                                  value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(component_access_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address

        call lower_i32_expression(arena, node%value_index, context, value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_derived_component_address(arena, target, context, &
                                             component_address, error_msg, &
                                             'derived type component assignment target')
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_store(context%session, value, component_address, &
                                                 error_msg)) then
            error_msg = 'component_store: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_assignment

    subroutine lower_derived_component_load(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address

        call lower_derived_component_address(arena, node, context, &
                                             component_address, error_msg, &
                                             'derived type component expression')
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_load(context%session, component_address, value, &
                                                error_msg)) then
            error_msg = 'component_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_load

    subroutine lower_derived_component_address(arena, node, context, &
                                               component_address, error_msg, &
                                               feature)
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: component_address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: feature
        character(len=:), allocatable :: base_name
        integer :: component_index
        integer :: symbol_index
        integer :: type_index
        type(lr_operand_desc_t) :: offset

        call component_base_name(arena, node%base_expr_index, base_name, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           trim(error_msg), error_msg)
            return
        end if

        symbol_index = find_symbol(context, base_name)
        if (symbol_index <= 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           'component base was not declared: '// &
                                           trim(base_name), error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_derived) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           'component base is not a derived type: '// &
                                           trim(base_name), error_msg)
            return
        end if

        type_index = context%symbols(symbol_index)%derived_type_index
        component_index = find_derived_component(context, type_index, &
                                                 node%component_name)
        if (component_index <= 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           'component is not declared for type '// &
                                           trim(context%derived_types(type_index)%name)// &
                                           ': '//trim(node%component_name), &
                                           error_msg)
            return
        end if

        offset = i32_immediate(context%session, int(component_index - 1, c_int64_t))
        if (.not. emit_i32_array_element_addr(context%session,  &
            context%derived_types(type_index)%component_count, &
            context%symbols(symbol_index)%element_address, offset, &
            component_address, error_msg)) then
            error_msg = 'component_gep: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_address
