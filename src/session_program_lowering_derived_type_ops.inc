    integer function total_component_slots(context, type_index)
        ! Total i32 slots for a derived instance: scalar components take one
        ! slot, array components take their element count.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer :: c

        total_component_slots = 0
        do c = 1, context%derived_types(type_index)%component_count
            total_component_slots = total_component_slots + &
                context%derived_types(type_index)%component_array_size(c)
        end do
    end function total_component_slots

    integer function component_start_slot(context, type_index, comp_index)
        ! Starting slot of component comp_index = sum of the sizes before it.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        integer :: c

        component_start_slot = 0
        do c = 1, comp_index - 1
            component_start_slot = component_start_slot + &
                context%derived_types(type_index)%component_array_size(c)
        end do
    end function component_start_slot

    subroutine init_derived_component_defaults(context, symbol_index, &
                                               type_index, error_msg)
        ! Store each scalar component's default initialiser into a freshly
        ! declared derived instance, at the component's starting slot.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: c
        integer :: slots
        type(lr_operand_desc_t) :: addr

        call set_empty(error_msg)
        slots = total_component_slots(context, type_index)
        if (.not. allocated(context%derived_types(type_index)% &
                            component_has_default)) return
        do c = 1, context%derived_types(type_index)%component_count
            if (.not. context%derived_types(type_index)%component_has_default(c)) &
                cycle
            if (.not. emit_i32_array_element_addr(context%session, slots, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, &
                        int(component_start_slot(context, type_index, c), &
                            c_int64_t)), addr, error_msg)) return
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, &
                        context%derived_types(type_index)%component_default_value(c)), &
                    addr, error_msg)) return
        end do
    end subroutine init_derived_component_defaults

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
        if (node%has_initializer) then
            call unsupported_feature_error('derived type constructor', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived type constructors', &
                                           error_msg)
            return
        end if
        if (node%is_array) then
            call lower_derived_array_declaration(node, context, &
                                                 derived_type_index, error_msg)
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

    subroutine lower_derived_array_declaration(node, context, derived_type_index, &
                                               error_msg)
        ! type(t) :: arr(N): storage is N * slots(t) i32 slots.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: lower_bound, n_elems, slots, index

        if (node%is_multi_declaration) then
            error_msg = 'multi-variable derived array declarations are not supported'
            return
        end if
        if (.not. allocated(node%var_name)) then
            error_msg = 'derived array declaration did not expose a variable name'
            return
        end if
        name = node%var_name
        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate derived array declaration: '//trim(name)
            return
        end if
        call get_array_bounds(node, context, lower_bound, n_elems, error_msg)
        if (len_trim(error_msg) > 0) return
        if (n_elems <= 0) then
            call unsupported_feature_error('derived array declaration', node%line, &
                node%column, 'derived arrays need a positive compile-time size', &
                error_msg)
            return
        end if

        call grow_symbols(context)
        slots = total_component_slots(context, derived_type_index)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_DERIVED
        context%symbols(index)%is_derived = .true.
        context%symbols(index)%is_array = .true.
        context%symbols(index)%derived_type_index = derived_type_index
        context%symbols(index)%array_size = n_elems * slots
        if (.not. emit_i32_array_alloca(context%session, n_elems * slots, &
                context%symbols(index)%element_address, error_msg)) then
            error_msg = 'derived_array_alloca['//trim(name)//']: '//error_msg
            return
        end if
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine lower_derived_array_declaration

    subroutine define_derived_symbol(context, node, name, derived_type_index, &
                                     error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: slot_count
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate derived type declaration: '//trim(name)
            return
        end if
      call grow_symbols(context)

        slot_count = total_component_slots(context, derived_type_index)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_DERIVED
        context%symbols(index)%is_derived = .true.
        context%symbols(index)%derived_type_index = derived_type_index
        context%symbols(index)%array_size = slot_count

        if (.not. emit_i32_array_alloca(context%session,  &
            slot_count, context%symbols(index)%element_address, &
            error_msg)) then
            error_msg = 'derived_alloca['//trim(name)//']: '//error_msg
            return
        end if

        call init_derived_component_defaults(context, index, derived_type_index, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return

        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_derived_symbol

    subroutine derived_component_element_address(arena, access_index, &
            sub_index, context, addr, error_msg)
        ! Address of obj%comp(i): slot = start_slot(comp) + (i - 1).
        ! access_index points to the component_access node (obj%comp); the
        ! enclosing subscript supplies sub_index.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        integer, intent(in) :: sub_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name, comp_name
        integer :: base_index, symbol_index, type_index, component_index
        type(lr_operand_desc_t) :: sub_val, zero_based, slot

        if (.not. node_exists(arena, access_index)) then
            error_msg = 'component element base is not an AST node'
            return
        end if
        select type (acc => arena%entries(access_index)%node)
        type is (component_access_node)
            base_index = acc%base_expr_index
            comp_name = acc%component_name
        class default
            error_msg = 'array component access requires obj%comp(i) form'
            return
        end select

        call component_base_name(arena, base_index, base_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, base_name)
        if (symbol_index <= 0) then
            error_msg = 'component base was not declared: '//trim(base_name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_derived) then
            error_msg = 'component base is not a derived type: '//trim(base_name)
            return
        end if
        type_index = context%symbols(symbol_index)%derived_type_index
        component_index = find_derived_component(context, type_index, comp_name)
        if (component_index <= 0) then
            error_msg = 'component is not declared for type '// &
                trim(context%derived_types(type_index)%name)//': '//trim(comp_name)
            return
        end if
        if (context%derived_types(type_index)%component_array_size(component_index) &
            <= 1) then
            error_msg = 'component is not an array: '//trim(comp_name)
            return
        end if

        call lower_i32_expression(arena, sub_index, context, sub_val, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, sub_val, &
                i32_immediate(context%session, 1_c_int64_t), zero_based, &
                error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, zero_based, &
                i32_immediate(context%session, &
                    int(component_start_slot(context, type_index, component_index), &
                        c_int64_t)), slot, error_msg)) return
        if (.not. emit_i32_array_element_addr(context%session, &
                total_component_slots(context, type_index), &
                context%symbols(symbol_index)%element_address, slot, addr, &
                error_msg)) then
            error_msg = 'component_element_gep: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine derived_component_element_address

    subroutine lower_derived_component_element_assignment(arena, node, target, &
                                                          context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, addr

        if (.not. allocated(target%arg_indices) .or. &
            size(target%arg_indices) /= 1) then
            call unsupported_feature_error('derived component element', &
                node%line, node%column, 'only single-index array component '// &
                'access is supported', error_msg)
            return
        end if
        call lower_i32_expression(arena, node%value_index, context, value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call derived_component_element_address(arena, target%base_expr_index, &
            target%arg_indices(1), context, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_store(context%session, value, addr, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_derived_component_element_assignment

    subroutine lower_derived_component_element_load(arena, node, context, value, &
                                                    error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr

        if (.not. allocated(node%arg_indices) .or. &
            size(node%arg_indices) /= 1) then
            call unsupported_feature_error('derived component element', &
                node%line, node%column, 'only single-index array component '// &
                'access is supported', error_msg)
            return
        end if
        call derived_component_element_address(arena, node%base_expr_index, &
            node%arg_indices(1), context, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_load(context%session, addr, value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_derived_component_element_load

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

    subroutine resolve_component_base(arena, base_index, context, symbol_index, &
                                      type_index, element_base, error_msg)
        ! Resolve the base of a component access to (symbol, type, element slot
        ! base). For a plain derived scalar `x` the element base is 0; for an
        ! element of a derived array `arr(i)` it is (i - 1) * slots-per-element.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: base_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: symbol_index
        integer, intent(out) :: type_index
        type(lr_operand_desc_t), intent(out) :: element_base
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: sub_val, zero_based
        integer :: slots

        symbol_index = 0
        type_index = 0
        if (.not. node_exists(arena, base_index)) then
            error_msg = 'component base index does not reference an AST node'
            return
        end if
        select type (base => arena%entries(base_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, base%name)
            if (symbol_index <= 0) then
                error_msg = 'component base was not declared: '//trim(base%name)
                return
            end if
            element_base = i32_immediate(context%session, 0_c_int64_t)
        type is (call_or_subscript_node)
            if (.not. allocated(base%name) .or. .not. allocated(base%arg_indices)) &
                then
                error_msg = 'array component base requires arr(i) form'
                return
            end if
            if (size(base%arg_indices) /= 1) then
                error_msg = 'only one-dimensional derived arrays are supported'
                return
            end if
            symbol_index = find_symbol(context, base%name)
            if (symbol_index <= 0) then
                error_msg = 'component base was not declared: '//trim(base%name)
                return
            end if
            if (.not. context%symbols(symbol_index)%is_array) then
                error_msg = 'component base is not an array: '//trim(base%name)
                return
            end if
            type_index = context%symbols(symbol_index)%derived_type_index
            slots = total_component_slots(context, type_index)
            call lower_i32_expression(arena, base%arg_indices(1), context, &
                                      sub_val, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_SUB, sub_val, &
                    i32_immediate(context%session, 1_c_int64_t), zero_based, &
                    error_msg)) return
            if (.not. emit_i32_binary(context%session, LR_OP_MUL, zero_based, &
                    i32_immediate(context%session, int(slots, c_int64_t)), &
                    element_base, error_msg)) return
        class default
            error_msg = 'nested component access is not supported'
            return
        end select
        if (.not. context%symbols(symbol_index)%is_derived) then
            error_msg = 'component base is not a derived type'
            return
        end if
        type_index = context%symbols(symbol_index)%derived_type_index
        call set_empty(error_msg)
    end subroutine resolve_component_base

    subroutine lower_derived_component_address(arena, node, context, &
                                               component_address, error_msg, &
                                               feature)
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: component_address
        character(len=*), intent(in) :: feature
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: component_index
        integer :: symbol_index
        integer :: type_index
        type(lr_operand_desc_t) :: element_base, slot

        call resolve_component_base(arena, node%base_expr_index, context, &
                                    symbol_index, type_index, element_base, &
                                    error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           trim(error_msg), error_msg)
            return
        end if

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

        if (.not. emit_i32_binary(context%session, LR_OP_ADD, element_base, &
                i32_immediate(context%session, &
                    int(component_start_slot(context, type_index, component_index), &
                        c_int64_t)), slot, error_msg)) return
        if (.not. emit_i32_array_element_addr(context%session,  &
            context%symbols(symbol_index)%array_size, &
            context%symbols(symbol_index)%element_address, slot, &
            component_address, error_msg)) then
            error_msg = 'component_gep: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_address
