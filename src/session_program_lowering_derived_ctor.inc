    subroutine lower_derived_whole_assignment(arena, node, context, &
                                              symbol_index, handled, error_msg)
        ! Whole-derived scalar assignment: a structure constructor RHS
        ! (y = t(...)) or a plain derived variable copy (y = x). Sets handled
        ! only when the form is lowered; otherwise the caller reports the
        ! existing unsupported-assignment diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index, src_index
        character(len=:), allocatable :: id_name, id_err

        handled = .false.
        call set_empty(error_msg)
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0) return
        if (context%symbols(symbol_index)%is_array) return
        if (.not. node_exists(arena, node%value_index)) return

        if (is_derived_constructor_call(arena, node%value_index, context, &
                                        type_index)) then
            if (constructor_uses_user_routine(context, &
                    context%derived_types(type_index)%name)) return
            ! y = t() re-applies the type's default initialisation, covering
            ! types with nested components the argument path cannot construct.
            if (constructor_call_is_argumentless(arena, node%value_index)) then
                call store_type_default_slots(context, symbol_index, &
                    total_component_slots(context, type_index), type_index, 0, &
                    error_msg)
                if (len_trim(error_msg) == 0) handled = .true.
                return
            end if
            call lower_derived_constructor_assignment(arena, node%value_index, &
                context, symbol_index, type_index, handled, error_msg)
            return
        end if

        if (is_identifier(arena, node%value_index)) then
            call get_identifier_name(arena, node%value_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            src_index = find_symbol(context, id_name)
            if (src_index <= 0) return
            if (.not. context%symbols(src_index)%is_derived) return
            if (context%symbols(src_index)%is_array) return
            if (context%symbols(src_index)%derived_type_index /= type_index) return
            call lower_derived_scalar_copy(context, symbol_index, src_index, &
                                           error_msg)
            if (len_trim(error_msg) == 0) handled = .true.
        end if
    end subroutine lower_derived_whole_assignment

    subroutine lower_derived_scalar_copy(context, dest_index, src_index, error_msg)
        ! Copy a derived scalar slot-by-slot. Raw i32 slots preserve every
        ! component kind (int, real f32/f64 spans, logical, character bytes,
        ! nested aggregates) since the layout is identical for both operands.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: dest_index, src_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index, slots, s
        type(lr_operand_desc_t) :: src_addr, dst_addr, val

        call set_empty(error_msg)
        type_index = context%symbols(dest_index)%derived_type_index
        slots = total_component_slots(context, type_index)
        do s = 0, slots - 1
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(src_index)%array_size, &
                    context%symbols(src_index)%element_address, &
                    i32_immediate(context%session, int(s, c_int64_t)), &
                    src_addr, error_msg)) return
            if (.not. emit_i32_load(context%session, src_addr, val, error_msg)) &
                return
            if (.not. emit_i32_array_element_addr(context%session, &
                    context%symbols(dest_index)%array_size, &
                    context%symbols(dest_index)%element_address, &
                    i32_immediate(context%session, int(s, c_int64_t)), &
                    dst_addr, error_msg)) return
            if (.not. emit_i32_store(context%session, val, dst_addr, error_msg)) &
                return
        end do
    end subroutine lower_derived_scalar_copy

    subroutine lower_derived_constructor_assignment(arena, call_index, context, &
            symbol_index, type_index, handled, error_msg)
        ! y = t(a1, ..., an): store each provided component value into the
        ! target's slot layout; components omitted from the constructor keep
        ! their compile-time default. Only scalar integer/real/logical
        ! components are handled; any other component leaves handled false so
        ! the caller falls back to the unsupported-constructor diagnostic.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: n_comp, c
        integer, allocatable :: comp_value_index(:)

        handled = .false.
        call set_empty(error_msg)
        n_comp = context%derived_types(type_index)%component_count
        if (n_comp <= 0) return
        do c = 1, n_comp
            if (.not. constructor_component_storeable(context, type_index, c)) return
        end do

        call map_constructor_args(arena, call_index, context, type_index, &
                                  comp_value_index)
        if (.not. allocated(comp_value_index)) return
        do c = 1, n_comp
            if (comp_value_index(c) <= 0 .and. &
                .not. context%derived_types(type_index)%component_has_default(c)) &
                return
        end do

        do c = 1, n_comp
            if (comp_value_index(c) > 0) then
                call emit_constructor_component(arena, context, symbol_index, &
                    type_index, c, comp_value_index(c), error_msg)
            else
                call emit_constructor_default(context, symbol_index, type_index, &
                    c, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do
        handled = .true.
    end subroutine lower_derived_constructor_assignment

    logical function constructor_component_storeable(context, type_index, &
            comp_index)
        ! A component the structure-constructor path can store: a scalar
        ! integer, real, or logical living in its own slots (no nested derived,
        ! array, character, or c_ptr components).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index

        constructor_component_storeable = .false.
        if (context%derived_types(type_index)%component_type_index(comp_index) /= 0) &
            return
        if (context%derived_types(type_index)%component_array_size(comp_index) /= 1) &
            return
        select case (component_kind(context, type_index, comp_index))
        case (VALUE_I32, VALUE_F32, VALUE_F64, VALUE_LOGICAL)
            constructor_component_storeable = .true.
        end select
    end function constructor_component_storeable

    subroutine map_constructor_args(arena, call_index, context, type_index, &
            comp_value_index)
        ! Map constructor actual arguments to component slots. Positional
        ! arguments fill components in declaration order; keyword arguments
        ! (name=value) target the named component. On any inconsistency the
        ! result stays unallocated so the caller declines the constructor.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, allocatable, intent(out) :: comp_value_index(:)
        integer :: n_comp, a, pos, ci, value_index
        logical :: is_kw
        character(len=:), allocatable :: kw_name

        n_comp = context%derived_types(type_index)%component_count
        allocate(comp_value_index(n_comp))
        comp_value_index = 0
        pos = 0
        select type (cn => arena%entries(call_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(cn%arg_indices)) return
            do a = 1, size(cn%arg_indices)
                call classify_constructor_arg(arena, cn%arg_indices(a), is_kw, &
                                              kw_name, value_index)
                if (is_kw) then
                    ci = find_derived_component(context, type_index, kw_name)
                    if (ci <= 0) then
                        deallocate(comp_value_index); return
                    end if
                else
                    pos = pos + 1
                    ci = pos
                    if (ci > n_comp) then
                        deallocate(comp_value_index); return
                    end if
                end if
                if (comp_value_index(ci) /= 0) then
                    deallocate(comp_value_index); return
                end if
                if (.not. is_safe_constructor_value(arena, value_index)) then
                    deallocate(comp_value_index); return
                end if
                comp_value_index(ci) = value_index
            end do
        class default
            deallocate(comp_value_index)
        end select
    end subroutine map_constructor_args

    subroutine classify_constructor_arg(arena, arg_index, is_kw, kw_name, &
            value_index)
        ! A constructor actual argument is either positional (value_index is
        ! the argument itself) or keyword name=value (a binary '=' whose left
        ! identifier names the component and whose right operand is the value).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        logical, intent(out) :: is_kw
        character(len=:), allocatable, intent(out) :: kw_name
        integer, intent(out) :: value_index
        character(len=:), allocatable :: op, id_err
        integer :: left_index, right_index, ln, col

        is_kw = .false.
        call set_empty(kw_name)
        value_index = arg_index
        if (.not. is_binary_op(arena, arg_index)) return
        call get_binary_op_info(arena, arg_index, op, left_index, right_index, &
                                ln, col, id_err)
        if (len_trim(id_err) > 0) return
        if (op /= '=') return
        if (.not. is_identifier(arena, left_index)) return
        call get_identifier_name(arena, left_index, kw_name, id_err)
        if (len_trim(id_err) > 0) then
            call set_empty(kw_name)
            return
        end if
        is_kw = .true.
        value_index = right_index
    end subroutine classify_constructor_arg

    logical function is_safe_constructor_value(arena, node_index)
        ! A constructor value the scalar-component path can evaluate directly.
        ! Anything else (e.g. a keyword-argument wrapper the frontend keeps as a
        ! non-value node) makes the caller decline the constructor rather than
        ! risk mapping a value to the wrong component.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_safe_constructor_value = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            is_safe_constructor_value = .true.
            return
        end if
        if (is_literal(arena, node_index)) then
            is_safe_constructor_value = .true.
            return
        end if
        if (is_binary_op(arena, node_index)) then
            is_safe_constructor_value = .true.
            return
        end if
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            is_safe_constructor_value = .true.
        type is (component_access_node)
            is_safe_constructor_value = .true.
        end select
    end function is_safe_constructor_value

    subroutine emit_constructor_component(arena, context, symbol_index, &
            type_index, comp_index, value_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        integer, intent(in) :: value_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr, val
        integer :: comp_kind

        comp_kind = component_kind(context, type_index, comp_index)
        call emit_component_slot_address(context, symbol_index, type_index, &
                                         comp_index, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_component_rhs_value(arena, value_index, comp_kind, context, &
                                       val, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_typed_component_store(context, comp_kind, val, addr, error_msg)
    end subroutine emit_constructor_component

    subroutine emit_constructor_default(context, symbol_index, type_index, &
            comp_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer(c_int64_t) :: default_value

        call emit_component_slot_address(context, symbol_index, type_index, &
                                         comp_index, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        default_value = &
            context%derived_types(type_index)%component_default_value(comp_index)
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, default_value), addr, error_msg)) return
    end subroutine emit_constructor_default

    subroutine emit_component_slot_address(context, symbol_index, type_index, &
            comp_index, addr, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, &
                    int(component_start_slot(context, type_index, comp_index), &
                        c_int64_t)), addr, error_msg)) then
            error_msg = 'constructor_component_gep: '//error_msg
        end if
    end subroutine emit_component_slot_address

    logical function constructor_uses_user_routine(context, name)
        ! True when name also denotes a user procedure (a generic interface or
        ! a contained function), so t(...) must dispatch to that routine rather
        ! than perform default structure construction.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        constructor_uses_user_routine = .true.
        do i = 1, context%generic_count
            if (same_name(context%generics(i)%generic_name, name)) return
        end do
        if (contained_function_kind(context, name) /= 0) return
        constructor_uses_user_routine = .false.
    end function constructor_uses_user_routine
