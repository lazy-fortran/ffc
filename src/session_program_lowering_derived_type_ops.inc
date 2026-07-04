    integer function component_slot_width(context, type_index, comp_index)
        ! i32 slots one element of a component occupies: an allocatable component
        ! holds an 8-byte data pointer (two slots); f64 spans two 4-byte slots;
        ! every other supported scalar kind fits in one.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index

        component_slot_width = 1
        if (component_is_alloc_array(context, type_index, comp_index)) then
            component_slot_width = 4
            return
        end if
        if (component_is_allocatable(context, type_index, comp_index)) then
            component_slot_width = 2
            return
        end if
        if (.not. allocated(context%derived_types(type_index)% &
                            component_value_kind)) return
        if (context%derived_types(type_index)%component_value_kind(comp_index) == &
            VALUE_F64) component_slot_width = 2
    end function component_slot_width

    logical function component_is_alloc_array(context, type_index, comp_index)
        ! True for a rank-1 allocatable array component (inline descriptor).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index

        component_is_alloc_array = .false.
        if (.not. allocated(context%derived_types(type_index)% &
                            component_is_alloc_array)) return
        component_is_alloc_array = &
            context%derived_types(type_index)%component_is_alloc_array(comp_index)
    end function component_is_alloc_array

    logical function component_is_allocatable(context, type_index, comp_index)
        ! True for a scalar allocatable component (descriptor-in-struct pointer).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index

        component_is_allocatable = .false.
        if (.not. allocated(context%derived_types(type_index)% &
                            component_is_allocatable)) return
        component_is_allocatable = &
            context%derived_types(type_index)%component_is_allocatable(comp_index)
    end function component_is_allocatable

    integer function component_kind(context, type_index, comp_index)
        ! Scalar value kind of a component; VALUE_I32 when not yet recorded.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index

        component_kind = VALUE_I32
        if (.not. allocated(context%derived_types(type_index)% &
                            component_value_kind)) return
        component_kind = &
            context%derived_types(type_index)%component_value_kind(comp_index)
    end function component_kind

    integer function total_component_slots(context, type_index)
        ! Total i32 slots for a derived instance: each component contributes
        ! element_count * slot_width(kind).
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer :: c

        total_component_slots = 0
        do c = 1, context%derived_types(type_index)%component_count
            total_component_slots = total_component_slots + &
                context%derived_types(type_index)%component_array_size(c) * &
                component_slot_width(context, type_index, c)
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
                context%derived_types(type_index)%component_array_size(c) * &
                component_slot_width(context, type_index, c)
        end do
    end function component_start_slot

    subroutine init_derived_component_defaults(context, symbol_index, &
                                               type_index, error_msg)
        ! Store each scalar component's default initialiser into a freshly
        ! declared derived instance, recursing through nested derived components
        ! so an inner type's own component defaults materialise inline.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: outer_slots

        outer_slots = total_component_slots(context, type_index)
        call store_type_default_slots(context, symbol_index, outer_slots, &
                                      type_index, 0, error_msg)
    end subroutine init_derived_component_defaults

    recursive subroutine store_type_default_slots(context, symbol_index, &
                                                  outer_slots, type_index, &
                                                  base_slot, error_msg)
        ! Emit i32 stores for every defaulted scalar leaf of type_index into the
        ! outer instance's slot storage, base_slot offsetting a nested type into
        ! its inline span.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: outer_slots
        integer, intent(in) :: type_index
        integer, intent(in) :: base_slot
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: c, nested, abs_slot
        type(lr_operand_desc_t) :: addr

        call set_empty(error_msg)
        if (.not. allocated(context%derived_types(type_index)% &
                            component_type_index)) return
        do c = 1, context%derived_types(type_index)%component_count
            abs_slot = base_slot + component_start_slot(context, type_index, c)
            nested = context%derived_types(type_index)%component_type_index(c)
            if (nested /= 0) then
                call store_type_default_slots(context, symbol_index, outer_slots, &
                                              nested, abs_slot, error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (component_is_alloc_array(context, type_index, c)) then
                call null_allocatable_component_slots(context, symbol_index, &
                                                      outer_slots, abs_slot, 4, &
                                                      error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (component_is_allocatable(context, type_index, c)) then
                call null_allocatable_component_slots(context, symbol_index, &
                                                      outer_slots, abs_slot, 2, &
                                                      error_msg)
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            if (.not. allocated(context%derived_types(type_index)% &
                                component_has_default)) cycle
            if (.not. context%derived_types(type_index)%component_has_default(c)) &
                cycle
            if (.not. emit_i32_array_element_addr(context%session, outer_slots, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, int(abs_slot, c_int64_t)), &
                    addr, error_msg)) return
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, &
                        context%derived_types(type_index)%component_default_value(c)), &
                    addr, error_msg)) return
        end do
    end subroutine store_type_default_slots

    subroutine null_allocatable_component_slots(context, symbol_index, outer_slots, &
                                                base_slot, n_slots, error_msg)
        ! Zero the n_slots i32 slots that hold an allocatable component's inline
        ! descriptor (a scalar data pointer spans two slots; a rank-1 array
        ! descriptor spans four), marking it unallocated in a freshly declared
        ! instance.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: outer_slots
        integer, intent(in) :: base_slot
        integer, intent(in) :: n_slots
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer :: k

        do k = 0, n_slots - 1
            if (.not. emit_i32_array_element_addr(context%session, outer_slots, &
                    context%symbols(symbol_index)%element_address, &
                    i32_immediate(context%session, int(base_slot + k, c_int64_t)), &
                    addr, error_msg)) return
            if (.not. emit_i32_store(context%session, &
                    i32_immediate(context%session, 0_c_int64_t), addr, error_msg)) &
                return
        end do
        call set_empty(error_msg)
    end subroutine null_allocatable_component_slots

    recursive subroutine fill_type_default_slots(context, type_index, base_slot, &
                                                 slots)
        ! Write type_index's defaulted scalar leaves into a static slot array
        ! (1-based), recursing through nested derived components. Slots without a
        ! default keep their existing (zero) value.
        use, intrinsic :: iso_c_binding, only: c_int32_t
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: base_slot
        integer(c_int32_t), intent(inout) :: slots(:)
        integer :: c, nested, abs_slot

        if (.not. allocated(context%derived_types(type_index)% &
                            component_type_index)) return
        do c = 1, context%derived_types(type_index)%component_count
            abs_slot = base_slot + component_start_slot(context, type_index, c)
            nested = context%derived_types(type_index)%component_type_index(c)
            if (nested /= 0) then
                call fill_type_default_slots(context, nested, abs_slot, slots)
                cycle
            end if
            if (.not. allocated(context%derived_types(type_index)% &
                                component_has_default)) cycle
            if (.not. context%derived_types(type_index)%component_has_default(c)) &
                cycle
            slots(abs_slot + 1) = int(context%derived_types(type_index)% &
                                      component_default_value(c), c_int32_t)
        end do
    end subroutine fill_type_default_slots

    subroutine lower_derived_type_declaration(node, context, derived_type_index, &
                                              error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        ! A scalar derived PARAMETER must carry a constructor initialiser
        ! (t(...)); it lowers exactly like an initialised scalar derived local,
        ! materialising the constant instance into read-only-in-practice slots.
        ! Array or uninitialised derived parameters stay unsupported.
        if (node%is_parameter) then
            if (node%is_array .or. .not. node%has_initializer) then
                call unsupported_feature_error( &
                    'derived type parameter declaration', &
                    node%line, node%column, &
                    'direct LIRIC session supports only a scalar derived '// &
                    'parameter with a constructor initialiser', error_msg)
                return
            end if
        end if
        if (node%has_initializer .and. node%is_array) then
            call unsupported_feature_error('derived type array initializer', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived array initializers', &
                                           error_msg)
            return
        end if
        if (node%is_array) then
            call lower_derived_array_declaration(node, context, &
                                                 derived_type_index, error_msg)
            return
        end if
        ! A scalar derived initializer (v = t(...) or v = other) reuses the
        ! runtime whole-derived assignment path: define the instance (alloca +
        ! component defaults) then store the initial value into its slots.
        if (node%has_initializer .and. allocated(node%var_name) .and. &
            .not. node%is_multi_declaration) then
            call define_derived_symbol(context, node, node%var_name, &
                                       derived_type_index, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_derived_scalar_initializer(node, context, &
                derived_type_index, error_msg)
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

    subroutine lower_derived_scalar_initializer(node, context, type_index, &
                                                error_msg)
        ! Store a scalar derived variable's declared initial value. A valid
        ! derived initializer is a constant structure constructor t(...); the
        ! runtime whole-derived constructor lowering materialises it into the
        ! freshly declared slots. Any other form surfaces the unsupported
        ! diagnostic and keeps the unit xfail.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index, init_index
        logical :: handled

        call set_empty(error_msg)
        init_index = node%initializer_index
        if (.not. node_exists(context%arena, init_index)) then
            call unsupported_feature_error('derived type initializer', &
                node%line, node%column, 'derived initializer did not expose a '// &
                'value expression', error_msg)
            return
        end if
        symbol_index = find_symbol(context, node%var_name)
        if (symbol_index <= 0) then
            error_msg = 'derived initializer target not defined: '//trim(node%var_name)
            return
        end if
        handled = .false.
        if (is_derived_constructor_call(context%arena, init_index, context, &
                                        type_index)) then
            if (.not. constructor_uses_user_routine(context, &
                    context%derived_types(type_index)%name)) then
                ! A bare `t()` equals default initialisation, already applied by
                ! define_derived_symbol; this also covers types with nested
                ! components the argument-storing path cannot construct.
                if (constructor_call_is_argumentless(context%arena, init_index)) &
                    then
                    handled = .true.
                else
                    call lower_derived_constructor_assignment(context%arena, &
                        init_index, context, symbol_index, type_index, handled, &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end if
        end if
        if (.not. handled) then
            call unsupported_feature_error('derived type constructor', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived type constructors', &
                                           error_msg)
        end if
    end subroutine lower_derived_scalar_initializer

    logical function constructor_call_is_argumentless(arena, call_index)
        ! True when a structure-constructor node t() carries no arguments.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: call_index

        constructor_call_is_argumentless = .false.
        if (.not. node_exists(arena, call_index)) return
        select type (node => arena%entries(call_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%arg_indices)) then
                constructor_call_is_argumentless = .true.
            else if (size(node%arg_indices) == 0) then
                constructor_call_is_argumentless = .true.
            end if
        end select
    end function constructor_call_is_argumentless

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

        index = find_symbol(context, name)
        if (index > 0) then
            ! Re-declaring the pre-bound function result variable (result(p)
            ! with type(t) :: p inside the body) is a no-op; its storage is the
            ! caller-supplied hidden argument.
            if (context%symbols(index)%is_derived .and. &
                context%symbols(index)%derived_type_index == derived_type_index) then
                call set_empty(error_msg)
                return
            end if
            ! A derived dummy: the parameter pointer is the base address of the
            ! caller's derived storage; component access reads through it.
            if (context%symbols(index)%is_parameter .and. &
                .not. context%symbols(index)%is_derived) then
                context%symbols(index)%value_kind = VALUE_DERIVED
                context%symbols(index)%is_derived = .true.
                context%symbols(index)%derived_type_index = derived_type_index
                context%symbols(index)%array_size = &
                    total_component_slots(context, derived_type_index)
                context%symbols(index)%element_address = &
                    context%symbols(index)%address
                call set_empty(error_msg)
                return
            end if
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
            sub_index, context, addr, error_msg, comp_kind)
        ! Address of obj%comp(i): slot = start_slot(comp) + (i - 1) * width,
        ! where width is the per-element i32-slot span of the component kind
        ! (2 for f64, 1 otherwise). access_index points to the
        ! component_access node (obj%comp); the enclosing subscript supplies
        ! sub_index. comp_kind returns the element value kind so the caller
        ! can pick a matching load/store width.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        integer, intent(in) :: sub_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(out), optional :: comp_kind
        integer :: symbol_index, type_index, component_index, width
        character(len=:), allocatable :: comp_name
        type(lr_operand_desc_t) :: sub_val, zero_based, scaled

        if (present(comp_kind)) comp_kind = VALUE_I32
        call resolve_component_ref(arena, access_index, context, symbol_index, &
                                   type_index, component_index, comp_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (component_is_alloc_array(context, type_index, component_index)) then
            if (present(comp_kind)) &
                comp_kind = component_kind(context, type_index, component_index)
            call alloc_array_component_element_address(arena, access_index, &
                sub_index, context, &
                component_kind(context, type_index, component_index), addr, &
                error_msg)
            return
        end if
        call resolve_array_component(arena, access_index, context, symbol_index, &
                                     type_index, component_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (present(comp_kind)) &
            comp_kind = component_kind(context, type_index, component_index)
        width = component_slot_width(context, type_index, component_index)

        call lower_i32_expression(arena, sub_index, context, sub_val, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, sub_val, &
                i32_immediate(context%session, 1_c_int64_t), zero_based, &
                error_msg)) return
        if (.not. emit_i32_binary(context%session, LR_OP_MUL, zero_based, &
                i32_immediate(context%session, int(width, c_int64_t)), scaled, &
                error_msg)) return
        call component_slot_address(context, symbol_index, type_index, &
                                    component_index, scaled, addr, error_msg)
    end subroutine derived_component_element_address

    subroutine alloc_array_component_descriptor(arena, access_index, context, &
                                                desc_addr, error_msg)
        ! Inline descriptor base address of a rank-1 allocatable array component
        ! (obj%comp): the component's start slot in the instance storage. The
        ! data pointer lives at byte 0, the i64 element extent at byte 8.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: desc_addr
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. node_exists(arena, access_index)) then
            error_msg = 'allocatable array component base is not an AST node'
            return
        end if
        select type (acc => arena%entries(access_index)%node)
        type is (component_access_node)
            call lower_derived_component_address(arena, acc, context, desc_addr, &
                error_msg, 'allocatable array component')
        class default
            error_msg = 'allocatable array component requires obj%comp form'
        end select
    end subroutine alloc_array_component_descriptor

    subroutine alloc_array_component_element_address(arena, access_index, &
            sub_index, context, comp_kind, addr, error_msg)
        ! Runtime element address of obj%comp(i) for a rank-1 allocatable array
        ! component: load the data pointer, then offset by (i - 1) * elem_size.
        ! The lower bound is 1.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        integer, intent(in) :: sub_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: comp_kind
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: desc_addr, data_ptr
        type(lr_operand_desc_t) :: idx_i32, zero_based, idx_i64, byte_offset

        call alloc_array_component_descriptor(arena, access_index, context, &
                                              desc_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_load(context%session, desc_addr, data_ptr, error_msg)) &
            return
        call lower_i32_expression(arena, sub_index, context, idx_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_binary(context%session, LR_OP_SUB, idx_i32, &
                i32_immediate(context%session, 1_c_int64_t), zero_based, &
                error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, zero_based, idx_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, idx_i64, &
                i64_immediate(context%session, allocatable_elem_size(comp_kind)), &
                byte_offset, error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                addr, error_msg)) return
        call set_empty(error_msg)
    end subroutine alloc_array_component_element_address

    subroutine lower_alloc_array_component_size(arena, arg_index, context, value, &
                                                handled, error_msg)
        ! size(obj%comp) for a rank-1 allocatable array component: load the i64
        ! element extent from descriptor byte 8 and narrow to i32. Leaves handled
        ! .false. for any other argument.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: desc_addr, extent_i64

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, arg_index)) return
        select type (arg => arena%entries(arg_index)%node)
        type is (component_access_node)
            if (.not. component_access_is_alloc_array(arena, arg, context)) return
            call alloc_array_component_descriptor(arena, arg_index, context, &
                                                  desc_addr, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i64_load_at(context%session, desc_addr, 8_c_int64_t, &
                    extent_i64, error_msg)) return
            if (.not. emit_liric_i64_to_i32(context%session, extent_i64, value, &
                    error_msg)) return
            handled = .true.
        end select
    end subroutine lower_alloc_array_component_size

    subroutine resolve_component_ref(arena, access_index, context, symbol_index, &
                                     type_index, component_index, comp_name, &
                                     error_msg)
        ! Resolve obj%comp to (symbol, type, component) indices and the component
        ! name, without applying the fixed-size-array gates. The base must be a
        ! plain derived variable.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: symbol_index, type_index, component_index
        character(len=:), allocatable, intent(out) :: comp_name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: base_name
        integer :: base_index

        symbol_index = 0
        type_index = 0
        component_index = 0
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
        call set_empty(error_msg)
    end subroutine resolve_component_ref

    subroutine resolve_array_component(arena, access_index, context, symbol_index, &
                                       type_index, component_index, error_msg)
        ! Resolve obj%comp to (symbol, type, component) indices, requiring comp to
        ! be a fixed-size array component of the base derived variable.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: symbol_index, type_index, component_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: comp_name

        call resolve_component_ref(arena, access_index, context, symbol_index, &
                                   type_index, component_index, comp_name, error_msg)
        if (len_trim(error_msg) > 0) return
        if (component_is_alloc_array(context, type_index, component_index)) then
            error_msg = 'component is an allocatable array: '//trim(comp_name)
            return
        end if
        if (component_kind(context, type_index, component_index) == &
            VALUE_CHARACTER) then
            ! A character component reuses component_array_size as its slot count,
            ! so exclude it from the intrinsic-array-component path.
            error_msg = 'component is not an array: '//trim(comp_name)
            return
        end if
        if (context%derived_types(type_index)%component_array_size(component_index) &
            <= 1) then
            error_msg = 'component is not an array: '//trim(comp_name)
            return
        end if
        call set_empty(error_msg)
    end subroutine resolve_array_component

    subroutine component_slot_address(context, symbol_index, type_index, &
                                      component_index, element_offset, addr, &
                                      error_msg)
        ! Element address inside a fixed-size array component: the component's
        ! start slot plus element_offset (already scaled by the element width).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index, type_index, component_index
        type(lr_operand_desc_t), intent(in) :: element_offset
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot

        if (.not. emit_i32_binary(context%session, LR_OP_ADD, element_offset, &
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
    end subroutine component_slot_address

    subroutine component_element_address_at(context, symbol_index, type_index, &
                                            component_index, zero_based, addr, &
                                            error_msg)
        ! Element address for a compile-time linear index (0-based) into a
        ! fixed-size array component.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index, type_index, component_index
        integer, intent(in) :: zero_based
        type(lr_operand_desc_t), intent(out) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: width

        width = component_slot_width(context, type_index, component_index)
        call component_slot_address(context, symbol_index, type_index, &
            component_index, &
            i32_immediate(context%session, int(zero_based * width, c_int64_t)), &
            addr, error_msg)
    end subroutine component_element_address_at

    subroutine lower_derived_component_element_assignment(arena, node, target, &
                                                          context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, addr
        integer :: comp_kind

        if (.not. allocated(target%arg_indices) .or. &
            size(target%arg_indices) /= 1) then
            call unsupported_feature_error('derived component element', &
                node%line, node%column, 'only single-index array component '// &
                'access is supported', error_msg)
            return
        end if
        call derived_component_element_address(arena, target%base_expr_index, &
            target%arg_indices(1), context, addr, error_msg, comp_kind)
        if (len_trim(error_msg) > 0) return
        call lower_component_rhs_value(arena, node%value_index, comp_kind, &
                                       context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_typed_component_store(context, comp_kind, value, addr, error_msg)
    end subroutine lower_derived_component_element_assignment

    subroutine method_call_info(arena, node, context, base_var_index, &
                                impl_name, pass_name, found, error_msg)
        ! For obj%m(args): resolve m to its bound implementation via obj's
        ! derived type, returning the base variable's node index, the
        ! implementation name, and the pass(name) dummy (empty for default pass).
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: base_var_index
        character(len=:), allocatable, intent(out) :: impl_name
        character(len=:), allocatable, intent(out) :: pass_name
        logical, intent(out) :: found
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, method_name
        integer :: symbol_index, type_index, b

        found = .false.
        base_var_index = 0
        call set_empty(error_msg)
        call set_empty(impl_name)
        call set_empty(pass_name)
        if (node%base_expr_index <= 0) return
        if (.not. node_exists(arena, node%base_expr_index)) return
        select type (ca => arena%entries(node%base_expr_index)%node)
        type is (component_access_node)
            if (.not. allocated(ca%component_name)) return
            method_name = trim(ca%component_name)
            base_var_index = ca%base_expr_index
        class default
            return
        end select
        if (.not. is_identifier(arena, base_var_index)) return
        call get_identifier_name(arena, base_var_index, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, var_name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_derived) return
        type_index = context%symbols(symbol_index)%derived_type_index
        if (type_index <= 0) return
        if (.not. allocated(context%derived_types(type_index)%binding_method_names)) &
            return
        do b = 1, context%derived_types(type_index)%binding_count
            if (trim(context%derived_types(type_index)%binding_method_names(b)) == &
                method_name) then
                impl_name = trim(context%derived_types(type_index)% &
                                 binding_target_names(b))
                if (allocated(context%derived_types(type_index)%binding_pass_names)) &
                    pass_name = trim(context%derived_types(type_index)% &
                                     binding_pass_names(b))
                found = .true.
                return
            end if
        end do
    end subroutine method_call_info

    integer function impl_pass_position(arena, impl_name, pass_name) result(pos)
        ! 1-based position of the dummy named pass_name in impl's parameter
        ! list; 1 (the default pass position) when pass_name is empty or absent.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: impl_name
        character(len=*), intent(in) :: pass_name
        integer :: n, p
        integer, allocatable :: params(:)
        character(len=:), allocatable :: pname, perr

        pos = 1
        if (len_trim(pass_name) == 0) return
        call impl_param_indices(arena, impl_name, params)
        if (.not. allocated(params)) return
        do p = 1, size(params)
            call parameter_name(arena, params(p), pname, perr)
            if (len_trim(perr) > 0) cycle
            if (trim(pname) == trim(pass_name)) then
                pos = p
                return
            end if
        end do
    end function impl_pass_position

    subroutine impl_param_indices(arena, impl_name, params)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: impl_name
        integer, allocatable, intent(out) :: params(:)
        integer :: n

        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (subroutine_def_node)
                if (allocated(node%name)) then
                    if (same_name(node%name, impl_name)) then
                        if (allocated(node%param_indices)) params = node%param_indices
                        return
                    end if
                end if
            type is (function_def_node)
                if (allocated(node%name)) then
                    if (same_name(node%name, impl_name)) then
                        if (allocated(node%param_indices)) params = node%param_indices
                        return
                    end if
                end if
            end select
        end do
    end subroutine impl_param_indices

    logical function is_type_bound_method_call(arena, node, context)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: base_var_index
        character(len=:), allocatable :: impl_name, pass_name, err

        call method_call_info(arena, node, context, base_var_index, impl_name, &
                              pass_name, is_type_bound_method_call, err)
        if (len_trim(err) > 0) is_type_bound_method_call = .false.
    end function is_type_bound_method_call

    subroutine method_call_arg_indices(node, base_var_index, position, combined)
        ! Insert obj (base_var_index) at the pass position; explicit method
        ! arguments fill the remaining slots in order.
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: base_var_index
        integer, intent(in) :: position
        integer, allocatable, intent(out) :: combined(:)
        integer :: n, i, k, pos

        n = 0
        if (allocated(node%arg_indices)) n = size(node%arg_indices)
        allocate (combined(n + 1))
        pos = position
        if (pos < 1 .or. pos > n + 1) pos = 1
        combined(pos) = base_var_index
        k = 0
        do i = 1, n + 1
            if (i == pos) cycle
            k = k + 1
            combined(i) = node%arg_indices(k)
        end do
    end subroutine method_call_arg_indices

    logical function is_method_subroutine_call(name)
        ! A type-bound subroutine call is parsed as `var%method`.
        character(len=*), intent(in) :: name

        is_method_subroutine_call = index(name, '%') > 0
    end function is_method_subroutine_call

    subroutine lower_method_subroutine_call(arena, name, arg_indices, context, &
                                            error_msg)
        ! call obj%m(args) -> m_impl(obj, args). obj is passed first.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, method_name, impl_name
        character(len=:), allocatable :: pass_name
        integer :: pct, symbol_index, type_index, b, n, position, i, k
        type(lr_operand_desc_t), allocatable :: method_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)

        call set_empty(error_msg)
        pct = index(name, '%')
        var_name = trim(name(1:pct - 1))
        method_name = trim(name(pct + 1:))
        symbol_index = find_symbol(context, var_name)
        if (symbol_index <= 0) then
            error_msg = 'type-bound call base was not declared: '//var_name
            return
        end if
        if (.not. context%symbols(symbol_index)%is_derived) then
            error_msg = 'type-bound call base is not a derived type: '//var_name
            return
        end if
        type_index = context%symbols(symbol_index)%derived_type_index
        call set_empty(impl_name)
        call set_empty(pass_name)
        do b = 1, context%derived_types(type_index)%binding_count
            if (trim(context%derived_types(type_index)%binding_method_names(b)) == &
                method_name) then
                impl_name = trim(context%derived_types(type_index)% &
                                 binding_target_names(b))
                if (allocated(context%derived_types(type_index)%binding_pass_names)) &
                    pass_name = trim(context%derived_types(type_index)% &
                                     binding_pass_names(b))
                exit
            end if
        end do
        if (len_trim(impl_name) == 0) then
            error_msg = 'no type-bound procedure "'//method_name//'" on '//var_name
            return
        end if

        position = impl_pass_position(arena, impl_name, pass_name)
        if (allocated(arg_indices)) then
            call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
                                        impl_name, method_args, copyback_indices, &
                                        error_msg, self_position=position)
            if (len_trim(error_msg) > 0) return
        else
            allocate (method_args(0))
            allocate (copyback_indices(0))
        end if
        n = size(method_args)
        allocate (all_args(n + 1))
        if (position < 1 .or. position > n + 1) position = 1
        all_args(position) = context%symbols(symbol_index)%element_address
        k = 0
        do i = 1, n + 1
            if (i == position) cycle
            k = k + 1
            all_args(i) = method_args(k)
        end do
        if (.not. emit_void_call(context%session, &
                call_emit_name(arena, impl_name), all_args, error_msg)) return
        call copy_back_reference_args(context, method_args, copyback_indices, &
                                      error_msg)
    end subroutine lower_method_subroutine_call

    subroutine lower_method_call_i32(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: base_var_index, position
        character(len=:), allocatable :: impl_name, pass_name
        logical :: found
        integer, allocatable :: combined(:)
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)

        call method_call_info(arena, node, context, base_var_index, impl_name, &
                              pass_name, found, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. found) then
            error_msg = 'not a type-bound method call'
            return
        end if
        position = impl_pass_position(arena, impl_name, pass_name)
        call method_call_arg_indices(node, base_var_index, position, combined)
        call prepare_reference_args(arena, combined, context, VALUE_I32, &
                                    impl_name, args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_call(context%session, call_emit_name(arena, impl_name), &
                                args, value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_method_call_i32

    subroutine lower_derived_component_element_load(arena, node, context, value, &
                                                    error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer :: comp_kind

        if (.not. allocated(node%arg_indices) .or. &
            size(node%arg_indices) /= 1) then
            call unsupported_feature_error('derived component element', &
                node%line, node%column, 'only single-index array component '// &
                'access is supported', error_msg)
            return
        end if
        call derived_component_element_address(arena, node%base_expr_index, &
            node%arg_indices(1), context, addr, error_msg, comp_kind)
        if (len_trim(error_msg) > 0) return
        call emit_typed_component_load(context, comp_kind, addr, value, error_msg)
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
        integer :: comp_kind
        integer :: type_index, comp_index, dest_len
        logical :: is_array_component

        if (component_access_is_alloc_array(arena, target, context)) then
            value = i32_immediate(context%session, 0_c_int64_t)
            call lower_alloc_array_component_assignment(arena, node, target, &
                                                        context, error_msg)
            return
        end if

        call component_target_is_array(arena, node%target_index, context, &
                                       is_array_component)
        if (is_array_component) then
            value = i32_immediate(context%session, 0_c_int64_t)
            call lower_derived_component_array_assignment(arena, node, context, &
                                                          error_msg)
            return
        end if

        comp_kind = derived_component_access_kind(arena, target, context)
        if (comp_kind == VALUE_CHARACTER) then
            call lower_derived_component_address(arena, target, context, &
                component_address, error_msg, &
                'derived type character component assignment target')
            if (len_trim(error_msg) > 0) return
            type_index = component_base_type_index(arena, target%base_expr_index, &
                                                   context)
            comp_index = 0
            if (type_index > 0) comp_index = find_derived_component(context, &
                type_index, target%component_name)
            dest_len = 0
            if (comp_index > 0) dest_len = derived_component_char_length(context, &
                type_index, comp_index)
            if (dest_len <= 0) then
                call unsupported_feature_error('derived type character '// &
                    'component access', target%line, target%column, &
                    'writing a character component is not supported by '// &
                    'direct LIRIC session', error_msg)
                return
            end if
            call store_char_component_value(arena, node%value_index, &
                component_address, dest_len, context, error_msg)
            value = i32_immediate(context%session, 0_c_int64_t)
            return
        end if
        call lower_component_rhs_value(arena, node%value_index, comp_kind, &
                                       context, value, error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_derived_component_address(arena, target, context, &
                                             component_address, error_msg, &
                                             'derived type component assignment target')
        if (len_trim(error_msg) > 0) return

        if (component_access_is_allocatable(arena, target, context)) then
            call store_allocatable_component_value(context, component_address, &
                                                   comp_kind, value, error_msg)
            return
        end if

        call emit_typed_component_store(context, comp_kind, value, &
                                        component_address, error_msg)
    end subroutine lower_derived_component_assignment

    subroutine component_target_is_array(arena, target_index, context, is_array)
        ! True when obj%comp names a fixed-size array component of a plain
        ! derived variable, so the whole-component array assignment path applies.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(in) :: context
        logical, intent(out) :: is_array
        character(len=:), allocatable :: err
        integer :: symbol_index, type_index, component_index

        is_array = .false.
        call resolve_array_component(arena, target_index, context, symbol_index, &
                                     type_index, component_index, err)
        if (len_trim(err) == 0) is_array = .true.
    end subroutine component_target_is_array

    subroutine lower_derived_component_array_assignment(arena, node, context, &
                                                        error_msg)
        ! obj%v = rhs for a fixed-size intrinsic array component: an array
        ! constructor, a scalar broadcast, or a whole-array copy from another
        ! array. Character and derived-array components stay out of this path.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index, type_index, component_index, comp_kind, array_size

        call resolve_array_component(arena, node%target_index, context, &
                                     symbol_index, type_index, component_index, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        comp_kind = component_kind(context, type_index, component_index)
        array_size = context%derived_types(type_index)% &
                     component_array_size(component_index)
        if (comp_kind == VALUE_CHARACTER) then
            call unsupported_feature_error('array component assignment', &
                node%line, node%column, 'character array component assignment '// &
                'is not supported', error_msg)
            return
        end if
        if (node_exists(arena, node%value_index)) then
            select type (rhs => arena%entries(node%value_index)%node)
            type is (array_literal_node)
                call store_component_array_constructor(arena, rhs, symbol_index, &
                    type_index, component_index, comp_kind, array_size, context, &
                    error_msg)
                return
            end select
        end if
        call store_component_array_copy(arena, node%value_index, symbol_index, &
            type_index, component_index, comp_kind, array_size, context, error_msg)
    end subroutine lower_derived_component_array_assignment

    subroutine store_component_array_constructor(arena, ctor, symbol_index, &
            type_index, component_index, comp_kind, array_size, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(array_literal_node), intent(in) :: ctor
        integer, intent(in) :: symbol_index, type_index, component_index
        integer, intent(in) :: comp_kind, array_size
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, addr
        integer, allocatable :: flat(:)
        integer :: i

        if (is_implied_do_constructor(arena, ctor) .or. &
            .not. allocated(ctor%element_indices)) then
            call unsupported_feature_error('array component assignment', &
                ctor%line, ctor%column, 'only an explicit array constructor '// &
                'is supported for array components', error_msg)
            return
        end if
        allocate (flat(0))
        call flatten_constructor_elements(arena, ctor%element_indices, flat, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        if (size(flat) /= array_size) then
            error_msg = 'array constructor element count does not match the '// &
                'declared component size'
            return
        end if
        do i = 1, array_size
            call lower_component_rhs_value(arena, flat(i), comp_kind, context, &
                                           value, error_msg)
            if (len_trim(error_msg) > 0) return
            call component_element_address_at(context, symbol_index, type_index, &
                                              component_index, i - 1, addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_typed_component_store(context, comp_kind, value, addr, &
                                            error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine store_component_array_constructor

    subroutine store_component_array_copy(arena, value_index, symbol_index, &
            type_index, component_index, comp_kind, array_size, context, error_msg)
        ! obj%v = scalar (broadcast) or obj%v = other_array (elementwise copy
        ! from a plain array variable or another array component).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: symbol_index, type_index, component_index
        integer, intent(in) :: comp_kind, array_size
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value, addr
        logical :: broadcast
        integer :: i

        broadcast = is_scalar_broadcast_rhs(arena, value_index, context)
        if (broadcast) then
            call lower_component_rhs_value(arena, value_index, comp_kind, context, &
                                           value, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        do i = 1, array_size
            if (.not. broadcast) then
                call load_component_array_source(arena, value_index, comp_kind, &
                    array_size, i - 1, context, value, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            call component_element_address_at(context, symbol_index, type_index, &
                                              component_index, i - 1, addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_typed_component_store(context, comp_kind, value, addr, &
                                            error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine store_component_array_copy

    subroutine load_component_array_source(arena, value_index, comp_kind, &
            array_size, zero_based, context, value, error_msg)
        ! Load element zero_based of a whole-array RHS: a plain array variable or
        ! another fixed-size array component. Anything else declines.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index, comp_kind, array_size, zero_based
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        character(len=:), allocatable :: name
        integer :: sym, src_type, src_comp

        if (.not. node_exists(arena, value_index)) then
            error_msg = 'array component assignment rhs is not an AST node'
            return
        end if
        if (is_identifier(arena, value_index)) then
            call get_identifier_name(arena, value_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, name)
            if (sym <= 0) then
                error_msg = 'array component assignment rhs was not declared: '// &
                            trim(name)
                return
            end if
            if (.not. is_elementwise_array_operand(context, sym)) then
                error_msg = 'array component assignment requires an array rhs'
                return
            end if
            if (context%symbols(sym)%array_size /= array_size) then
                error_msg = 'array component assignment rhs size does not match'
                return
            end if
            call load_array_linear_element(context, sym, &
                int(zero_based, c_int64_t), value, error_msg)
            return
        end if
        select type (rhs => arena%entries(value_index)%node)
        type is (component_access_node)
            call resolve_array_component(arena, value_index, context, sym, &
                                         src_type, src_comp, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%derived_types(src_type)% &
                component_array_size(src_comp) /= array_size) then
                error_msg = 'array component assignment rhs size does not match'
                return
            end if
            call component_element_address_at(context, sym, src_type, src_comp, &
                                              zero_based, addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call emit_typed_component_load(context, comp_kind, addr, value, &
                                           error_msg)
            return
        end select
        call unsupported_feature_error('array component assignment', &
            get_node_line(arena, value_index), get_node_column(arena, value_index), &
            'this array component rhs form is not supported', error_msg)
    end subroutine load_component_array_source

    subroutine try_load_array_component_element(arena, access_index, zero_based, &
                                                context, value, handled, error_msg)
        ! Load element zero_based of an array component obj%v used as a whole-array
        ! operand. handled stays .false. when obj%v is not a fixed-size intrinsic
        ! array component, leaving the caller's other paths untouched.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: access_index, zero_based
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: addr
        integer :: sym, type_index, comp_index, comp_kind

        handled = .false.
        call set_empty(error_msg)
        call resolve_array_component(arena, access_index, context, sym, type_index, &
                                     comp_index, error_msg)
        if (len_trim(error_msg) > 0) then
            call set_empty(error_msg)
            return
        end if
        comp_kind = component_kind(context, type_index, comp_index)
        if (comp_kind == VALUE_CHARACTER) return
        call component_element_address_at(context, sym, type_index, comp_index, &
                                          zero_based, addr, error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_typed_component_load(context, comp_kind, addr, value, error_msg)
        handled = .true.
    end subroutine try_load_array_component_element

    subroutine store_allocatable_component_value(context, slot_address, comp_kind, &
                                                 value, error_msg)
        ! Write a scalar allocatable component: load the inline data pointer from
        ! its slot, then store the value through that pointer. The component must
        ! already be allocated.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: slot_address
        integer, intent(in) :: comp_kind
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr

        if (.not. emit_ptr_load(context%session, slot_address, data_ptr, &
                                error_msg)) return
        call emit_typed_component_store(context, comp_kind, value, data_ptr, &
                                        error_msg)
    end subroutine store_allocatable_component_value

    subroutine lower_allocate_scalar_component(arena, target_index, context, &
                                               handled, error_msg)
        ! allocate(x%c) for a scalar allocatable component: malloc a typed slot
        ! and store its pointer into the component's inline descriptor. Leaves
        ! handled .false. for any other target so the caller keeps its own paths.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot_address, data_ptr
        integer :: comp_kind

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, target_index)) return
        select type (target => arena%entries(target_index)%node)
        type is (component_access_node)
            if (.not. component_access_is_allocatable(arena, target, context)) return
            comp_kind = derived_component_access_kind(arena, target, context)
            call lower_derived_component_address(arena, target, context, &
                slot_address, error_msg, 'allocate component target')
            if (len_trim(error_msg) > 0) return
            if (.not. emit_malloc(context%session, &
                    i64_immediate(context%session, &
                        allocatable_elem_size(comp_kind)), data_ptr, error_msg)) &
                return
            if (.not. emit_ptr_store(context%session, data_ptr, slot_address, &
                                     error_msg)) return
            handled = .true.
        end select
    end subroutine lower_allocate_scalar_component

    subroutine lower_allocate_array_component(arena, target_index, context, &
                                              handled, error_msg)
        ! allocate(obj%comp(n)) for a rank-1 allocatable array component: malloc
        ! n * elem_size bytes, store the data pointer at descriptor byte 0 and the
        ! i64 element extent at byte 8. Leaves handled .false. for any other
        ! target so the caller keeps its own paths.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: target_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: desc_addr, data_ptr, n_i32, n_i64, bytes_i64
        integer :: comp_kind

        handled = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, target_index)) return
        select type (target => arena%entries(target_index)%node)
        type is (call_or_subscript_node)
            if (target%base_expr_index <= 0) return
            if (.not. allocated(target%arg_indices)) return
            if (size(target%arg_indices) /= 1) return
            if (.not. node_exists(arena, target%base_expr_index)) return
            select type (base => arena%entries(target%base_expr_index)%node)
            type is (component_access_node)
                if (.not. component_access_is_alloc_array(arena, base, context)) &
                    return
                comp_kind = derived_component_access_kind(arena, base, context)
            class default
                return
            end select
            call alloc_array_component_descriptor(arena, target%base_expr_index, &
                                                  context, desc_addr, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_i32_expression(arena, target%arg_indices(1), context, n_i32, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, &
                    error_msg)) return
            if (.not. emit_i64_binary(context%session, LR_OP_MUL, n_i64, &
                    i64_immediate(context%session, allocatable_elem_size(comp_kind)), &
                    bytes_i64, error_msg)) return
            if (.not. emit_malloc(context%session, bytes_i64, data_ptr, error_msg)) &
                return
            if (.not. emit_ptr_store(context%session, data_ptr, desc_addr, &
                                     error_msg)) return
            if (.not. emit_i64_store_at(context%session, n_i64, desc_addr, &
                    8_c_int64_t, error_msg)) return
            handled = .true.
        end select
    end subroutine lower_allocate_array_component

    subroutine lower_alloc_array_component_assignment(arena, node, target, &
                                                      context, error_msg)
        ! obj%v = rhs for a rank-1 allocatable array component: reallocate the
        ! component to the rhs extent, then store each element in order. Supports
        ! an explicit array constructor and a whole plain-array or allocatable
        ! variable rhs whose extent is known at compile time.
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(component_access_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: comp_kind, n, i, src_sym
        integer, allocatable :: flat(:)
        type(lr_operand_desc_t) :: desc_addr, data_ptr, elem_val
        logical :: from_ctor

        comp_kind = derived_component_access_kind(arena, target, context)
        if (comp_kind == VALUE_CHARACTER) then
            call unsupported_feature_error('allocatable array component '// &
                'assignment', target%line, target%column, 'character allocatable '// &
                'array component assignment is not supported', error_msg)
            return
        end if
        call alloc_component_assign_shape(arena, node%value_index, context, &
                                          from_ctor, flat, src_sym, n, error_msg)
        if (len_trim(error_msg) > 0) return
        call alloc_array_component_descriptor(arena, node%target_index, context, &
                                              desc_addr, error_msg)
        if (len_trim(error_msg) > 0) return
        call realloc_alloc_component(context, desc_addr, comp_kind, n, data_ptr, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 1, n
            if (from_ctor) then
                call lower_component_rhs_value(arena, flat(i), comp_kind, context, &
                                               elem_val, error_msg)
            else
                call load_array_linear_element(context, src_sym, &
                    int(i - 1, c_int64_t), elem_val, error_msg)
            end if
            if (len_trim(error_msg) > 0) return
            call store_alloc_component_element(context, data_ptr, comp_kind, &
                                               i - 1, elem_val, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_alloc_array_component_assignment

    subroutine alloc_component_assign_shape(arena, value_index, context, &
                                            from_ctor, flat, src_sym, n, error_msg)
        ! Classify the rhs of a whole allocatable-array-component assignment:
        ! an explicit array constructor (from_ctor, flat holds element indices),
        ! or a whole array/allocatable variable of compile-time extent n (src_sym).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(in) :: context
        logical, intent(out) :: from_ctor
        integer, allocatable, intent(out) :: flat(:)
        integer, intent(out) :: src_sym, n
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name

        from_ctor = .false.
        src_sym = 0
        n = 0
        if (.not. node_exists(arena, value_index)) then
            error_msg = 'allocatable array component assignment rhs is not a node'
            return
        end if
        select type (rhs => arena%entries(value_index)%node)
        type is (array_literal_node)
            if (is_implied_do_constructor(arena, rhs) .or. &
                .not. allocated(rhs%element_indices)) then
                call unsupported_feature_error('allocatable array component '// &
                    'assignment', rhs%line, rhs%column, 'only an explicit array '// &
                    'constructor is supported for a constructor rhs', error_msg)
                return
            end if
            allocate (flat(0))
            call flatten_constructor_elements(arena, rhs%element_indices, flat, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            n = size(flat)
            from_ctor = .true.
            call set_empty(error_msg)
            return
        end select
        if (.not. is_identifier(arena, value_index)) then
            call unsupported_feature_error('allocatable array component '// &
                'assignment', get_node_line(arena, value_index), &
                get_node_column(arena, value_index), 'this rhs form is not '// &
                'supported for an allocatable array component', error_msg)
            return
        end if
        call get_identifier_name(arena, value_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        src_sym = find_symbol(context, name)
        if (src_sym <= 0) then
            error_msg = 'allocatable array component assignment rhs was not '// &
                        'declared: '//trim(name)
            return
        end if
        if (.not. is_elementwise_array_operand(context, src_sym)) then
            call unsupported_feature_error('allocatable array component '// &
                'assignment', get_node_line(arena, value_index), &
                get_node_column(arena, value_index), 'a whole-array rhs must be '// &
                'a compile-time-sized array', error_msg)
            return
        end if
        if (context%symbols(src_sym)%is_allocatable) then
            n = context%symbols(src_sym)%allocatable_static_size
        else
            n = context%symbols(src_sym)%array_size
        end if
        if (n <= 0) then
            call unsupported_feature_error('allocatable array component '// &
                'assignment', get_node_line(arena, value_index), &
                get_node_column(arena, value_index), 'a whole-array rhs must have '// &
                'a positive compile-time extent', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine alloc_component_assign_shape

    subroutine realloc_alloc_component(context, desc_addr, comp_kind, n, data_ptr, &
                                       error_msg)
        ! Size a rank-1 allocatable array component to n elements: free any
        ! current data (free(null) is a no-op on the default-nulled descriptor),
        ! malloc n * elem_size, then store the data pointer at byte 0 and the i64
        ! element extent at byte 8. Returns the fresh data pointer.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: desc_addr
        integer, intent(in) :: comp_kind, n
        type(lr_operand_desc_t), intent(out) :: data_ptr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: old_ptr, bytes, extent
        integer(c_int64_t) :: elem_size

        elem_size = allocatable_elem_size(comp_kind)
        if (.not. emit_ptr_load(context%session, desc_addr, old_ptr, error_msg)) &
            return
        if (.not. emit_free(context%session, old_ptr, error_msg)) return
        bytes = i64_immediate(context%session, int(n, c_int64_t) * elem_size)
        if (.not. emit_malloc(context%session, bytes, data_ptr, error_msg)) return
        if (.not. emit_ptr_store(context%session, data_ptr, desc_addr, error_msg)) &
            return
        extent = i64_immediate(context%session, int(n, c_int64_t))
        if (.not. emit_i64_store_at(context%session, extent, desc_addr, &
                8_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine realloc_alloc_component

    subroutine store_alloc_component_element(context, data_ptr, comp_kind, &
                                             zero_based, value, error_msg)
        ! Store value at element zero_based of an allocatable array component whose
        ! data pointer is data_ptr.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_ptr
        integer, intent(in) :: comp_kind, zero_based
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: off, addr

        off = i64_immediate(context%session, &
            int(zero_based, c_int64_t) * allocatable_elem_size(comp_kind))
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, off, addr, &
                error_msg)) return
        call emit_typed_component_store(context, comp_kind, value, addr, error_msg)
    end subroutine store_alloc_component_element

    subroutine lower_component_rhs_value(arena, value_index, comp_kind, context, &
                                         value, error_msg)
        ! Evaluate a scalar RHS in the component's kind so the store width and
        ! the value type agree.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: comp_kind
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (comp_kind == VALUE_CHARACTER) then
            error_msg = 'character component access is not supported by direct '// &
                        'LIRIC session'
            return
        end if
        select case (comp_kind)
        case (VALUE_F64)
            call lower_f64_expression(arena, value_index, context, value, error_msg)
        case (VALUE_F32)
            call lower_f32_expression(arena, value_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, value_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, value_index, context, value, error_msg)
        end select
    end subroutine lower_component_rhs_value

    subroutine lower_derived_component_load(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address
        integer :: comp_kind

        if (component_access_is_alloc_array(arena, node, context)) then
            call unsupported_feature_error('allocatable array component access', &
                node%line, node%column, 'reading a whole allocatable array '// &
                'component is not supported by direct LIRIC session', error_msg)
            return
        end if
        comp_kind = derived_component_access_kind(arena, node, context)
        if (comp_kind == VALUE_CHARACTER) then
            call unsupported_feature_error('derived type character component '// &
                'access', node%line, node%column, 'reading a character '// &
                'component is not supported by direct LIRIC session', error_msg)
            return
        end if
        call lower_derived_component_address(arena, node, context, &
                                             component_address, error_msg, &
                                             'derived type component expression')
        if (len_trim(error_msg) > 0) return

        if (component_access_is_allocatable(arena, node, context)) then
            call load_allocatable_component_value(context, component_address, &
                                                  comp_kind, value, error_msg)
            return
        end if

        call emit_typed_component_load(context, comp_kind, component_address, &
                                       value, error_msg)
    end subroutine lower_derived_component_load

    logical function component_access_is_allocatable(arena, node, context) &
            result(is_alloc)
        ! True when a component_access node names a scalar allocatable component.
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: type_index, component_index

        is_alloc = .false.
        type_index = component_base_type_index(arena, node%base_expr_index, context)
        if (type_index <= 0) return
        component_index = find_derived_component(context, type_index, &
                                                 node%component_name)
        if (component_index <= 0) return
        is_alloc = component_is_allocatable(context, type_index, component_index)
    end function component_access_is_allocatable

    logical function component_access_is_alloc_array(arena, node, context) &
            result(is_alloc)
        ! True when a component_access node names a rank-1 allocatable array
        ! component.
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: type_index, component_index

        is_alloc = .false.
        type_index = component_base_type_index(arena, node%base_expr_index, context)
        if (type_index <= 0) return
        component_index = find_derived_component(context, type_index, &
                                                 node%component_name)
        if (component_index <= 0) return
        is_alloc = component_is_alloc_array(context, type_index, component_index)
    end function component_access_is_alloc_array

    subroutine load_allocatable_component_value(context, slot_address, comp_kind, &
                                                value, error_msg)
        ! Read a scalar allocatable component: load the inline data pointer from
        ! its slot, then load the value through that pointer.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: slot_address
        integer, intent(in) :: comp_kind
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr

        if (.not. emit_ptr_load(context%session, slot_address, data_ptr, &
                                error_msg)) return
        call emit_typed_component_load(context, comp_kind, data_ptr, value, &
                                       error_msg)
    end subroutine load_allocatable_component_value

    logical function is_derived_result_call(arena, value_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(in) :: context

        is_derived_result_call = .false.
        if (.not. node_exists(arena, value_index)) return
        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            if (allocated(v%name) .and. v%base_expr_index == 0 .and. &
                .not. v%is_array_access) &
                is_derived_result_call = is_contained_derived_function(context, &
                                                                       v%name)
        end select
    end function is_derived_result_call

    logical function is_array_result_call(arena, value_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(in) :: context

        is_array_result_call = .false.
        if (.not. node_exists(arena, value_index)) return
        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            if (allocated(v%name) .and. v%base_expr_index == 0 .and. &
                .not. v%is_array_access) &
                is_array_result_call = is_contained_array_function(context, v%name)
        end select
    end function is_array_result_call

    subroutine lower_array_result_call(arena, value_index, dest_symbol_index, &
                                       context, error_msg)
        ! r = vec_fn(...): pass r's array storage as the hidden sret pointer; the
        ! callee writes its array result straight into r (array function results).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: dest_symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(call_or_subscript_node) :: call_node
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        integer :: i

        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            call_node = v
        class default
            error_msg = 'expected function call for array result'
            return
        end select
        func_name = call_node%name

        if (allocated(call_node%arg_indices)) then
            call prepare_reference_args(arena, call_node%arg_indices, context, &
                                        VALUE_I32, call_node%name, user_args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (user_args(0))
            allocate (copyback_indices(0))
        end if

        allocate (all_args(size(user_args) + 1))
        all_args(1) = context%symbols(dest_symbol_index)%element_address
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do

        if (.not. emit_void_call(context%session, &
                                 call_emit_name(arena, trim(func_name)), &
                                 all_args, error_msg)) return
        call copy_back_reference_args(context, user_args, copyback_indices, &
                                      error_msg)
    end subroutine lower_array_result_call

    subroutine lower_derived_result_call(arena, value_index, dest_symbol_index, &
                                         context, error_msg)
        ! q = make_point(): pass q's storage as the hidden result pointer; the
        ! callee writes its derived result directly into it.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: dest_symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(call_or_subscript_node) :: call_node
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        integer :: i

        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            call_node = v
        class default
            error_msg = 'expected function call for derived result'
            return
        end select
        func_name = call_node%name

        if (allocated(call_node%arg_indices)) then
            call prepare_reference_args(arena, call_node%arg_indices, context, &
                                        VALUE_I32, call_node%name, user_args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (user_args(0))
            allocate (copyback_indices(0))
        end if

        allocate (all_args(size(user_args) + 1))
        all_args(1) = context%symbols(dest_symbol_index)%element_address
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do

        if (.not. emit_void_call(context%session, &
                                 call_emit_name(arena, trim(func_name)), &
                                 all_args, error_msg)) return
        call copy_back_reference_args(context, user_args, copyback_indices, &
                                      error_msg)
    end subroutine lower_derived_result_call

    recursive subroutine resolve_component_base(arena, base_index, context, &
                                      symbol_index, type_index, element_base, &
                                      error_msg)
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
        character(len=:), allocatable :: id_name

        symbol_index = 0
        type_index = 0
        if (.not. node_exists(arena, base_index)) then
            error_msg = 'component base index does not reference an AST node'
            return
        end if
        if (is_identifier(arena, base_index)) then
            call get_identifier_name(arena, base_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, id_name)
            if (symbol_index <= 0) then
                error_msg = 'component base was not declared: '//trim(id_name)
                return
            end if
            element_base = i32_immediate(context%session, 0_c_int64_t)
        else
        select type (base => arena%entries(base_index)%node)
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
        type is (component_access_node)
            ! Nested derived base x%c: resolve x%c to (symbol, inner type, slot
            ! base) so the leaf access x%c%field adds its own start slot on top.
            call resolve_nested_component_base(arena, base, context, &
                symbol_index, type_index, element_base, error_msg)
            return
        class default
            error_msg = 'nested component access is not supported'
            return
        end select
        end if
        if (.not. context%symbols(symbol_index)%is_derived) then
            error_msg = 'component base is not a derived type'
            return
        end if
        type_index = context%symbols(symbol_index)%derived_type_index
        call set_empty(error_msg)
    end subroutine resolve_component_base

    recursive subroutine resolve_nested_component_base(arena, node, context, &
            symbol_index, type_index, element_base, error_msg)
        ! Resolve a derived component access x%c (the base of a deeper access) to
        ! the owning symbol, the inner component's derived type index, and the
        ! slot offset of the inner aggregate within the flat instance layout.
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: symbol_index
        integer, intent(out) :: type_index
        type(lr_operand_desc_t), intent(out) :: element_base
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: outer_type, component_index
        type(lr_operand_desc_t) :: base_off

        call resolve_component_base(arena, node%base_expr_index, context, &
                                    symbol_index, outer_type, base_off, error_msg)
        if (len_trim(error_msg) > 0) return
        component_index = find_derived_component(context, outer_type, &
                                                 node%component_name)
        if (component_index <= 0) then
            error_msg = 'component is not declared for type '// &
                trim(context%derived_types(outer_type)%name)//': '// &
                trim(node%component_name)
            return
        end if
        type_index = context%derived_types(outer_type)% &
                     component_type_index(component_index)
        if (type_index <= 0) then
            error_msg = 'component is not a derived type: '// &
                trim(node%component_name)
            return
        end if
        if (.not. emit_i32_binary(context%session, LR_OP_ADD, base_off, &
                i32_immediate(context%session, &
                    int(component_start_slot(context, outer_type, &
                        component_index), c_int64_t)), element_base, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine resolve_nested_component_base

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

    integer function derived_component_access_kind(arena, node, context) &
            result(kind)
        ! Scalar value kind of the component named by a component_access node.
        ! Falls back to VALUE_I32 when the base or component cannot be resolved
        ! (the i32 path then reports the real error). Does not emit IR, so it is
        ! safe to call before the address is materialised.
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: type_index, component_index

        kind = VALUE_I32
        type_index = component_base_type_index(arena, node%base_expr_index, context)
        if (type_index <= 0) return
        component_index = find_derived_component(context, type_index, &
                                                 node%component_name)
        if (component_index <= 0) return
        kind = component_kind(context, type_index, component_index)
    end function derived_component_access_kind

    integer function derived_component_char_length(context, type_index, &
            comp_index) result(n)
        ! Declared character(len=N) length of a component; 0 when comp_index
        ! is out of range or the component is not VALUE_CHARACTER.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: comp_index

        n = 0
        if (.not. allocated(context%derived_types(type_index)% &
                            component_char_length)) return
        if (comp_index < 1 .or. comp_index > &
            size(context%derived_types(type_index)%component_char_length)) return
        n = context%derived_types(type_index)%component_char_length(comp_index)
    end function derived_component_char_length

    subroutine char_component_operands(arena, node, context, data_ptr, length, &
                                       error_msg)
        ! (data pointer, i32 length) for a fixed-length character component
        ! access, mirroring char_length_operands for a plain character symbol.
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index, comp_index, char_len

        type_index = component_base_type_index(arena, node%base_expr_index, context)
        comp_index = 0
        if (type_index > 0) comp_index = find_derived_component(context, &
            type_index, node%component_name)
        if (type_index <= 0 .or. comp_index <= 0) then
            error_msg = 'derived type character component expression is not '// &
                        'declared: '//trim(node%component_name)
            return
        end if
        char_len = derived_component_char_length(context, type_index, comp_index)
        if (char_len <= 0) then
            error_msg = 'derived type character component expression has no '// &
                        'fixed length'
            return
        end if
        call lower_derived_component_address(arena, node, context, data_ptr, &
            error_msg, 'derived type character component expression')
        if (len_trim(error_msg) > 0) return
        length = i32_immediate(context%session, int(char_len, c_int64_t))
    end subroutine char_component_operands

    subroutine store_char_component_value(arena, value_index, addr, dest_len, &
                                          context, error_msg)
        ! Copy a scalar character expression into a fixed-length component's
        ! inline byte storage: blank-pad to dest_len, truncate on overflow, and
        ! NUL-terminate the reserved spare byte at addr+dest_len so the stored
        ! text also prints and concatenates like any other character operand.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lr_operand_desc_t), intent(in) :: addr
        integer, intent(in) :: dest_len
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dest_len_i32, src_data, src_len, cmp
        type(lr_operand_desc_t) :: copy_len, copy_len64, offset

        dest_len_i32 = i32_immediate(context%session, int(dest_len, c_int64_t))
        call char_expr_operands(arena, value_index, context, src_data, src_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call fill_spaces(context, addr, dest_len_i32, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLE, src_len, &
                dest_len_i32, cmp, error_msg)) return
        call select_value(context, cmp, src_len, dest_len_i32, copy_len, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_to_i64(context%session, copy_len, copy_len64, &
                error_msg)) return
        if (.not. emit_memcpy(context%session, addr, src_data, copy_len64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_ADD, addr, &
                i64_immediate(context%session, int(dest_len, c_int64_t)), offset, &
                error_msg)) return
        if (.not. emit_i8_store(context%session, &
                i8_immediate(context%session, 0_c_int64_t), offset, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine store_char_component_value

    recursive function component_base_type_index(arena, base_index, context) &
            result(type_index)
        ! Derived type index of a component-access base (a plain derived
        ! identifier, a derived-array element arr(i), or a nested derived
        ! component x%c) without emitting IR.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: base_index
        type(lowering_context_t), intent(in) :: context
        integer :: type_index
        integer :: symbol_index, outer_type, component_index
        character(len=:), allocatable :: id_name, err

        type_index = 0
        if (.not. node_exists(arena, base_index)) return
        symbol_index = 0
        if (is_identifier(arena, base_index)) then
            call get_identifier_name(arena, base_index, id_name, err)
            if (len_trim(err) > 0) return
            symbol_index = find_symbol(context, id_name)
        else
            select type (base => arena%entries(base_index)%node)
            type is (call_or_subscript_node)
                if (allocated(base%name)) symbol_index = find_symbol(context, base%name)
            type is (component_access_node)
                outer_type = component_base_type_index(arena, &
                                                       base%base_expr_index, context)
                if (outer_type <= 0) return
                component_index = find_derived_component(context, outer_type, &
                                                         base%component_name)
                if (component_index <= 0) return
                type_index = context%derived_types(outer_type)% &
                             component_type_index(component_index)
                return
            end select
        end if
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_derived) return
        type_index = context%symbols(symbol_index)%derived_type_index
    end function component_base_type_index

    subroutine emit_typed_component_store(context, comp_kind, value, addr, &
                                          error_msg)
        ! Store a component value through its address using the load/store width
        ! that matches the component kind.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: comp_kind
        type(lr_operand_desc_t), intent(in) :: value
        type(lr_operand_desc_t), intent(in) :: addr
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: ok

        if (comp_kind == VALUE_CHARACTER) then
            error_msg = 'character component access is not supported by direct '// &
                        'LIRIC session'
            return
        end if
        select case (comp_kind)
        case (VALUE_F64)
            ok = emit_liric_f64_store(context%session, value, addr, error_msg)
        case (VALUE_F32)
            ok = emit_liric_f32_store(context%session, value, addr, error_msg)
        case default
            ok = emit_i32_store(context%session, value, addr, error_msg)
        end select
        if (.not. ok) error_msg = 'component_store: '//error_msg
    end subroutine emit_typed_component_store

    subroutine emit_typed_component_load(context, comp_kind, addr, value, &
                                         error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: comp_kind
        type(lr_operand_desc_t), intent(in) :: addr
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        logical :: ok

        if (comp_kind == VALUE_CHARACTER) then
            error_msg = 'character component access is not supported by direct '// &
                        'LIRIC session'
            return
        end if
        select case (comp_kind)
        case (VALUE_F64)
            ok = emit_liric_f64_load(context%session, addr, value, error_msg)
        case (VALUE_F32)
            ok = emit_liric_f32_load(context%session, addr, value, error_msg)
        case default
            ok = emit_i32_load(context%session, addr, value, error_msg)
        end select
        if (.not. ok) error_msg = 'component_load: '//error_msg
    end subroutine emit_typed_component_load
