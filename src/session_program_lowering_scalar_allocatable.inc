    subroutine lower_scalar_allocatable_declaration(node, context, value_kind, &
                                                    error_msg)
        ! integer/real/logical, allocatable :: x (no array spec, W11). The
        ! variable starts unassociated (no storage); allocate(x) mallocs a
        ! typed slot and adopts it as the symbol's address, reusing the
        ! pointer-style has_address/is_reference/address trio so every
        ! existing scalar read/write/print path works unchanged.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
            value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL) then
            call unsupported_feature_error('allocatable declaration', &
                node%line, node%column, &
                'direct LIRIC session supports scalar integer, real, and '// &
                'logical allocatables only', error_msg)
            return
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call declare_scalar_allocatable(context, node%var_names(i), &
                                                value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call declare_scalar_allocatable(context, node%var_name, value_kind, &
                                            error_msg)
        else
            error_msg = 'allocatable declaration did not expose a variable name'
        end if
    end subroutine lower_scalar_allocatable_declaration

    subroutine declare_scalar_allocatable(context, name, value_kind, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        index = find_symbol_compat(context, name)
        if (index > 0) then
            if (context%symbols(index)%is_parameter .and. &
                .not. context%symbols(index)%is_array .and. &
                .not. context%symbols(index)%is_allocatable) then
                ! A scalar allocatable dummy argument: the caller's value has
                ! no stable memory slot to alias (a local scalar allocatable
                ! reuses its malloc'd data pointer directly as its address, W11
                ! scope), so this ABI is not yet supported.
                call unsupported_feature_error('allocatable declaration', &
                    0, 0, 'direct LIRIC session does not support a scalar '// &
                    'allocatable dummy argument: '//trim(name), error_msg)
                return
            end if
            error_msg = 'duplicate allocatable declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_allocatable = .true.
        context%symbols(index)%has_address = .false.
        context%symbols(index)%is_reference = .false.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine declare_scalar_allocatable

    subroutine lower_allocate_scalar(symbol_index, context, error_msg)
        ! allocate(x) for a scalar integer/real/logical allocatable: malloc a
        ! typed slot and adopt it as the variable's storage. Mirrors
        ! emit_pointer_target_slot but without an initial value store, since a
        ! bare allocate leaves the contents undefined per the standard.
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_ptr

        if (context%symbols(symbol_index)%is_derived) then
            call lower_allocate_scalar_derived(symbol_index, context, error_msg)
            return
        end if
        if (.not. emit_malloc(context%session, &
                i64_immediate(context%session, &
                    allocatable_elem_size(context%symbols(symbol_index)%value_kind)), &
                data_ptr, error_msg)) return
        context%symbols(symbol_index)%address = data_ptr
        context%symbols(symbol_index)%has_address = .true.
        context%symbols(symbol_index)%is_reference = .true.
        call set_empty(error_msg)
    end subroutine lower_allocate_scalar

    subroutine declare_scalar_allocatable_derived(context, node, &
                                                  derived_type_index, error_msg)
        ! type(t), allocatable :: x: a scalar allocatable derived variable starts
        ! unassociated with no storage. allocate(x) mallocs one instance and adopts
        ! the heap pointer as element_address, so every existing component
        ! read/write/print path reaches the heap through the same base pointer a
        ! stack-allocated derived instance would use.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        logical :: is_class

        ! class(t), allocatable is polymorphic: it gets the canonical class
        ! descriptor so ALLOCATE can record a dynamic type that differs from
        ! the declared type (#421). type(t), allocatable keeps the plain
        ! heap-pointer form.
        is_class = .false.
        if (allocated(node%type_name)) &
            is_class = is_class_derived_type_spec(node%type_name)

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call declare_one_scalar_allocatable_derived(context, &
                    node%var_names(i), derived_type_index, is_class, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call declare_one_scalar_allocatable_derived(context, node%var_name, &
                derived_type_index, is_class, error_msg)
        else
            error_msg = 'allocatable derived declaration did not expose a name'
        end if
    end subroutine declare_scalar_allocatable_derived

    subroutine declare_one_scalar_allocatable_derived(context, name, &
                                                      derived_type_index, &
                                                      is_class, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: derived_type_index
        logical, intent(in) :: is_class
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        index = find_symbol_compat(context, name)
        if (index > 0) then
            ! A pre-bound allocatable derived result (class(t)/type(t) function
            ! result, or a re-declaration of an already-defined instance): its
            ! storage is caller-supplied, so reuse it rather than reserving a
            ! deferred heap slot. Mirrors define_derived_symbol's result guard.
            if (context%symbols(index)%is_derived .and. &
                context%symbols(index)%derived_type_index == derived_type_index) then
                call set_empty(error_msg)
                return
            end if
            ! A scalar allocatable derived dummy: the parameter pointer already
            ! aliases the caller's storage. Re-declaring it in the body just
            ! records the derived type so component access can resolve.
            if (context%symbols(index)%is_parameter .and. &
                .not. context%symbols(index)%is_derived) then
                context%symbols(index)%value_kind = VALUE_DERIVED
                context%symbols(index)%is_derived = .true.
                context%symbols(index)%is_allocatable = .true.
                context%symbols(index)%array_rank = 0
                context%symbols(index)%derived_type_index = derived_type_index
                context%symbols(index)%element_address = &
                    context%symbols(index)%address
                context%symbols(index)%has_address = .true.
                call set_empty(error_msg)
                return
            end if
            error_msg = 'duplicate allocatable derived declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_DERIVED
        context%symbols(index)%is_derived = .true.
        context%symbols(index)%is_allocatable = .true.
        context%symbols(index)%array_rank = 0
        context%symbols(index)%derived_type_index = derived_type_index
        context%symbols(index)%has_address = .false.
        context%symbol_count = index
        call set_empty(error_msg)
        if (.not. is_class) return
        call bind_class_allocatable_descriptor(context, index, &
                                               derived_type_index, error_msg)
    end subroutine declare_one_scalar_allocatable_derived

    subroutine lower_allocate_scalar_derived(symbol_index, context, error_msg, &
                                             dynamic_type_index)
        ! allocate(x) for a scalar allocatable derived variable: malloc one
        ! instance (total_component_slots i32 slots), adopt the heap pointer as
        ! element_address, and run the type's default-init so component defaults
        ! and nested allocatable descriptors start in their declared state.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: dynamic_type_index
        type(lr_operand_desc_t) :: data_ptr
        integer :: type_index, slots

        ! Without a type-spec the dynamic type of a class allocatable is its
        ! declared type; allocate(ext_t :: p) names an extension instead and
        ! allocates that concrete layout (#421).
        type_index = context%symbols(symbol_index)%derived_type_index
        if (present(dynamic_type_index)) then
            if (dynamic_type_index > 0) type_index = dynamic_type_index
        end if
        slots = total_component_slots(context, type_index)
        if (slots <= 0) slots = 1
        if (.not. emit_malloc(context%session, &
                i64_immediate(context%session, int(slots, c_int64_t)*4_c_int64_t), &
                data_ptr, error_msg)) return
        context%symbols(symbol_index)%element_address = data_ptr
        context%symbols(symbol_index)%array_size = slots
        context%symbols(symbol_index)%has_address = .true.
        call init_derived_component_defaults(context, symbol_index, type_index, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        call record_class_allocation(context, symbol_index, data_ptr, &
            i64_immediate(context%session, int(type_index, c_int64_t)), error_msg)
    end subroutine lower_allocate_scalar_derived

    subroutine lower_deallocate_scalar(symbol_index, context, error_msg)
        ! deallocate(x) for a scalar allocatable: free the current storage and
        ! mark the variable unassociated again.
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. context%symbols(symbol_index)%has_address) then
            error_msg = 'deallocate target is not currently allocated'
            return
        end if
        if (context%symbols(symbol_index)%is_derived) then
            ! Deallocation ends the value's lifetime: finalize before the
            ! storage is released (#403).
            call emit_scalar_finalizer_call(context, symbol_index, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_free(context%session, &
                    context%symbols(symbol_index)%element_address, error_msg)) return
            context%symbols(symbol_index)%has_address = .false.
            ! The released address must not stay reachable through the class
            ! descriptor, so ownership is given up exactly once (#421).
            call release_class_allocation(context, symbol_index, error_msg)
            return
        end if
        if (.not. emit_free(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
        context%symbols(symbol_index)%has_address = .false.
        context%symbols(symbol_index)%is_reference = .false.
        call set_empty(error_msg)
    end subroutine lower_deallocate_scalar

    subroutine lower_scalar_allocate_mold_source(arena, node, target_sym, context, &
                                                 error_msg)
        ! allocate(x, source=<expr>) / allocate(x, mold=<expr>) for a scalar
        ! integer/real/logical allocatable target: malloc a typed slot and, for
        ! SOURCE, store the source expression's value (any expression, not just
        ! an identifier) into it using the target's element kind. MOLD leaves
        ! the slot's value undefined.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        integer, intent(in) :: target_sym
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: value

        ! A derived or class allocatable takes the whole-value SOURCE=/MOLD=
        ! path: its dynamic type and exact storage size come from the source
        ! value rather than from the target's element kind (#421).
        if (context%symbols(target_sym)%is_derived) then
            call lower_derived_allocate_source(arena, node, target_sym, context, &
                                               error_msg)
            return
        end if
        call lower_allocate_scalar(target_sym, context, error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%source_expr_index <= 0) then
            call set_empty(error_msg)
            return
        end if
        call lower_allocatable_element_value(arena, node%source_expr_index, &
            context%symbols(target_sym)%value_kind, context, value, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_allocatable_element_store(context, &
                context%symbols(target_sym)%value_kind, value, &
                context%symbols(target_sym)%address, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_scalar_allocate_mold_source
