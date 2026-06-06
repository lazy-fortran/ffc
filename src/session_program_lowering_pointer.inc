    subroutine lower_scalar_pointer_target(node, context, error_msg)
        ! Scalar integer POINTER/TARGET (#245 slice B3a). A target is given a
        ! stable in-memory slot so a pointer can address it; a pointer starts
        ! unassociated.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_scalar_pointer_target_symbol(context, node, &
                                                         node%var_names(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_scalar_pointer_target_symbol(context, node, node%var_name, &
                                                     error_msg)
        else
            error_msg = 'pointer/target declaration did not expose a variable name'
        end if
    end subroutine lower_scalar_pointer_target

    subroutine define_scalar_pointer_target_symbol(context, node, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index

        symbol_index = find_symbol(context, name)
        if (symbol_index > 0) then
            ! A dummy argument already has its own pass-by-reference storage, so a
            ! `target` attribute on it needs no new slot; just record the flag.
            ! A genuine local pointer/target re-declaration is benign here too.
            if (node%is_target) context%symbols(symbol_index)%is_target = .true.
            if (node%is_pointer) then
                context%symbols(symbol_index)%is_pointer = .true.
                context%symbols(symbol_index)%is_associated = .false.
            end if
            call set_empty(error_msg)
            return
        end if
        call define_i32_symbol(context, name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'pointer/target declaration lost its symbol: '//trim(name)
            return
        end if
        if (node%is_target) then
            ! Materialise the target in memory so a pointer can hold its address.
            context%symbols(symbol_index)%is_target = .true.
            if (.not. emit_i32_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_i32_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
            context%symbols(symbol_index)%has_address = .true.
            context%symbols(symbol_index)%is_reference = .true.
        else
            ! A pointer starts unassociated; `p => t` fills in the slot address.
            context%symbols(symbol_index)%is_pointer = .true.
            context%symbols(symbol_index)%is_associated = .false.
            context%symbols(symbol_index)%has_address = .false.
            context%symbols(symbol_index)%is_reference = .false.
        end if
        call set_empty(error_msg)
    end subroutine define_scalar_pointer_target_symbol

    subroutine lower_pointer_assignment(arena, node, context, error_msg)
        ! p => t : the pointer adopts the target's storage address.
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        integer :: ptr_index, target_index

        call get_identifier_name(arena, node%pointer_index, ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_index = find_symbol(context, ptr_name)
        if (ptr_index <= 0) then
            error_msg = 'pointer assignment target was not declared: '//trim(ptr_name)
            return
        end if
        if (.not. context%symbols(ptr_index)%is_pointer) then
            error_msg = 'left of => is not a pointer: '//trim(ptr_name)
            return
        end if
        if (.not. is_identifier(arena, node%target_index)) then
            error_msg = 'direct LIRIC session pointer assignment requires a '// &
                        'scalar target variable'
            return
        end if
        call get_identifier_name(arena, node%target_index, target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        target_index = find_symbol(context, target_name)
        if (target_index <= 0) then
            error_msg = 'pointer assignment source was not declared: '// &
                        trim(target_name)
            return
        end if
        if (.not. (context%symbols(target_index)%is_target .or. &
                   context%symbols(target_index)%is_pointer)) then
            error_msg = 'right of => is not a target or pointer: '// &
                        trim(target_name)
            return
        end if
        if (.not. context%symbols(target_index)%has_address) then
            error_msg = 'pointer assignment source has no storage: '// &
                        trim(target_name)
            return
        end if
        context%symbols(ptr_index)%address = context%symbols(target_index)%address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        call set_empty(error_msg)
    end subroutine lower_pointer_assignment

    subroutine lower_nullify(arena, node, context, error_msg)
        ! nullify(p [, q ...]) : disassociate each pointer.
        type(ast_arena_t), intent(in) :: arena
        type(nullify_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: i, symbol_index

        call set_empty(error_msg)
        if (.not. allocated(node%pointer_indices)) then
            error_msg = 'nullify has no pointer arguments'
            return
        end if
        do i = 1, size(node%pointer_indices)
            call get_identifier_name(arena, node%pointer_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            symbol_index = find_symbol(context, name)
            if (symbol_index <= 0) then
                error_msg = 'nullify argument was not declared: '//trim(name)
                return
            end if
            if (.not. context%symbols(symbol_index)%is_pointer) then
                error_msg = 'nullify argument is not a pointer: '//trim(name)
                return
            end if
            context%symbols(symbol_index)%is_associated = .false.
            context%symbols(symbol_index)%has_address = .false.
            context%symbols(symbol_index)%is_reference = .false.
        end do
    end subroutine lower_nullify

    subroutine lower_associated(arena, arg_indices, context, value, error_msg)
        ! associated(p) : a compile-time logical for straight-line association
        ! state (#245 slice B3a). The two-argument form lands in B3c.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'associated requires one argument'
            return
        end if
        if (size(arg_indices) /= 1) then
            error_msg = 'direct LIRIC session supports only one-argument '// &
                        'associated (#245 slice B3a)'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = 'associated argument must be a pointer variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'associated argument was not declared: '//trim(name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_pointer) then
            error_msg = 'associated argument is not a pointer: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%is_associated) then
            value = i32_immediate(context%session, 1_c_int64_t)
        else
            value = i32_immediate(context%session, 0_c_int64_t)
        end if
    end subroutine lower_associated
