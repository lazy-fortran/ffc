    subroutine lower_scalar_pointer_target(node, context, value_kind, error_msg)
        ! Scalar integer/real/logical POINTER/TARGET (#245 slice B3a). A target
        ! is given a stable in-memory slot so a pointer can address it; a pointer
        ! starts unassociated.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_scalar_pointer_target_symbol(context, node, &
                                              node%var_names(i), value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_scalar_pointer_target_symbol(context, node, node%var_name, &
                                                     value_kind, error_msg)
        else
            error_msg = 'pointer/target declaration did not expose a variable name'
        end if
    end subroutine lower_scalar_pointer_target

    subroutine lower_pointer_target_array(node, context, error_msg)
        ! Rank-1 fixed-size integer/real/logical POINTER/TARGET arrays (#245 B3a).
        ! A target array gets ordinary explicit-shape storage; a deferred-shape
        ! pointer array (p(:)) starts unassociated and adopts a target array's
        ! base, shape, and element kind at `p => t`. Anything outside this subset
        ! (rank-2, allocatable, derived, character, bounds-remapped pointers)
        ! keeps the prior unsupported diagnostic.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: value_kind, dim_count, array_lower_bound, array_size, i

        if (node%is_allocatable) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if
        dim_count = 0
        if (allocated(node%dimension_indices)) &
            dim_count = size(node%dimension_indices)
        if (dim_count /= 1) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if
        call declaration_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (value_kind /= VALUE_I32 .and. value_kind /= VALUE_F32 .and. &
            value_kind /= VALUE_F64 .and. value_kind /= VALUE_LOGICAL) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if

        ! A pointer array is deferred-shape p(:): no compile-time storage, just a
        ! stub symbol that `p => t` fills in. A target array is explicit-shape and
        ! is laid out exactly like an ordinary fixed-size array.
        if (node%is_pointer .and. declaration_is_assumed_shape(node, context)) then
            if (node%is_multi_declaration .and. allocated(node%var_names)) then
                do i = 1, size(node%var_names)
                    call define_pointer_array_stub(context, node%var_names(i), &
                                                   value_kind, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(node%var_name)) then
                call define_pointer_array_stub(context, node%var_name, &
                                               value_kind, error_msg)
            else
                error_msg = 'pointer array declaration did not expose a name'
            end if
            return
        end if

        ! Explicit-shape target array (or an explicit-shape pointer, which we
        ! treat like a target with storage). A deferred-shape target is invalid
        ! Fortran, so the only remaining shape here is explicit rank-1.
        if (declaration_is_assumed_shape(node, context)) then
            call unsupported_pointer_target_array(node, error_msg)
            return
        end if
        call get_array_bounds(node, context, array_lower_bound, array_size, &
                              error_msg)
        if (len_trim(error_msg) > 0) return
        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_target_array_symbol(context, node, node%var_names(i), &
                                                array_lower_bound, array_size, &
                                                value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_target_array_symbol(context, node, node%var_name, &
                                            array_lower_bound, array_size, &
                                            value_kind, error_msg)
        else
            error_msg = 'target array declaration did not expose a name'
        end if
    end subroutine lower_pointer_target_array

    subroutine unsupported_pointer_target_array(node, error_msg)
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: error_msg

        call unsupported_feature_error('pointer/target array declaration', &
            node%line, node%column, &
            'direct LIRIC session supports rank-1 fixed-size integer, real, '// &
            'and logical pointer/target arrays only (#245 slice B3a)', error_msg)
    end subroutine unsupported_pointer_target_array

    subroutine define_target_array_symbol(context, node, name, array_lower_bound, &
                                          array_size, value_kind, error_msg)
        ! Lay out a fixed-size target array exactly like an ordinary array, then
        ! mark it as an addressable target so a pointer can alias its storage.
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: array_lower_bound
        integer, intent(in) :: array_size
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        call define_declared_array_symbol(context, node, name, array_lower_bound, &
                                          array_size, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        index = find_symbol(context, name)
        if (index <= 0) then
            error_msg = 'target array declaration lost its symbol: '//trim(name)
            return
        end if
        if (node%is_target) context%symbols(index)%is_target = .true.
        if (node%is_pointer) then
            context%symbols(index)%is_pointer = .true.
            context%symbols(index)%is_associated = .true.
        end if
        ! element_address is the array base; expose it as the alias address so
        ! the shared `p => t` path can adopt it.
        context%symbols(index)%address = context%symbols(index)%element_address
        context%symbols(index)%has_address = .true.
    end subroutine define_target_array_symbol

    subroutine define_pointer_array_stub(context, name, value_kind, error_msg)
        ! A deferred-shape pointer array p(:): no storage and no shape until a
        ! `p => t` association supplies the target's base and extent.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        index = find_symbol(context, name)
        if (index > 0) then
            ! Re-declaration (e.g. dummy refresh): just record the attributes.
            context%symbols(index)%is_pointer = .true.
            context%symbols(index)%is_array = .true.
            context%symbols(index)%is_associated = .false.
            call set_empty(error_msg)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = value_kind
        context%symbols(index)%is_array = .true.
        context%symbols(index)%array_rank = 1
        context%symbols(index)%array_size = 0
        context%symbols(index)%is_pointer = .true.
        context%symbols(index)%is_associated = .false.
        context%symbols(index)%has_address = .false.
        context%symbols(index)%is_reference = .false.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_pointer_array_stub

    subroutine define_scalar_pointer_target_symbol(context, node, name, value_kind, &
                                                   error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_kind
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
        call define_symbol(context, name, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'pointer/target declaration lost its symbol: '//trim(name)
            return
        end if
        if (node%is_target) then
            ! Materialise the target in memory so a pointer can hold its address.
            context%symbols(symbol_index)%is_target = .true.
            call emit_pointer_target_slot(context, symbol_index, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
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

    subroutine emit_pointer_target_slot(context, symbol_index, value_kind, error_msg)
        ! Allocate a typed slot for a target scalar and store its initial value,
        ! so a pointer can later adopt the slot address (deref via the typed
        ! is_reference load/store paths).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        select case (value_kind)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_i32_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_liric_f32_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_alloca(context%session, &
                context%symbols(symbol_index)%address, error_msg)) return
            if (.not. emit_liric_f64_store(context%session, &
                context%symbols(symbol_index)%value, &
                context%symbols(symbol_index)%address, error_msg)) return
        case default
            error_msg = 'unsupported target scalar kind for direct LIRIC session'
            return
        end select
        call set_empty(error_msg)
    end subroutine emit_pointer_target_slot

    subroutine lower_pointer_assignment(arena, node, context, error_msg)
        ! p => t : the pointer adopts the target's storage address.
        ! Dispatches to lower_proc_pointer_assignment when p is a procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        integer :: ptr_index, target_index

        call get_identifier_name(arena, node%pointer_index, ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_index = find_symbol(context, ptr_name)
        if (ptr_index > 0 .and. context%symbols(ptr_index)%is_proc_pointer) then
            call lower_proc_pointer_assignment(arena, node, context, error_msg)
            return
        end if
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
                        'scalar or whole-array target variable'
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
        ! Whole-array p => t: the pointer adopts the target array's base,
        ! shape, and element kind. The bounds-remapped form p(lo:hi) => t is
        ! out of scope and carries pointer bounds on the LHS, which we reject.
        if (context%symbols(ptr_index)%is_array .or. &
            context%symbols(target_index)%is_array) then
            if (.not. is_identifier(arena, node%pointer_index)) then
                error_msg = 'direct LIRIC session pointer assignment does not '// &
                            'support pointer bounds remapping'
                return
            end if
            call alias_pointer_array(context, ptr_index, target_index, error_msg)
            return
        end if
        context%symbols(ptr_index)%address = context%symbols(target_index)%address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        call set_empty(error_msg)
    end subroutine lower_pointer_assignment

    subroutine alias_pointer_array(context, ptr_index, target_index, error_msg)
        ! Whole-array p => t: the pointer array adopts the target array's base,
        ! extent, element kind, and bounds, so element access through the pointer
        ! GEPs the same storage. Both sides must be rank-1 fixed-size arrays of
        ! the same element kind (the supported subset of #245 B3a).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: ptr_index, target_index
        character(len=:), allocatable :: error_msg
        integer :: i

        if (.not. context%symbols(ptr_index)%is_array) then
            error_msg = 'left of => is a scalar pointer but the target is an array'
            return
        end if
        if (.not. context%symbols(target_index)%is_array) then
            error_msg = 'left of => is an array pointer but the target is a scalar'
            return
        end if
        if (context%symbols(ptr_index)%array_rank /= &
            context%symbols(target_index)%array_rank) then
            error_msg = 'pointer and target array ranks do not match'
            return
        end if
        if (context%symbols(ptr_index)%value_kind /= &
            context%symbols(target_index)%value_kind) then
            error_msg = 'pointer and target array element kinds do not match'
            return
        end if
        ! Adopt the target's element storage and shape. element_address is the
        ! array base used by the GEP element-addressing path; array_size and the
        ! per-dimension shape drive size(), bounds, and whole-array print.
        context%symbols(ptr_index)%element_address = &
            context%symbols(target_index)%element_address
        context%symbols(ptr_index)%array_size = &
            context%symbols(target_index)%array_size
        context%symbols(ptr_index)%array_lower_bound = &
            context%symbols(target_index)%array_lower_bound
        do i = 1, 2
            context%symbols(ptr_index)%array_dim_sizes(i) = &
                context%symbols(target_index)%array_dim_sizes(i)
            context%symbols(ptr_index)%array_dim_lowers(i) = &
                context%symbols(target_index)%array_dim_lowers(i)
        end do
        context%symbols(ptr_index)%address = &
            context%symbols(target_index)%address
        context%symbols(ptr_index)%has_address = .true.
        context%symbols(ptr_index)%is_reference = .true.
        context%symbols(ptr_index)%is_associated = .true.
        call set_empty(error_msg)
    end subroutine alias_pointer_array

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
        ! associated(p) / associated(p, t): compile-time association check
        ! (#245 B3a / B3c). One-argument form folds to the `is_associated` flag.
        ! Two-argument form additionally checks that p and t point to the same
        ! target (identical address operand in compile-time straight-line code).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        integer :: ptr_idx, tgt_idx
        logical :: same_target

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'associated requires at least one argument'
            return
        end if
        if (size(arg_indices) < 1 .or. size(arg_indices) > 2) then
            error_msg = 'associated requires one or two arguments'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = 'associated first argument must be a pointer variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_idx = find_symbol(context, ptr_name)
        if (ptr_idx <= 0) then
            error_msg = 'associated argument was not declared: '//trim(ptr_name)
            return
        end if
        if (.not. context%symbols(ptr_idx)%is_pointer .and. &
            .not. context%symbols(ptr_idx)%is_proc_pointer) then
            error_msg = 'associated argument is not a pointer: '//trim(ptr_name)
            return
        end if
        if (size(arg_indices) == 1) then
            ! One-argument: fold to compile-time association flag.
            if (context%symbols(ptr_idx)%is_associated) then
                value = i32_immediate(context%session, 1_c_int64_t)
            else
                value = i32_immediate(context%session, 0_c_int64_t)
            end if
            return
        end if
        ! Two-argument form: also require same target (B3c).
        if (.not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'associated second argument must be a target variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(2), target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        tgt_idx = find_symbol(context, target_name)
        if (tgt_idx <= 0) then
            error_msg = 'associated second argument was not declared: '// &
                        trim(target_name)
            return
        end if
        ! For compile-time straight-line code compare address vreg payloads.
        same_target = .false.
        if (context%symbols(ptr_idx)%is_associated .and. &
            context%symbols(tgt_idx)%has_address) then
            same_target = &
                context%symbols(ptr_idx)%address%payload == &
                context%symbols(tgt_idx)%address%payload
        end if
        if (same_target) then
            value = i32_immediate(context%session, 1_c_int64_t)
        else
            value = i32_immediate(context%session, 0_c_int64_t)
        end if
    end subroutine lower_associated

    subroutine lower_proc_pointer_declaration(node, context, error_msg)
        ! procedure(iface), pointer :: fp => null() (#245 slice B3d).
        ! A procedure pointer is stored as a ptr alloca slot; the callee address
        ! is loaded from that slot at each indirect call site.
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_proc_pointer_symbol(context, node%var_names(i), &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_proc_pointer_symbol(context, node%var_name, error_msg)
        else
            error_msg = 'procedure pointer declaration has no variable name'
        end if
    end subroutine lower_proc_pointer_declaration

    subroutine define_proc_pointer_symbol(context, name, error_msg)
        ! Allocate a ptr slot on the stack; the slot holds the function address.
        ! Initially null (zeroed by alloca semantics at declaration).
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: idx

        idx = find_symbol(context, name)
        if (idx > 0) then
            ! Already declared (e.g. a dummy re-declaration); benign.
            call set_empty(error_msg)
            return
        end if
        call grow_symbols(context)
        idx = context%symbol_count + 1
        context%symbols(idx)%name = trim(name)
        context%symbols(idx)%value_kind = VALUE_PROC_PTR
        context%symbols(idx)%is_pointer = .true.
        context%symbols(idx)%is_proc_pointer = .true.
        context%symbols(idx)%is_associated = .false.
        ! Allocate a ptr slot on the stack; address stores the slot address.
        if (.not. emit_ptr_alloca(context%session, &
                                  context%symbols(idx)%address, error_msg)) return
        context%symbols(idx)%has_address = .true.
        context%symbol_count = idx
        call set_empty(error_msg)
    end subroutine define_proc_pointer_symbol

    subroutine lower_proc_pointer_assignment(arena, node, context, error_msg)
        ! fp => my_func : store the function address into fp's ptr slot.
        ! Also handles fp => null() which clears the association.
        type(ast_arena_t), intent(in) :: arena
        type(pointer_assignment_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ptr_name, target_name
        integer :: ptr_idx, target_idx
        type(lr_operand_desc_t) :: callee_global
        integer(c_int32_t) :: symbol_id
        character(kind=c_char), allocatable :: c_name(:)

        call get_identifier_name(arena, node%pointer_index, ptr_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_idx = find_symbol(context, ptr_name)
        if (ptr_idx <= 0) then
            error_msg = 'procedure pointer was not declared: '//trim(ptr_name)
            return
        end if
        if (.not. context%symbols(ptr_idx)%is_proc_pointer) then
            error_msg = 'left of => is not a procedure pointer: '//trim(ptr_name)
            return
        end if
        if (.not. is_identifier(arena, node%target_index)) then
            error_msg = 'procedure pointer assignment requires a procedure name'
            return
        end if
        call get_identifier_name(arena, node%target_index, target_name, error_msg)
        if (len_trim(error_msg) > 0) return
        ! null() clears the association.
        if (same_name(target_name, 'null')) then
            context%symbols(ptr_idx)%is_associated = .false.
            call set_empty(error_msg)
            return
        end if
        ! Check whether target is another procedure pointer being copied.
        target_idx = find_symbol(context, target_name)
        if (target_idx > 0 .and. context%symbols(target_idx)%is_proc_pointer) then
            ! Copy the callee ptr from the source slot into the destination slot.
            if (.not. emit_ptr_load(context%session, &
                                    context%symbols(target_idx)%address, &
                                    callee_global, error_msg)) return
            if (.not. emit_ptr_store(context%session, callee_global, &
                                     context%symbols(ptr_idx)%address, &
                                     error_msg)) return
            context%symbols(ptr_idx)%is_associated = &
                context%symbols(target_idx)%is_associated
            call set_empty(error_msg)
            return
        end if
        ! Target is a named procedure: intern its symbol and store as ptr.
        call to_c_chars(trim(call_emit_name(context%arena, target_name)), c_name)
        symbol_id = lr_session_intern(context%session%handle, c_name)
        if (symbol_id < 0_c_int32_t) then
            error_msg = 'could not intern procedure symbol: '//trim(target_name)
            return
        end if
        callee_global%kind = LR_OP_KIND_GLOBAL
        callee_global%payload = int(symbol_id, c_int64_t)
        callee_global%typ = lr_type_ptr_s(context%session%handle)
        callee_global%global_offset = 0_c_int64_t
        if (.not. emit_ptr_store(context%session, callee_global, &
                                 context%symbols(ptr_idx)%address, error_msg)) return
        context%symbols(ptr_idx)%is_associated = .true.
        call set_empty(error_msg)
    end subroutine lower_proc_pointer_assignment

    logical function is_proc_pointer_call(context, name) result(ok)
        ! Return .true. when name is a declared procedure pointer symbol.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: idx

        idx = find_symbol(context, name)
        ok = idx > 0 .and. context%symbols(idx)%is_proc_pointer
    end function is_proc_pointer_call

    subroutine lower_i32_proc_ptr_call(arena, node, context, value, error_msg)
        ! Indirect integer function call through a procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: callee_ptr
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: idx

        idx = find_symbol(context, node%name)
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                callee_ptr, error_msg)) return
        if (allocated(node%arg_indices)) then
            call prepare_reference_args(arena, node%arg_indices, context, &
                                        VALUE_I32, node%name, args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (args(0))
            allocate (copyback_indices(0))
        end if
        if (.not. emit_i32_indirect_call(context%session, callee_ptr, args, &
                                         value, error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_i32_proc_ptr_call

    subroutine lower_void_proc_ptr_call(arena, node_index, context, error_msg)
        ! Indirect void (subroutine) call through a procedure pointer.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer, allocatable :: arg_indices(:)
        type(lr_operand_desc_t) :: callee_ptr
        type(lr_operand_desc_t), allocatable :: args(:)
        integer, allocatable :: copyback_indices(:)
        integer :: idx

        call get_subroutine_call_name(arena, node_index, name, error_msg)
        if (len_trim(error_msg) > 0) return
        idx = find_symbol(context, name)
        if (.not. emit_ptr_load(context%session, context%symbols(idx)%address, &
                                callee_ptr, error_msg)) return
        call get_subroutine_call_arg_indices(arena, node_index, arg_indices, &
                                             error_msg)
        if (len_trim(error_msg) > 0) return
        call prepare_reference_args(arena, arg_indices, context, VALUE_I32, &
                                    name, args, copyback_indices, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_void_indirect_call(context%session, callee_ptr, args, &
                                          error_msg)) return
        call copy_back_reference_args(context, args, copyback_indices, error_msg)
    end subroutine lower_void_proc_ptr_call
