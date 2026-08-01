    ! Monomorphic-subset polymorphism (#141). A class(t) scalar dummy or local
    ! whose dynamic type is only ever its declared type t lowers exactly like
    ! type(t): the derived-type declaration path already routes both spellings
    ! (extracted_derived_type_name treats type(t) and class(t) alike). This file
    ! adds SELECT TYPE dispatch for such a declared-type selector: the dynamic
    ! type equals the declared type, so exactly one arm is ever live and is
    ! chosen statically. Full dynamic dispatch on an extensible allocatable /
    ! pointer class object (a runtime vtable / type-id) stays out of scope and
    ! declines gracefully.

    ! Runtime type identity for class scalars (#417). A class(t) scalar dummy is
    ! passed as the canonical scalar class descriptor of docs/RUNTIME_ABI.md
    ! (data, declared_type, dynamic_type, ownership), built by the caller and
    ! borrowed by the callee. The declared type is fixed by the dummy's
    ! declaration; the dynamic type travels with the value, so re-passing a
    ! class dummy onwards preserves the identity of the original actual instead
    ! of collapsing it to the declared type.

    recursive logical function derived_type_extends(context, type_index, &
                                                    ancestor_index) result(yes)
        ! True when type_index is ancestor_index or extends it, transitively.
        ! This is the type-compatibility test of Fortran 2018 7.3.2.3: an actual
        ! is valid for a class(t) dummy exactly when its declared type is t or
        ! an extension of t.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: ancestor_index

        yes = .false.
        if (type_index <= 0) return
        if (ancestor_index <= 0) return
        if (type_index == ancestor_index) then
            yes = .true.
            return
        end if
        if (type_index > context%derived_type_count) return
        if (len_trim(context%derived_types(type_index)%parent_name) == 0) return
        yes = derived_type_extends(context, &
            find_derived_type(context, &
                trim(context%derived_types(type_index)%parent_name)), &
            ancestor_index)
    end function derived_type_extends

    subroutine class_dummy_declared_type_index(context, arena, callee_name, &
                                               param_pos, type_index)
        ! Registered index of the declared type of the callee's param_pos-th
        ! dummy when that dummy is class(t); 0 when it is not.
        type(lowering_context_t), intent(in) :: context
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: param_pos
        integer, intent(out) :: type_index
        character(len=:), allocatable :: type_name

        type_index = 0
        call callee_dummy_class_declared_type(arena, callee_name, param_pos, &
                                              type_name)
        if (len_trim(type_name) == 0) return
        type_index = find_derived_type(context, type_name)
    end subroutine class_dummy_declared_type_index

    subroutine class_actual_dynamic_type(context, symbol_index, dynamic_id, &
                                         error_msg)
        ! The dynamic type identity of a derived actual. A monomorphic actual's
        ! dynamic type is its declared type, known at compile time. A class
        ! dummy being re-passed already carries a runtime identity, which is
        ! loaded and forwarded so the original value's identity is preserved
        ! across any number of hand-offs.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: dynamic_id
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (context%symbols(symbol_index)%is_polymorphic .and. &
            context%symbols(symbol_index)%has_dynamic_type_address) then
            if (.not. emit_i64_load(context%session, &
                    context%symbols(symbol_index)%dynamic_type_address, &
                    dynamic_id, error_msg)) return
            return
        end if
        dynamic_id = i64_immediate(context%session, &
            int(context%symbols(symbol_index)%derived_type_index, c_int64_t))
    end subroutine class_actual_dynamic_type

    subroutine build_class_scalar_descriptor(context, symbol_index, &
                                             declared_type_index, descriptor, &
                                             error_msg)
        ! Borrowed class descriptor for a whole derived actual named by a symbol.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: declared_type_index
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dynamic_id

        call class_actual_dynamic_type(context, symbol_index, dynamic_id, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        call emit_class_scalar_descriptor(context, &
            context%symbols(symbol_index)%element_address, declared_type_index, &
            dynamic_id, descriptor, error_msg)
    end subroutine build_class_scalar_descriptor

    subroutine emit_class_scalar_descriptor(context, data_address, &
                                            declared_type_index, dynamic_id, &
                                            descriptor, error_msg)
        ! Materialise the borrowed scalar class descriptor a class(t) dummy
        ! receives. The descriptor is a caller-frame temporary that points at
        ! storage the caller owns: ownership is BORROWED, so the callee never
        ! frees the data and the data outlives the call.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: data_address
        integer, intent(in) :: declared_type_index
        type(lr_operand_desc_t), intent(in) :: dynamic_id
        type(lr_operand_desc_t), intent(out) :: descriptor
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: field_addr

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, &
                    int(POLYMORPHIC_DESCRIPTOR_SIZE, c_int64_t)), descriptor, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, c_int64_t), field_addr, &
                error_msg)) return
        if (.not. emit_ptr_store(context%session, data_address, field_addr, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DECLARED_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, &
                i64_immediate(context%session, &
                    int(declared_type_index, c_int64_t)), field_addr, &
                error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i64_store(context%session, dynamic_id, field_addr, &
                                 error_msg)) return
        if (.not. emit_ptr_offset(context%session, descriptor, &
                int(POLYMORPHIC_DESCRIPTOR_OWNERSHIP_OFFSET, c_int64_t), &
                field_addr, error_msg)) return
        if (.not. emit_i32_store(context%session, &
                i32_immediate(context%session, &
                    int(POLYMORPHIC_OWNERSHIP_BORROWED, c_int64_t)), field_addr, &
                error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_class_scalar_descriptor

    subroutine bind_class_dummy_descriptor(context, symbol_index, declared_index, &
                                           error_msg)
        ! Bind a class(t) dummy to the descriptor the caller passed. The
        ! parameter pointer is the descriptor; the value's storage is the data
        ! field, so every component access keeps using element_address and reads
        ! the declared-type prefix of whatever dynamic type arrived.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: declared_index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: data_addr, data_value, dynamic_addr

        if (.not. emit_ptr_offset(context%session, &
                context%symbols(symbol_index)%address, &
                int(POLYMORPHIC_DESCRIPTOR_DATA_OFFSET, c_int64_t), data_addr, &
                error_msg)) return
        if (.not. emit_ptr_load(context%session, data_addr, data_value, &
                                error_msg)) return
        if (.not. emit_ptr_offset(context%session, &
                context%symbols(symbol_index)%address, &
                int(POLYMORPHIC_DESCRIPTOR_DYNAMIC_TYPE_OFFSET, c_int64_t), &
                dynamic_addr, error_msg)) return
        context%symbols(symbol_index)%class_descriptor_address = &
            context%symbols(symbol_index)%address
        context%symbols(symbol_index)%element_address = data_value
        context%symbols(symbol_index)%dynamic_type_address = dynamic_addr
        context%symbols(symbol_index)%has_dynamic_type_address = .true.
        context%symbols(symbol_index)%is_polymorphic = .true.
        context%symbols(symbol_index)%derived_type_index = declared_index
        call set_empty(error_msg)
    end subroutine bind_class_dummy_descriptor

    subroutine class_receiver_argument(context, arena, impl_name, position, &
                                       symbol_index, argument, error_msg)
        ! Argument the receiver of a type-bound call occupies. The passed-object
        ! dummy is conventionally class(t), so it takes the same scalar class
        ! descriptor an explicit class actual takes; a type(t) passed-object
        ! dummy keeps the bare address.
        type(lowering_context_t), intent(inout) :: context
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: impl_name
        integer, intent(in) :: position
        integer, intent(in) :: symbol_index
        type(lr_operand_desc_t), intent(out) :: argument
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: declared_index

        call class_dummy_declared_type_index(context, arena, impl_name, position, &
                                             declared_index)
        if (declared_index <= 0) then
            argument = context%symbols(symbol_index)%element_address
            call set_empty(error_msg)
            return
        end if
        call build_class_scalar_descriptor(context, symbol_index, declared_index, &
                                           argument, error_msg)
    end subroutine class_receiver_argument

    logical function select_type_selector_is_monomorphic(context, sel_index)
        ! True when the selector names a plain declared-type derived object (a
        ! class(t) dummy or local lowered as type(t)): non-allocatable,
        ! non-pointer, with a known derived type. Such an object's dynamic type
        ! is its declared type, so SELECT TYPE resolves statically. An
        ! allocatable / pointer class object may hold an extension at runtime and
        ! needs a real vtable, so it is excluded.
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: sel_index

        select_type_selector_is_monomorphic = .false.
        if (sel_index <= 0) return
        if (context%symbols(sel_index)%value_kind /= VALUE_DERIVED) return
        if (context%symbols(sel_index)%is_allocatable) return
        if (context%symbols(sel_index)%is_pointer) return
        if (context%symbols(sel_index)%derived_type_index <= 0) return
        select_type_selector_is_monomorphic = .true.
    end function select_type_selector_is_monomorphic

    subroutine select_type_guard_type_name(arena, guard_index, guard_kind, &
                                           type_name, error_msg)
        ! Resolve a SELECT TYPE guard's kind ('type_is', 'class_is',
        ! 'class_default') and, for the first two, the derived type name it
        ! names. type_name is empty for class default.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        character(len=:), allocatable, intent(out) :: guard_kind
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: tname_index
        integer, allocatable :: body_tmp(:)
        character(len=:), allocatable :: terr, lit_type

        call set_empty(error_msg)
        guard_kind = ''
        type_name = ''
        call get_type_guard_info(arena, guard_index, guard_kind, tname_index, &
                                 body_tmp)
        if (trim(guard_kind) == 'class_default') return
        if (is_identifier(arena, tname_index)) then
            call get_identifier_name(arena, tname_index, type_name, terr)
            if (len_trim(terr) > 0) error_msg = terr
        else if (is_literal(arena, tname_index)) then
            call get_literal_info(arena, tname_index, type_name, lit_type, terr)
            if (len_trim(terr) > 0) error_msg = terr
        else
            error_msg = 'select type guard type name is not resolvable'
        end if
    end subroutine select_type_guard_type_name

    logical function select_type_discriminates_subtypes(arena, guard_indices, &
                                                         context, declared_type, &
                                                         error_msg)
        ! True when a guard names a derived type different from the selector's
        ! declared type. That is the signature of a SELECT TYPE written to
        ! discriminate runtime subtypes (genuine polymorphism), which the
        ! monomorphic static-dispatch path must not silently miscompile - a
        ! non-allocatable class(t) dummy can still receive a child actual by
        ! reference. Such a construct declines to the vtable-needed diagnostic.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: guard_indices(:)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: declared_type
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_kind, type_name
        integer :: g

        select_type_discriminates_subtypes = .false.
        call set_empty(error_msg)
        if (.not. allocated(guard_indices)) return
        do g = 1, size(guard_indices)
            call select_type_guard_type_name(arena, guard_indices(g), &
                                             guard_kind, type_name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (trim(guard_kind) == 'class_default') cycle
            if (same_name(type_name, declared_type)) cycle
            ! A guard naming another declared type (not the selector's declared
            ! type and not an intrinsic kind) means subtype discrimination.
            if (find_derived_type(context, type_name) > 0) then
                select_type_discriminates_subtypes = .true.
                return
            end if
        end do
    end function select_type_discriminates_subtypes

    subroutine choose_monomorphic_guard(arena, guard_indices, default_idx, &
                                        declared_type, chosen_index, error_msg)
        ! Pick the single live arm for a declared-type selector. A `type is (t)`
        ! guard naming the declared type wins; otherwise a `class is (t)` guard
        ! naming it; otherwise the class default arm (default_idx, may be 0). An
        ! intrinsic-type guard (type is (integer) etc.) never matches a derived
        ! selector and is skipped.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: guard_indices(:)
        integer, intent(in) :: default_idx
        character(len=*), intent(in) :: declared_type
        integer, intent(out) :: chosen_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_kind, type_name
        integer :: g

        chosen_index = 0
        call set_empty(error_msg)
        if (allocated(guard_indices)) then
            do g = 1, size(guard_indices)
                call select_type_guard_type_name(arena, guard_indices(g), &
                                                 guard_kind, type_name, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(guard_kind) == 'type_is' .and. &
                    same_name(type_name, declared_type)) then
                    chosen_index = guard_indices(g)
                    return
                end if
            end do
            do g = 1, size(guard_indices)
                call select_type_guard_type_name(arena, guard_indices(g), &
                                                 guard_kind, type_name, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(guard_kind) == 'class_is' .and. &
                    same_name(type_name, declared_type)) then
                    chosen_index = guard_indices(g)
                    return
                end if
            end do
        end if
        chosen_index = default_idx
    end subroutine choose_monomorphic_guard

    subroutine lower_select_type_derived(arena, node_index, sel_index, &
                                         sel_name, assoc_name, context, error_msg)
        ! SELECT TYPE on a monomorphic declared-type selector. The dynamic type
        ! equals the declared type, so the live arm is resolved at compile time
        ! and lowered inline. The associate name aliases the same derived
        ! storage (identical element_address / derived_type_index), so component
        ! access inside the arm reads and writes the selector's slots.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: sel_index
        character(len=*), intent(in) :: sel_name, assoc_name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: selector_idx, default_idx, chosen_index, idx
        integer, allocatable :: guard_indices(:)
        integer, allocatable :: body(:)
        character(len=:), allocatable :: declared_type
        type(lr_operand_desc_t) :: value
        logical :: terminated

        call set_empty(error_msg)
        call get_select_type_info(arena, node_index, selector_idx, &
                                  guard_indices, default_idx)
        declared_type = trim(context%derived_types( &
            context%symbols(sel_index)%derived_type_index)%name)

        ! A construct that discriminates other subtypes is genuine polymorphism
        ! and needs a runtime type id; decline rather than statically mispick.
        if (select_type_discriminates_subtypes(arena, guard_indices, context, &
                                               declared_type, error_msg)) then
            if (len_trim(error_msg) > 0) return
            call unsupported_feature_error('select type statement', 0, 0, &
                'direct LIRIC session supports monomorphic select type '// &
                '(guards naming only the selector''s declared type); '// &
                'discriminating runtime subtypes needs a vtable', error_msg)
            return
        end if
        if (len_trim(error_msg) > 0) return

        call choose_monomorphic_guard(arena, guard_indices, default_idx, &
                                      declared_type, chosen_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (chosen_index <= 0) return

        ! Bind the associate name to the selector's derived storage. For
        ! `select type (v)` the names coincide and the symbol is already bound.
        if (trim(assoc_name) /= trim(sel_name)) then
            idx = find_symbol_compat(context, assoc_name)
            if (idx <= 0) then
                call grow_symbols(context)
                idx = context%symbol_count + 1
                context%symbol_count = idx
            end if
            context%symbols(idx) = context%symbols(sel_index)
            context%symbols(idx)%name = trim(assoc_name)
        end if

        call select_type_guard_body(arena, chosen_index, body)
        if (.not. allocated(body)) return
        terminated = .false.
        call lower_statement_list(arena, body, context, value, terminated, &
                                  error_msg)
    end subroutine lower_select_type_derived
