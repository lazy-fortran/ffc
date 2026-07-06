    ! Monomorphic-subset polymorphism (#141). A class(t) scalar dummy or local
    ! whose dynamic type is only ever its declared type t lowers exactly like
    ! type(t): the derived-type declaration path already routes both spellings
    ! (extracted_derived_type_name treats type(t) and class(t) alike). This file
    ! adds SELECT TYPE dispatch for such a declared-type selector: the dynamic
    ! type equals the declared type, so exactly one arm is ever live and is
    ! chosen statically. Full dynamic dispatch on an extensible allocatable /
    ! pointer class object (a runtime vtable / type-id) stays out of scope and
    ! declines gracefully.

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
            idx = find_symbol(context, assoc_name)
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
