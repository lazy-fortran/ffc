submodule (session_program_lowering_impl) session_program_lowering_reject_alloc
    use session_program_lowering_reject_alloc_order
    implicit none
contains
    ! Allocation and pointer definition-target validation (#380).
    !
    ! Five constraint checks share one family: an entity that is allocated,
    ! deallocated, pointed at, or passed where the callee may define or
    ! reassociate it must actually carry the attributes that make that legal.
    ! All of them run before lowering, on typed declarations and resolved
    ! callee signatures, so an invalid target never reaches code generation.
    module procedure check_alloc_pointer_targets

        call check_polymorphic_entity_attributes(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_deferred_length_attributes(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_pointer_shape_specs(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_alloc_definition_contexts(arena, error_msg)
        if (len_trim(error_msg) > 0) return
        call check_argument_definition_contexts(arena, error_msg)
    end procedure check_alloc_pointer_targets
    ! F2018 C708: a CLASS entity takes its dynamic type from somewhere else, so
    ! it must be a dummy argument, allocatable or a pointer. This covers both
    ! local/module entities and derived-type components, whose declarations live
    ! in the same arena.
    module procedure check_polymorphic_entity_attributes
        integer :: n
        character(len=:), allocatable :: low, name
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (decl%is_allocatable) cycle
                if (decl%is_pointer) cycle
                if (.not. allocated(decl%type_name)) cycle
                low = trim(lowercase_text(decl%type_name))
                if (len(low) < 7) cycle
                if (low(1:6) /= 'class(') cycle
                if (low == 'class(*)') cycle
                if (decl_names_include_dummy(arena, decl)) cycle
                name = decl_display_name(decl)
                write (location, '(" at line ",I0,", column ",I0)') &
                    decl%line, decl%column
                error_msg = 'CLASS entity '''//name//''' must be dummy, '// &
                    'allocatable or pointer'//trim(location)
                return
            end select
        end do
    end procedure check_polymorphic_entity_attributes
    ! F2018 C723: a deferred character length (len=:) is only meaningful when
    ! the entity can acquire a length, i.e. when it is allocatable or a pointer.
    module procedure check_deferred_length_attributes
        integer :: n
        character(len=:), allocatable :: name
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (decl%is_allocatable) cycle
                if (decl%is_pointer) cycle
                if (.not. decl%has_character_length) cycle
                if (.not. allocated(decl%character_length_expr)) cycle
                if (trim(adjustl(decl%character_length_expr)) /= ':') cycle
                if (decl_names_have_pointer_attr(arena, decl)) cycle
                name = decl_display_name(decl)
                write (location, '(" at line ",I0,", column ",I0)') &
                    decl%line, decl%column
                error_msg = 'entity '''//name//''' with deferred character '// &
                    'length must have the POINTER or ALLOCATABLE attribute'// &
                    trim(location)
                return
            end select
        end do
    end procedure check_deferred_length_attributes
    ! F2018 C832: a POINTER array is deferred-shape; an explicit-shape or
    ! assumed-size spec on a pointer has no valid meaning.
    module procedure check_pointer_shape_specs
        integer :: n, d
        character(len=:), allocatable :: name
        character(len=64) :: location

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. decl%is_pointer) cycle
                if (.not. decl%is_array) cycle
                if (.not. allocated(decl%dimension_indices)) cycle
                do d = 1, size(decl%dimension_indices)
                    if (dim_is_assumed_shape(arena, &
                                             decl%dimension_indices(d))) cycle
                    name = decl_display_name(decl)
                    write (location, '(" at line ",I0,", column ",I0)') &
                        decl%line, decl%column
                    error_msg = 'POINTER array '''//name//''' must have a '// &
                        'deferred shape or assumed rank'//trim(location)
                    return
                end do
            end select
        end do
    end procedure check_pointer_shape_specs
    ! F2018 C932/C866: the allocate-object of ALLOCATE or DEALLOCATE appears in a
    ! variable definition context, so an INTENT(IN) dummy can never be one.
    module procedure check_alloc_definition_contexts
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_alloc_targets(arena, nd%body_indices, error_msg)
            type is (subroutine_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_alloc_targets(arena, nd%body_indices, error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_alloc_definition_contexts
    module procedure check_scope_alloc_targets
        integer :: i, v

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (stmt => arena%entries(body_indices(i))%node)
            type is (allocate_statement_node)
                if (.not. allocated(stmt%var_indices)) cycle
                do v = 1, size(stmt%var_indices)
                    call check_definable_target(arena, body_indices, &
                        stmt%var_indices(v), 'ALLOCATE', error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            type is (deallocate_statement_node)
                if (.not. allocated(stmt%var_indices)) cycle
                do v = 1, size(stmt%var_indices)
                    call check_definable_target(arena, body_indices, &
                        stmt%var_indices(v), 'DEALLOCATE', error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end select
        end do
    end procedure check_scope_alloc_targets
    module procedure check_definable_target
        character(len=:), allocatable :: name
        character(len=64) :: location
        integer :: decl_index, line, column

        call set_empty(error_msg)
        call target_base_name(arena, target_index, name)
        if (len_trim(name) == 0) return
        call scope_decl_for_name(arena, body_indices, name, decl_index)
        if (decl_index <= 0) return
        line = 0
        column = 0
        select type (decl => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (.not. decl%has_intent) return
            if (.not. allocated(decl%intent)) return
            if (trim(lowercase_text(decl%intent)) /= 'in') return
            line = decl%line
            column = decl%column
        class default
            return
        end select
        line = get_node_line(arena, target_index)
        column = get_node_column(arena, target_index)
        write (location, '(" at line ",I0,", column ",I0)') line, column
        error_msg = 'INTENT(IN) dummy argument '''//trim(name)// &
            ''' cannot appear in a variable definition context ('// &
            context_name//' object)'//trim(location)
    end procedure check_definable_target
    ! Actual arguments must carry the attributes the dummy relies on: an
    ! ALLOCATABLE or POINTER dummy can be (re)associated by the callee, and an
    ! INTENT(OUT)/INTENT(INOUT) dummy is defined by it, so the actual must be a
    ! definable object with the matching attribute. In a PURE procedure a
    ! host- or use-associated variable is never definable through a dummy.
    module procedure check_argument_definition_contexts
        integer :: n

        call set_empty(error_msg)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (program_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_call_arguments(arena, nd%body_indices, &
                                                .false., error_msg)
            type is (function_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_call_arguments(arena, nd%body_indices, &
                    prefix_has_pure(nd%prefix_keywords), error_msg)
            type is (subroutine_def_node)
                if (.not. allocated(nd%body_indices)) cycle
                call check_scope_call_arguments(arena, nd%body_indices, &
                    prefix_has_pure(nd%prefix_keywords), error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_argument_definition_contexts
    module procedure prefix_has_pure
        integer :: i

        is_pure = .false.
        if (.not. allocated(prefix_keywords)) return
        do i = 1, size(prefix_keywords)
            if (trim(lowercase_text(prefix_keywords(i))) == 'pure') then
                is_pure = .true.
                return
            end if
        end do
    end procedure prefix_has_pure
    module procedure check_scope_call_arguments
        character(len=:), allocatable :: call_name, sub_err
        integer, allocatable :: arg_indices(:)
        integer :: i

        call set_empty(error_msg)
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            if (.not. is_subroutine_call_statement(arena, body_indices(i))) cycle
            call get_subroutine_call_name(arena, body_indices(i), call_name, &
                                          sub_err)
            if (len_trim(sub_err) > 0) cycle
            if (len_trim(call_name) == 0) cycle
            call get_subroutine_call_arg_indices(arena, body_indices(i), &
                                                 arg_indices, sub_err)
            if (len_trim(sub_err) > 0) cycle
            if (.not. allocated(arg_indices)) cycle
            call check_call_actual_attributes(arena, body_indices, call_name, &
                                              arg_indices, in_pure, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_scope_call_arguments
    module procedure check_call_actual_attributes
        integer, allocatable :: callee_params(:), callee_body(:)
        character(len=:), allocatable :: dummy_name
        logical :: found
        logical :: dummy_alloc, dummy_ptr, dummy_definable, dummy_ptr_intent_in
        character(len=:), allocatable :: dummy_intent
        integer :: a, decl_index

        call set_empty(error_msg)
        call procedure_signature(arena, callee_name, callee_params, &
                                 callee_body, found)
        if (.not. found) return
        do a = 1, min(size(arg_indices), size(callee_params))
            call param_name_at(arena, callee_params(a), dummy_name)
            if (len_trim(dummy_name) == 0) cycle
            call scope_decl_for_name(arena, callee_body, dummy_name, decl_index)
            if (decl_index <= 0) cycle
            dummy_alloc = .false.
            dummy_ptr = .false.
            dummy_definable = .false.
            dummy_ptr_intent_in = .false.
            dummy_intent = ''
            select type (decl => arena%entries(decl_index)%node)
            type is (declaration_node)
                dummy_alloc = decl%is_allocatable
                dummy_ptr = decl%is_pointer
                if (decl%has_intent) then
                    if (allocated(decl%intent)) then
                        dummy_intent = trim(lowercase_text(decl%intent))
                        dummy_definable = dummy_intent == 'out' .or. &
                                          dummy_intent == 'inout'
                    end if
                end if
            class default
                cycle
            end select
            if (dummy_ptr) dummy_ptr_intent_in = dummy_intent == 'in'
            call check_one_actual(arena, body_indices, arg_indices(a), &
                dummy_name, dummy_alloc, dummy_ptr, dummy_definable, &
                dummy_ptr_intent_in, in_pure, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end procedure check_call_actual_attributes
    module procedure check_one_actual
        character(len=:), allocatable :: name
        character(len=64) :: location
        integer :: decl_index
        logical :: actual_alloc, actual_ptr, actual_target, is_literal

        call set_empty(error_msg)
        if (.not. node_exists(arena, actual_index)) return
        write (location, '(" at line ",I0,", column ",I0)') &
            get_node_line(arena, actual_index), &
            get_node_column(arena, actual_index)
        is_literal = .false.
        select type (nd => arena%entries(actual_index)%node)
        type is (literal_node)
            is_literal = .true.
        end select
        if (is_literal) then
            if (dummy_definable .or. dummy_alloc .or. dummy_ptr) then
                error_msg = 'constant actual argument for dummy '''// &
                    trim(dummy_name)//''' cannot appear in a variable '// &
                    'definition context'//trim(location)
            end if
            return
        end if
        call target_base_name(arena, actual_index, name)
        if (len_trim(name) == 0) return
        call scope_decl_for_name(arena, body_indices, name, decl_index)
        if (decl_index <= 0) then
            if (.not. in_pure) return
            if (.not. (dummy_ptr .or. dummy_definable)) return
            call module_decl_for_name(arena, name, decl_index)
            if (decl_index <= 0) return
            error_msg = 'variable '''//trim(name)//''' is not local to this '// &
                'PURE procedure and cannot appear in a variable definition '// &
                'context'//trim(location)
            return
        end if
        actual_alloc = .false.
        actual_ptr = .false.
        actual_target = .false.
        select type (decl => arena%entries(decl_index)%node)
        type is (declaration_node)
            actual_alloc = decl%is_allocatable
            actual_ptr = decl%is_pointer
            actual_target = decl%is_target
        class default
            return
        end select
        if (dummy_alloc) then
            if (.not. actual_alloc) then
                error_msg = 'actual argument for ALLOCATABLE dummy '''// &
                    trim(dummy_name)//''' must be ALLOCATABLE'//trim(location)
                return
            end if
        end if
        if (dummy_ptr) then
            ! F2008 12.5.2.7: a POINTER dummy with INTENT(IN) also accepts a
            ! non-pointer actual that has the TARGET attribute; the dummy is
            ! then pointer-associated with that target.
            if (dummy_ptr_intent_in .and. actual_target) return
            if (.not. actual_ptr) then
                error_msg = 'actual argument for POINTER dummy '''// &
                    trim(dummy_name)//''' must be a pointer'//trim(location)
                return
            end if
        end if
    end procedure check_one_actual
    ! --- shared lookups -------------------------------------------------

    module procedure decl_display_name
        name = ''
        if (allocated(decl%var_name)) then
            name = trim(decl%var_name)
            if (len(name) > 0) return
        end if
        if (allocated(decl%var_names)) then
            if (size(decl%var_names) > 0) name = trim(decl%var_names(1))
        end if
    end procedure decl_display_name
    module procedure decl_names_include_dummy
        integer :: i

        is_dummy = .false.
        if (allocated(decl%var_name)) then
            is_dummy = name_is_dummy_anywhere(arena, trim(decl%var_name))
            if (is_dummy) return
        end if
        if (.not. allocated(decl%var_names)) return
        do i = 1, size(decl%var_names)
            is_dummy = name_is_dummy_anywhere(arena, trim(decl%var_names(i)))
            if (is_dummy) return
        end do
    end procedure decl_names_include_dummy
    ! A POINTER or ALLOCATABLE attribute statement (`pointer good`) is a
    ! separate declaration node for the same name; consult those before
    ! rejecting an entity for missing the attribute.
    module procedure decl_names_have_pointer_attr
        integer :: i

        has_attr = .false.
        if (allocated(decl%var_name)) then
            has_attr = name_has_pointer_attr_stmt(arena, trim(decl%var_name))
            if (has_attr) return
        end if
        if (.not. allocated(decl%var_names)) return
        do i = 1, size(decl%var_names)
            has_attr = name_has_pointer_attr_stmt(arena, &
                                                  trim(decl%var_names(i)))
            if (has_attr) return
        end do
    end procedure decl_names_have_pointer_attr
    module procedure name_has_pointer_attr_stmt
        integer :: n

        has_attr = .false.
        if (len_trim(name) == 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (.not. decl%is_pointer) then
                    if (.not. decl%is_allocatable) cycle
                end if
                if (.not. decl_declares_name(decl, name)) cycle
                has_attr = .true.
                return
            end select
        end do
    end procedure name_has_pointer_attr_stmt
    module procedure decl_declares_name
        integer :: i

        declares = .false.
        if (allocated(decl%var_name)) then
            if (same_name(decl%var_name, name)) then
                declares = .true.
                return
            end if
        end if
        if (.not. allocated(decl%var_names)) return
        do i = 1, size(decl%var_names)
            if (same_name(trim(decl%var_names(i)), name)) then
                declares = .true.
                return
            end if
        end do
    end procedure decl_declares_name
    module procedure name_is_dummy_anywhere
        integer :: n

        is_dummy = .false.
        if (len_trim(name) == 0) return
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (.not. allocated(nd%param_indices)) cycle
                is_dummy = params_contain_name(arena, nd%param_indices, name)
            type is (subroutine_def_node)
                if (.not. allocated(nd%param_indices)) cycle
                is_dummy = params_contain_name(arena, nd%param_indices, name)
            end select
            if (is_dummy) return
        end do
    end procedure name_is_dummy_anywhere
    module procedure params_contain_name
        integer :: p
        character(len=:), allocatable :: pname

        found = .false.
        do p = 1, size(param_indices)
            call param_name_at(arena, param_indices(p), pname)
            if (len_trim(pname) == 0) cycle
            if (same_name(pname, name)) then
                found = .true.
                return
            end if
        end do
    end procedure params_contain_name
    module procedure param_name_at

        name = ''
        if (.not. node_exists(arena, param_index)) return
        select type (pn => arena%entries(param_index)%node)
        type is (parameter_declaration_node)
            if (allocated(pn%name)) name = trim(pn%name)
        type is (declaration_node)
            name = decl_display_name(pn)
        type is (identifier_node)
            if (allocated(pn%name)) name = trim(pn%name)
        end select
    end procedure param_name_at
    module procedure procedure_signature
        integer :: n

        found = .false.
        allocate (param_indices(0))
        allocate (body_indices(0))
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (subroutine_def_node)
                if (.not. allocated(nd%name)) cycle
                if (.not. same_name(nd%name, name)) cycle
                if (allocated(nd%param_indices)) param_indices = nd%param_indices
                if (allocated(nd%body_indices)) body_indices = nd%body_indices
                found = .true.
                return
            end select
        end do
    end procedure procedure_signature
    module procedure scope_decl_for_name
        integer :: i

        decl_index = 0
        if (len_trim(name) == 0) return
        do i = 1, size(indices)
            if (.not. node_exists(arena, indices(i))) cycle
            select type (decl => arena%entries(indices(i))%node)
            type is (declaration_node)
                if (.not. decl_declares_name(decl, name)) cycle
                decl_index = indices(i)
                return
            end select
        end do
    end procedure scope_decl_for_name
    module procedure module_decl_for_name
        integer :: n

        decl_index = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (module_node)
                if (.not. allocated(nd%declaration_indices)) cycle
                call scope_decl_for_name(arena, nd%declaration_indices, name, &
                                         decl_index)
            end select
            if (decl_index > 0) return
        end do
    end procedure module_decl_for_name
    module procedure target_base_name
        ! Name of the variable a definition context ultimately reaches. A
        ! component or element chain (x%c, arr(i)%c%e(j)) defines the variable
        ! at the root of the chain, so walk down to it; a plain subscript or
        ! identifier is that root already.
        integer :: current

        root_name = ''
        current = node_index
        do while (node_exists(arena, current))
            select type (nd => arena%entries(current)%node)
            type is (identifier_node)
                if (allocated(nd%name)) root_name = trim(nd%name)
                return
            type is (call_or_subscript_node)
                if (nd%base_expr_index > 0) then
                    current = nd%base_expr_index
                else
                    if (allocated(nd%name)) root_name = trim(nd%name)
                    return
                end if
            type is (component_access_node)
                current = nd%base_expr_index
            class default
                return
            end select
        end do
    end procedure target_base_name
end submodule session_program_lowering_reject_alloc
