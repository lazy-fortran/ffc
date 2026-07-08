   subroutine collect_inferred_symbols(arena, root_index, context, error_msg)
        ! Walk the AST to find identifiers with FortFront-inferred types that have
        ! no explicit declaration. Register them as symbols so later lowering
        ! (assignment targets, DO indices, READ targets, array identifiers)
        ! resolves them instead of hard-erroring with "was not declared".
        ! Explicit declarations remain authoritative: a seeded symbol is never
        ! overwritten if an explicit declaration for the same name exists.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        character(len=:), allocatable :: id_name, err
        integer :: sym_idx

        call set_empty(error_msg)
        ! Pass 1: collect all explicitly declared variable names from the body.
        ! These names are authoritative; we never seed an inferred symbol for them.
        call collect_explicit_decl_names(arena, root_index, context, error_msg)
        if (len_trim(error_msg) > 0) return
        ! Also mark USE-imported module variable names as explicit so inferred
        ! seeds do not shadow them.
        call collect_module_export_names(context, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Pass 2: walk the arena for identifier nodes with inferred types.
        ! Register a symbol for each unique name that is not explicitly declared.
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            if (.not. is_identifier(arena, i)) cycle
            call get_identifier_name(arena, i, id_name, err)
            if (len_trim(err) > 0) cycle
            if (len_trim(id_name) == 0) cycle
            ! Skip if already explicitly declared
            if (name_is_explicitly_declared(context, id_name)) cycle
            ! Skip if already registered (from a prior inferred seed)
            sym_idx = find_symbol(context, id_name)
            if (sym_idx > 0) cycle
            ! Try to register from inferred type
            call try_seed_inferred_symbol(arena, i, id_name, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine collect_inferred_symbols

    subroutine collect_explicit_decl_names(arena, root_index, context, error_msg)
        ! Mark all variable names that have an explicit declaration_node in the
        ! program body. These names are authoritative and must not be overwritten
        ! by inferred symbols.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: body_indices(:)
        character(len=:), allocatable :: node_type, ff_err, dummy_name
        integer :: i

        call set_empty(error_msg)
        node_type = get_node_type_at(arena, root_index)
        if (node_type == 'program_node' .or. node_type == 'program') then
            call get_program_body_info(arena, root_index, dummy_name, &
                body_indices, ff_err)
            if (len_trim(ff_err) > 0 .or. .not. allocated(body_indices)) return
            do i = 1, size(body_indices)
                call mark_explicit_declarations(arena, body_indices(i), context)
            end do
        else if (node_type == 'multi_unit_container' .or. &
                 node_type == 'mixed_construct_container') then
            call get_unit_body_indices(arena, root_index, body_indices, ff_err)
            if (len_trim(ff_err) > 0 .or. .not. allocated(body_indices)) return
            do i = 1, size(body_indices)
                call mark_explicit_declarations(arena, body_indices(i), context)
            end do
        end if
    end subroutine collect_explicit_decl_names

    subroutine collect_module_export_names(context, error_msg)
        ! Mark USE-imported module variable names as explicitly declared so
        ! inferred seeds do not shadow them.
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64), allocatable :: names(:)
        integer :: m, vi, ni

        call set_empty(error_msg)
        do m = 1, context%module_export_count
            do vi = 1, context%module_exports(m)%variable_count
                call module_declaration_var_names(context%arena, &
                    context%module_exports(m)%variable_indices(vi), names)
                if (allocated(names)) then
                    do ni = 1, size(names)
                        if (len_trim(names(ni)) > 0) then
                            call add_explicit_decl_name(context, names(ni))
                        end if
                    end do
                end if
            end do
        end do
    end subroutine collect_module_export_names

    recursive subroutine mark_explicit_declarations(arena, node_index, context)
        ! Recursively walk a body, marking all variable names from declaration
        ! nodes as explicitly declared. Also descends into contained procedures,
        ! DO loops, IF blocks, and other compound nodes.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable :: var_name, err
        integer :: i, j

        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (declaration_node)
            if (allocated(node%var_name) .and. len_trim(node%var_name) > 0) then
                call add_explicit_decl_name(context, node%var_name)
            end if
            if (allocated(node%var_names)) then
                do i = 1, size(node%var_names)
                    if (len_trim(node%var_names(i)) > 0) then
                        call add_explicit_decl_name(context, node%var_names(i))
                    end if
                end do
            end if
        type is (do_loop_node)
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call mark_explicit_declarations(arena, node%body_indices(i), &
                        context)
                end do
            end if
        type is (if_node)
            if (allocated(node%then_body_indices)) then
                do i = 1, size(node%then_body_indices)
                    call mark_explicit_declarations(arena, node%then_body_indices(i), &
                        context)
                end do
            end if
            if (allocated(node%else_body_indices)) then
                do i = 1, size(node%else_body_indices)
                    call mark_explicit_declarations(arena, node%else_body_indices(i), &
                        context)
                end do
            end if
        type is (function_def_node)
            if (allocated(node%name) .and. len_trim(node%name) > 0) then
                call add_explicit_decl_name(context, node%name)
            end if
            if (allocated(node%result_variable) .and. &
                len_trim(node%result_variable) > 0) then
                call add_explicit_decl_name(context, node%result_variable)
            end if
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call mark_explicit_declarations(arena, node%body_indices(i), &
                        context)
                end do
            end if
        type is (subroutine_def_node)
            if (allocated(node%name) .and. len_trim(node%name) > 0) then
                call add_explicit_decl_name(context, node%name)
            end if
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call mark_explicit_declarations(arena, node%body_indices(i), &
                        context)
                end do
            end if
        type is (program_node)
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call mark_explicit_declarations(arena, node%body_indices(i), &
                        context)
                end do
            end if
        type is (select_case_node)
            if (allocated(node%case_indices)) then
                do i = 1, size(node%case_indices)
                    if (node_exists(arena, node%case_indices(i))) then
                        call mark_explicit_declarations(arena, &
                            node%case_indices(i), context)
                    end if
                end do
            end if
            if (node_exists(arena, node%default_index)) then
                call mark_explicit_declarations(arena, node%default_index, &
                    context)
            end if
        type is (where_node)
            if (allocated(node%where_body_indices)) then
                do i = 1, size(node%where_body_indices)
                    call mark_explicit_declarations(arena, &
                        node%where_body_indices(i), context)
                end do
            end if
            if (allocated(node%elsewhere_clauses)) then
                do i = 1, size(node%elsewhere_clauses)
                    if (allocated(node%elsewhere_clauses(i)%body_indices)) then
                        do j = 1, size(node%elsewhere_clauses(i)%body_indices)
                            call mark_explicit_declarations(arena, &
                                node%elsewhere_clauses(i)%body_indices(j), &
                                context)
                        end do
                    end if
                end do
            end if
        type is (forall_node)
            if (allocated(node%body_indices)) then
                do i = 1, size(node%body_indices)
                    call mark_explicit_declarations(arena, node%body_indices(i), &
                        context)
                end do
            end if
        end select
    end subroutine mark_explicit_declarations

    logical function name_is_explicitly_declared(context, name) result(found)
        ! Check if a name appears in the explicitly declared names list.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        found = .false.
        do i = 1, context%explicit_decl_count
            if (same_name(context%explicit_decl_names(i), name)) then
                found = .true.
                return
            end if
        end do
    end function name_is_explicitly_declared

    subroutine add_explicit_decl_name(context, name)
        ! Add a name to the explicitly declared names list (deduplicated).
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer :: new_size
        character(len=64), allocatable :: tmp(:)

        if (name_is_explicitly_declared(context, name)) return
        context%explicit_decl_count = context%explicit_decl_count + 1
        if (.not. allocated(context%explicit_decl_names)) then
            allocate(context%explicit_decl_names(32))
        else if (context%explicit_decl_count > size(context%explicit_decl_names)) then
            new_size = max(2 * size(context%explicit_decl_names), 64)
            allocate(tmp(new_size))
            tmp(1:context%explicit_decl_count - 1) = &
                context%explicit_decl_names(1:context%explicit_decl_count - 1)
            call move_alloc(tmp, context%explicit_decl_names)
        end if
        context%explicit_decl_names(context%explicit_decl_count) = trim(name)
    end subroutine add_explicit_decl_name

    subroutine try_seed_inferred_symbol(arena, node_index, name, context, error_msg)
        ! Attempt to register a symbol for an identifier with a FortFront-inferred
        ! type. Returns without action if the type is not a supported scalar kind
        ! or if the name is already registered.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(mono_type_t), allocatable :: inferred
        logical :: found
        integer :: value_kind

        call set_empty(error_msg)
        call get_type_for_node(arena, node_index, inferred, found)
        if (.not. found) return
        if (.not. allocated(inferred)) return
        if (inferred%kind <= 0) return
        value_kind = inferred_type_to_value_kind(inferred, context)
        if (value_kind <= 0) return
        call define_symbol(context, name, value_kind, error_msg)
    end subroutine try_seed_inferred_symbol

    integer function inferred_type_to_value_kind(inferred, context) result(vk)
        ! Map a FortFront mono_type_t kind to the ffc value_kind constant.
        ! Returns 0 for unsupported types (derived, character array, etc.).
        type(mono_type_t), intent(in) :: inferred
        type(lowering_context_t), intent(in), optional :: context
        type(mono_type_t) :: elem_type
        integer :: elem_kind

        vk = 0
        select case (inferred%kind)
        case (TINT)
            vk = VALUE_I32
        case (TREAL)
            vk = VALUE_F32
        case (TDOUBLE)
            vk = VALUE_F64
        case (TLOGICAL)
            vk = VALUE_LOGICAL
        case (TCOMPLEX)
            vk = VALUE_C4
        case (TCHAR)
            ! Character scalars from inference: supported as fixed-length.
            vk = VALUE_CHARACTER
        case (TARRAY)
            ! Array: extract element kind from the first type argument.
            if (inferred%has_args() .and. inferred%get_args_count() > 0) then
                elem_type = inferred%get_arg(1)
                elem_kind = elem_type%kind
                select case (elem_kind)
                case (TINT)
                    vk = VALUE_I32
                case (TREAL)
                    vk = VALUE_F32
                case (TDOUBLE)
                    vk = VALUE_F64
                case (TLOGICAL)
                    vk = VALUE_LOGICAL
                case (TCOMPLEX)
                    vk = VALUE_C4
                case default
                    vk = 0
                end select
            end if
        case default
            vk = 0
        end select
    end function inferred_type_to_value_kind
