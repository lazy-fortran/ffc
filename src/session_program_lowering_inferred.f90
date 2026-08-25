submodule (session_program_lowering_impl) session_program_lowering_inferred
    implicit none
contains
    module procedure collect_inferred_symbols
    ! Walk the AST to find identifiers with FortFront-inferred types that have
    ! no explicit declaration. Register them as symbols so later lowering
    ! (assignment targets, DO indices, READ targets, array identifiers)
    ! resolves them instead of hard-erroring with "was not declared".
    ! Explicit declarations remain authoritative: a seeded symbol is never
    ! overwritten if an explicit declaration for the same name exists.
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
        ! FortFront's binding at this reference is authoritative whenever it
        ! is available.
        if (identifier_has_explicit_binding(arena, i, id_name)) cycle
        ! Some reference nodes do not carry a FortFront binding even though
        ! the name is declared in the source.  Do not let that incomplete
        ! metadata seed a provisional scalar before the real declaration is
        ! lowered; otherwise ALLOCATABLE and array declarations look like
        ! duplicate declarations.  The explicit-name list is only a fallback
        ! for references lacking the authoritative binding above.
        if (name_is_explicitly_declared(context, id_name)) cycle
        ! Skip if already registered (from a prior inferred seed)
        sym_idx = find_symbol_compat(context, id_name)
        if (sym_idx > 0) cycle
        ! Try to register from inferred type
        call try_seed_inferred_symbol(arena, i, id_name, context, error_msg)
        if (len_trim(error_msg) > 0) return
    end do
    end procedure collect_inferred_symbols

    logical function identifier_has_explicit_binding(arena, node_index, name) &
        result(found)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name
        type(declaration_binding_t) :: binding
        character(len=:), allocatable :: resolve_error

        found = .false.
        call resolve_name_at_node(arena, node_index, name, binding, resolve_error)
        if (len_trim(resolve_error) > 0 .or. .not. binding%found) return
        if (binding%binding_kind == BINDING_NAMED_CONSTANT) then
            found = .true.
            return
        end if
        if (binding%binding_kind /= BINDING_DECLARATION .and. &
            binding%binding_kind /= BINDING_DUMMY_ARGUMENT .and. &
            binding%binding_kind /= BINDING_FUNCTION_RESULT) return
        if (.not. node_exists(arena, binding%declaration_node_index)) return
        select type (decl => arena%entries(binding%declaration_node_index)%node)
        type is (declaration_node)
            found = .not. decl%is_inferred
        class default
            found = .true.
        end select
    end function identifier_has_explicit_binding

    module procedure collect_explicit_decl_names
    ! Mark all variable names that have an explicit declaration_node in the
    ! program body. These names are authoritative and must not be overwritten
    ! by inferred symbols.
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
    end procedure collect_explicit_decl_names

    module procedure collect_module_export_names
    ! Mark USE-imported module variable names as explicitly declared so
    ! inferred seeds do not shadow them.
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
    end procedure collect_module_export_names

    module procedure mark_explicit_declarations
    ! Recursively walk a body, marking all variable names from declaration
    ! nodes as explicitly declared. Also descends into contained procedures,
    ! DO loops, IF blocks, and other compound nodes.
    character(len=:), allocatable :: var_name, err
    integer :: i, j

    if (.not. node_exists(arena, node_index)) return
    select type (node => arena%entries(node_index)%node)
        type is (declaration_node)
        ! FortFront inserts `is_inferred` declarations during lazy
        ! standardization. They are metadata for the type-inference pass, not
        ! user declarations and have no collected binding/storage record in
        ! ffc. Treating them as explicit here suppresses the inferred-symbol
        ! seed that executable references need (#2848).
        if (.not. node%is_inferred .and. .not. declaration_is_bare_dimension(node)) then
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
        type is (block_construct_node)
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                call mark_explicit_declarations(arena, node%body_indices(i), &
                    context)
            end do
        end if
        type is (associate_node)
        if (allocated(node%associations)) then
            do i = 1, size(node%associations)
                if (allocated(node%associations(i)%name)) then
                    call add_explicit_decl_name(context, &
                        node%associations(i)%name)
                end if
            end do
        end if
        if (allocated(node%body_indices)) then
            do i = 1, size(node%body_indices)
                call mark_explicit_declarations(arena, node%body_indices(i), &
                    context)
            end do
        end if
    end select
    end procedure mark_explicit_declarations

    module procedure name_is_explicitly_declared
    ! Check if a name appears in the explicitly declared names list.
    integer :: i

    found = .false.
    do i = 1, context%explicit_decl_count
        if (same_name(context%explicit_decl_names(i), name)) then
            found = .true.
            return
        end if
    end do
    end procedure name_is_explicitly_declared

    module procedure add_explicit_decl_name
    ! Add a name to the explicitly declared names list (deduplicated).
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
    end procedure add_explicit_decl_name

    module procedure try_seed_inferred_symbol
    ! Attempt to register a symbol for an identifier with a FortFront-inferred
    ! type. Returns without action if the type is not a supported scalar kind
    ! or if the name is already registered.
    type(mono_type_t), allocatable :: inferred
    logical :: found
    integer :: value_kind

    call set_empty(error_msg)
    call get_type_for_node(arena, node_index, inferred, found)
    if (.not. found) return
    if (.not. allocated(inferred)) return
    if (inferred%kind <= 0) return
    value_kind = inferred_type_to_value_kind(inferred, context)
    ! A bare DIMENSION's shape is materialized when its first executable
    ! reference is lowered (after the main function is active). Defining its
    ! alloca during this metadata prepass would fail with "no active block".
    if (inferred%kind == TARRAY .and. has_bare_dimension(arena, name)) return
    if (value_kind <= 0) return
    call define_symbol(context, name, value_kind, error_msg)
    end procedure try_seed_inferred_symbol

    logical function has_bare_dimension(arena, name) result(found)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: i

        found = .false.
        do i = 1, arena%size
            if (.not. node_exists(arena, i)) cycle
            select type (node => arena%entries(i)%node)
                type is (declaration_node)
                if (declaration_is_bare_dimension(node) .and. &
                    allocated(node%var_name) .and. &
                    same_name(node%var_name, name)) then
                    found = .true.
                    return
                end if
            end select
        end do
    end function has_bare_dimension

    module procedure inferred_type_to_value_kind
    ! Map a FortFront mono_type_t kind to the ffc value_kind constant.
    ! Returns 0 for unsupported types (derived, character array, etc.).
    type(mono_type_t) :: elem_type
    integer :: elem_kind

    vk = 0
    select case (inferred%kind)
    case (TINT)
        vk = VALUE_I32
    case (TREAL)
        vk = VALUE_F32
        ! Lazy Fortran default real is 8 bytes (#438). An inferred real
        ! must seed the same storage the explicit kind-less declaration
        ! records, otherwise the two collide when both are seen (#571).
        if (lazy_defaults_active(context)) vk = VALUE_F64
    case (TDOUBLE)
        vk = VALUE_F64
    case (TLOGICAL)
        vk = VALUE_LOGICAL
    case (TCOMPLEX)
        ! Complex storage must be created after the procedure's LIRIC block
        ! is active; leave FortFront's inferred declaration for normal
        ! lowering instead of allocating it in this metadata prepass.
        vk = 0
    case (TCHAR)
        ! The declaration carries the literal length and is lowered after the
        ! procedure block starts.  Seeding it here makes that declaration
        ! appear duplicate and loses its length.
        vk = 0
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
                if (lazy_defaults_active(context)) vk = VALUE_F64
            case (TDOUBLE)
                vk = VALUE_F64
            case (TLOGICAL)
                vk = VALUE_LOGICAL
            case (TCOMPLEX)
                vk = 0
            case default
                vk = 0
            end select
        end if
    case default
        vk = 0
    end select
    end procedure inferred_type_to_value_kind

end submodule session_program_lowering_inferred
