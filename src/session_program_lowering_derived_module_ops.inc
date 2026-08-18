    subroutine component_base_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'component base index does not reference an AST node'
            call set_empty(name)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            return
        end if
        error_msg = 'nested component access is not supported'
        call set_empty(name)
    end subroutine component_base_name

    subroutine lower_derived_whole_assignment_diagnostic(arena, node, context, &
                                                         symbol_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: symbol_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index

        type_index = context%symbols(symbol_index)%derived_type_index
        if (is_derived_constructor_call(arena, node%value_index, context, &
                                        type_index)) then
            call unsupported_feature_error('derived type constructor', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived type constructors', &
                                           error_msg)
            return
        end if
        call unsupported_feature_error('derived type assignment', &
                                       node%line, node%column, &
                                       'direct LIRIC session only supports '// &
                                       'component assignment', error_msg)
    end subroutine lower_derived_whole_assignment_diagnostic

    logical function is_derived_constructor_call(arena, node_index, context, &
                                                 type_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index

        is_derived_constructor_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%name)) then
                is_derived_constructor_call = &
                    same_name(node%name, context%derived_types(type_index)%name)
            end if
        end select
    end function is_derived_constructor_call

    integer function declaration_derived_type_index(context, node) result(type_index)
        type(lowering_context_t), intent(in) :: context
        type(declaration_node), intent(in) :: node
        character(len=:), allocatable :: type_name

        type_index = 0
        if (.not. allocated(node%type_name)) return
        call extracted_derived_type_name(node%type_name, type_name)
        if (len_trim(type_name) == 0) return
        type_index = find_derived_type(context, type_name)
        if (type_index > 0) return
        ! A parameterized type resolves to the concrete layout instantiated
        ! for its actual type parameters (#411).
        block
            character(len=:), allocatable :: instance_name, pdt_error
            logical :: is_pdt

            call pdt_declaration_instance(context%arena, context, node, &
                                          instance_name, is_pdt, pdt_error)
            if (is_pdt .and. len_trim(pdt_error) == 0) &
                type_index = find_derived_type(context, instance_name)
        end block
    end function declaration_derived_type_index

    subroutine extracted_derived_type_name(type_spec, type_name)
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable :: lowered
        integer :: close_pos
        integer :: open_pos

        lowered = trim(lowercase_text(type_spec))
        ! type(t) and class(t) both name derived type t. class(*) is unlimited
        ! polymorphic and stays non-derived (it routes to VALUE_CLASS_STAR). For
        ! a non-polymorphic declared-type object class(t) lowers exactly like
        ! type(t): the passed-object dummy is taken by reference (static
        ! dispatch), so both spellings share the derived-type path here.
        if (len_trim(lowered) >= 5) then
            if (lowered(1:5) == 'type(') then
                open_pos = index(lowered, '(')
                close_pos = index(lowered, ')', back=.true.)
                if (close_pos > open_pos + 1) then
                    type_name = lowered(open_pos + 1:close_pos - 1)
                else
                    type_name = ''
                end if
                if (type_name == '*') type_name = ''
                return
            end if
        end if
        if (len_trim(lowered) >= 6) then
            if (lowered(1:6) == 'class(') then
                open_pos = index(lowered, '(')
                close_pos = index(lowered, ')', back=.true.)
                if (close_pos > open_pos + 1) then
                    type_name = lowered(open_pos + 1:close_pos - 1)
                else
                    type_name = ''
                end if
                if (type_name == '*') type_name = ''
                return
            end if
        end if

        if (lowered /= 'integer' .and. lowered /= 'real' .and. &
            lowered /= 'logical' .and. .not. is_character_type_name(lowered)) then
            type_name = lowered
        else
            type_name = ''
        end if
    end subroutine extracted_derived_type_name

    logical function declaration_is_class_polymorphic(node)
        ! True when a declaration is spelled class(t) for a derived type t.
        type(declaration_node), intent(in) :: node

        declaration_is_class_polymorphic = .false.
        if (.not. allocated(node%type_name)) return
        ! A plain class(t) scalar takes the class descriptor here. A class
        ! allocatable also carries one, but it is bound at its own declaration
        ! path (#421) because its storage is deferred until ALLOCATE; a class
        ! pointer is deliberately kept out of this descriptor path.
        if (node%is_allocatable) return
        if (node%is_pointer) return
        declaration_is_class_polymorphic = is_class_derived_type_spec(node%type_name)
    end function declaration_is_class_polymorphic

    logical function declaration_is_class_pointer(node)
        ! True only for a pointer declared as class(t). This records the
        ! unresolved dispatch case without routing the pointer through the
        ! scalar class descriptor lowering.
        type(declaration_node), intent(in) :: node

        declaration_is_class_pointer = .false.
        if (.not. node%is_pointer) return
        if (.not. allocated(node%type_name)) return
        declaration_is_class_pointer = is_class_derived_type_spec(node%type_name)
    end function declaration_is_class_pointer

    logical function is_class_derived_type_spec(type_spec)
        ! True for class(t) naming a derived type t, i.e. a polymorphic scalar
        ! whose declared type is t. type(t) is monomorphic and class(*) is
        ! unlimited polymorphic; neither matches here.
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable :: lowered

        is_class_derived_type_spec = .false.
        lowered = trim(lowercase_text(type_spec))
        if (len_trim(lowered) < 6) return
        if (lowered(1:6) /= 'class(') return
        is_class_derived_type_spec = is_derived_type_spec(type_spec)
    end function is_class_derived_type_spec

    logical function is_derived_type_spec(type_spec)
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable :: type_name

        call extracted_derived_type_name(type_spec, type_name)
        is_derived_type_spec = len_trim(type_name) > 0
    end function is_derived_type_spec

    integer function find_derived_type(context, name) result(type_index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        type_index = 0
        do i = 1, context%derived_type_count
            if (same_name(context%derived_types(i)%name, name)) then
                type_index = i
                return
            end if
        end do
        do i = 1, context%derived_type_alias_count
            if (same_name(context%derived_type_alias_names(i), name)) then
                type_index = context%derived_type_alias_indices(i)
                return
            end if
        end do
    end function find_derived_type

    integer function find_canonical_derived_type(context, name, identity) &
            result(type_index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=*), intent(in), optional :: identity
        integer :: i

        type_index = 0
        do i = 1, context%derived_type_count
            if (.not. same_name(context%derived_types(i)%name, name)) cycle
            if (present(identity)) then
                if (len_trim(identity) > 0) then
                    if (len_trim(context%derived_types(i)%canonical_identity) == 0) cycle
                    if (.not. same_name(context%derived_types(i)%canonical_identity, &
                                        identity)) cycle
                end if
            end if
            type_index = i
            return
        end do
    end function find_canonical_derived_type

    integer function find_derived_type_identity(context, identity) result(type_index)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: identity
        integer :: i

        type_index = 0
        if (len_trim(identity) == 0) return
        do i = 1, context%derived_type_count
            if (len_trim(context%derived_types(i)%canonical_identity) == 0) cycle
            if (same_name(context%derived_types(i)%canonical_identity, identity)) then
                type_index = i
                return
            end if
        end do
    end function find_derived_type_identity

    subroutine register_derived_type_alias(context, alias_name, type_index, &
                                           error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: alias_name
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=64), allocatable :: old_names(:), new_names(:)
        integer, allocatable :: old_indices(:), new_indices(:)
        integer :: i, n

        call set_empty(error_msg)
        if (len_trim(alias_name) == 0) return
        if (type_index <= 0 .or. type_index > context%derived_type_count) then
            error_msg = 'cannot alias an invalid derived type index'
            return
        end if
        if (same_name(alias_name, context%derived_types(type_index)%name)) return
        do i = 1, context%derived_type_alias_count
            if (.not. same_name(context%derived_type_alias_names(i), alias_name)) cycle
            if (context%derived_type_alias_indices(i) /= type_index) then
                error_msg = 'derived type alias "'//trim(alias_name)// &
                    '" refers to conflicting types'
            end if
            return
        end do
        if (find_canonical_derived_type(context, alias_name) > 0) then
            error_msg = 'derived type alias "'//trim(alias_name)// &
                '" conflicts with a canonical type name'
            return
        end if
        n = context%derived_type_alias_count + 1
        allocate(new_names(n), new_indices(n))
        if (context%derived_type_alias_count > 0) then
            old_names = context%derived_type_alias_names
            old_indices = context%derived_type_alias_indices
            new_names(1:n - 1) = old_names
            new_indices(1:n - 1) = old_indices
        end if
        new_names(n) = trim(alias_name)
        new_indices(n) = type_index
        call move_alloc(new_names, context%derived_type_alias_names)
        call move_alloc(new_indices, context%derived_type_alias_indices)
        context%derived_type_alias_count = n
    end subroutine register_derived_type_alias

    integer function find_derived_component(context, type_index, name) &
        result(component_index)
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: name
        integer :: i

        component_index = 0
        if (type_index <= 0 .or. type_index > context%derived_type_count) return
        do i = 1, context%derived_types(type_index)%component_count
            if (same_name(context%derived_types(type_index)%component_names(i), &
                          name)) then
                component_index = i
                return
            end if
        end do
    end function find_derived_component

    subroutine import_module_derived_types(arena, use_node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: export_idx
        integer :: i
        integer :: type_idx
        integer :: type_idx_in_context
        character(len=:), allocatable :: module_lowered
        character(len=:), allocatable :: type_name

        call set_empty(error_msg)

        if (.not. allocated(use_node%module_name)) return

        module_lowered = lowercase_text(use_node%module_name)
        export_idx = find_module_export_index(context, module_lowered)
        if (export_idx <= 0) then
            ! A module defined in this unit but exporting nothing collectable
            ! (empty or procedure-only) is a no-op. Otherwise resolve through a
            ! .fmod on the include path (#189); intrinsic modules are a no-op.
            if (module_defined_in_unit(arena, module_lowered)) return
            call import_module_from_fmod(use_node, context, error_msg)
            return
        end if

        do i = 1, context%module_exports(export_idx)%derived_type_count
            type_idx = context%module_exports(export_idx)% &
                derived_type_indices(i)
            if (.not. node_exists(arena, type_idx)) then
                error_msg = 'invalid derived type arena index in module '// &
                    trim(context%module_exports(export_idx)%module_name)
                return
            end if
            if (.not. is_derived_type_node(arena, type_idx)) then
                cycle
            end if
            call get_derived_type_name(arena, type_idx, type_name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (len_trim(type_name) == 0) cycle
            if (.not. use_only_wants(use_node, trim(type_name))) cycle
            ! A parameterized type has no layout of its own (#411): its
            ! instances are registered per actual-parameter tuple instead.
            if (find_pdt_template(context, type_name) > 0) cycle
            type_idx_in_context = find_derived_type(context, type_name)
            if (type_idx_in_context > 0) then
                cycle
            end if
            call add_derived_type_from_arena(arena, type_idx, context, &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end do

        call import_module_constants(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call import_module_variables(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call validate_use_only_names(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call import_use_renames(arena, export_idx, use_node, context, error_msg)
    end subroutine import_module_derived_types

    recursive subroutine import_module_from_fmod(use_node, context, error_msg)
        ! Resolve `use <module>` against a .fmod on the include path. Intrinsic
        ! modules are a no-op; an unresolved non-intrinsic module is an error.
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(module_info_t) :: info
        character(len=:), allocatable :: name, path, targets, target
        integer :: i, j, b, sep
        type(use_statement_node) :: dependency_use

        call set_empty(error_msg)
        if (.not. allocated(use_node%module_name)) return
        name = trim(use_node%module_name)
        if (is_intrinsic_module(name)) return

        path = find_fmod_on_path(context, name)
        if (len_trim(path) == 0) then
            call unsupported_feature_error('use statement', use_node%line, &
                use_node%column, 'module "'//name//'" was not compiled in '// &
                'this unit and no '//name//'.fmod was found on the include '// &
                'path', error_msg)
            return
        end if

        call read_fmod(path, info, error_msg)
        if (len_trim(error_msg) > 0) return
        call validate_fmod_use_names(use_node, info, error_msg)
        if (len_trim(error_msg) > 0) return

        ! A public derived type may contain a type imported from another
        ! module. Reconstruct those transitive layouts before the public
        ! module's own layouts (#328/#447).
        if (allocated(info%uses)) then
            do i = 1, size(info%uses)
                if (.not. allocated(info%uses(i)%name)) cycle
                if (len_trim(info%uses(i)%name) == 0) cycle
                dependency_use%module_name = trim(info%uses(i)%name)
                call import_module_from_fmod(dependency_use, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if

        ! Derived types come first: a later component or variable may name one,
        ! and a nested component must find its inner type already registered.
        call import_fmod_derived_types(use_node, info, context, error_msg)
        if (len_trim(error_msg) > 0) return
        ! A type-bound target is an implementation detail of the exported
        ! type, not a name users need to repeat in USE ONLY. Import its
        ! callable procedure explicitly so imported dispatch can lower to the
        ! defining module's symbol.
        if (allocated(info%derived_types) .and. allocated(info%procedures)) then
            do i = 1, size(info%derived_types)
                if (.not. allocated(info%derived_types(i)%bindings)) cycle
                do b = 1, size(info%derived_types(i)%bindings)
                    targets = ''
                    if (allocated(info%derived_types(i)%bindings(b)%specific_names)) &
                        targets = trim(info%derived_types(i)%bindings(b)%specific_names)
                    if (len_trim(targets) == 0) targets = trim( &
                        info%derived_types(i)%bindings(b)%target_name)
                    do while (len_trim(targets) > 0)
                        sep = index(targets, ' ')
                        if (sep == 0) then
                            target = trim(targets)
                            targets = ''
                        else
                            target = trim(targets(1:sep - 1))
                            targets = adjustl(targets(sep + 1:))
                        end if
                        if (len_trim(target) == 0) cycle
                        do j = 1, size(info%procedures)
                            if (.not. allocated(info%procedures(j)%name)) cycle
                            if (.not. same_name(info%procedures(j)%name, target)) cycle
                            call import_fmod_procedure(use_node, name, &
                                info%procedures(j), context, error_msg, force=.true., &
                                require_class_identities= &
                                    info%fmod_schema == FMOD_SCHEMA_VERSION)
                            if (len_trim(error_msg) > 0) return
                            exit
                        end do
                    end do
                end do
            end do
        end if
        if (allocated(info%parameters)) then
            do i = 1, size(info%parameters)
                call import_fmod_parameter(use_node, info%parameters(i), &
                                           context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        if (allocated(info%variables)) then
            do i = 1, size(info%variables)
                call import_fmod_variable(use_node, info%variables(i), &
                                          context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        if (allocated(info%procedures)) then
            do i = 1, size(info%procedures)
                call import_fmod_procedure(use_node, name, info%procedures(i), &
                                           context, error_msg, &
                                           require_class_identities= &
                                               info%fmod_schema == FMOD_SCHEMA_VERSION)
                if (len_trim(error_msg) > 0) return
            end do
        end if
        if (allocated(info%generics)) then
            do i = 1, size(info%generics)
                call import_fmod_generic(use_node, name, info%generics(i), info, &
                                         context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
    end subroutine import_module_from_fmod

    subroutine import_fmod_derived_types(use_node, info, context, error_msg)
        ! Register every derived type the artefact describes, in the order it
        ! records them, so a nested component's inner type is already present
        ! when the outer type refers to it. The layout comes entirely from the
        ! artefact: component kinds, slot spans, character lengths, and fixed
        ! array shapes are read back, never re-derived from the module source
        ! (#414).
        type(use_statement_node), intent(in) :: use_node
        type(module_info_t), intent(in) :: info
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        character(len=:), allocatable :: local_name
        logical :: imported

        call set_empty(error_msg)
        if (.not. allocated(info%derived_types)) return
        do i = 1, size(info%derived_types)
            if (.not. fmod_derived_type_dependencies_available(info%derived_types(i), &
                                                               context) .and. &
                .not. allocated(use_node%only_list)) cycle
            call resolve_fmod_import_name(use_node, trim(info%derived_types(i)%name), &
                                          local_name, imported)
            if (.not. imported) local_name = trim(info%derived_types(i)%name)
            call import_fmod_derived_type(use_node, info%derived_types(i), &
                                          context, error_msg, local_name, &
                                          module_name=trim(info%name), &
                                          require_canonical_identity= &
                                              info%fmod_schema == FMOD_SCHEMA_VERSION)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine import_fmod_derived_types

    logical function fmod_derived_type_dependencies_available(dtype, context) &
            result(available)
        type(fmod_derived_type_t), intent(in) :: dtype
        type(lowering_context_t), intent(in) :: context
        integer :: i

        available = .true.
        if (.not. allocated(dtype%components)) return
        do i = 1, size(dtype%components)
            if (trim(dtype%components(i)%kind) /= 'derived') cycle
            if (allocated(dtype%components(i)%type_identity)) then
                if (len_trim(dtype%components(i)%type_identity) > 0) then
                    if (find_derived_type_identity(context, trim( &
                            dtype%components(i)%type_identity)) > 0) cycle
                end if
            end if
            if (find_derived_type(context, trim(dtype%components(i)%type_name)) <= 0) then
                available = .false.
                return
            end if
        end do
    end function fmod_derived_type_dependencies_available

    subroutine import_fmod_derived_type(use_node, dtype, context, error_msg, &
                                        local_name, module_name, &
                                        require_canonical_identity)
        ! Reconstruct one derived type's layout from its artefact record.
        type(use_statement_node), intent(in) :: use_node
        type(fmod_derived_type_t), intent(in) :: dtype
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: local_name
        character(len=*), intent(in), optional :: module_name
        logical, intent(in), optional :: require_canonical_identity
        integer :: type_index_new, k, nested, value_kind
        integer :: canonical_index, name_index, registered_index, parent_index
        integer(c_int64_t) :: no_default, slot_product, offset
        character(len=:), allocatable :: registered_name, canonical_name, &
            canonical_identity, record_name
        logical :: require_identity

        call set_empty(error_msg)
        require_identity = .false.
        if (present(require_canonical_identity)) require_identity = &
            require_canonical_identity
        if (.not. allocated(dtype%name)) return
        if (len_trim(dtype%name) == 0) return
        registered_name = trim(dtype%name)
        if (present(local_name)) then
            if (len_trim(local_name) > 0) registered_name = trim(local_name)
        end if
        canonical_name = trim(dtype%name)
        if (allocated(dtype%canonical_name)) then
            if (len_trim(dtype%canonical_name) > 0) then
                canonical_name = trim(dtype%canonical_name)
            end if
        end if
        canonical_identity = ''
        if (allocated(dtype%canonical_identity)) then
            if (len_trim(dtype%canonical_identity) > 0) &
                canonical_identity = trim(dtype%canonical_identity)
        end if
        if (len_trim(canonical_identity) == 0) then
            if (present(module_name)) then
                if (len_trim(module_name) > 0) canonical_identity = &
                    trim(module_name)//'::'//trim(canonical_name)
            end if
        end if
        if (len_trim(canonical_identity) == 0) canonical_identity = canonical_name
        canonical_index = find_derived_type_identity(context, canonical_identity)
        if (canonical_index > 0) then
            if (.not. same_name(registered_name, context%derived_types( &
                    canonical_index)%name)) then
                call register_derived_type_alias(context, registered_name, &
                                                 canonical_index, error_msg)
            end if
            return
        end if
        name_index = find_canonical_derived_type(context, canonical_name)
        if (name_index > 0 .and. same_name(registered_name, canonical_name)) then
            error_msg = 'derived type "'//trim(canonical_name)//'" from '// &
                trim(canonical_identity)//' conflicts with the imported type from '// &
                trim(context%derived_types(name_index)%canonical_identity)
            return
        end if
        registered_index = find_derived_type(context, trim(registered_name))
        if (registered_index > 0) then
            error_msg = 'derived type import name "'//trim(registered_name)// &
                '" conflicts with an existing derived type'
            return
        end if
        if (.not. allocated(dtype%components)) return

        record_name = trim(registered_name)
        call grow_derived_types(context)
        type_index_new = context%derived_type_count + 1
        context%derived_types(type_index_new)%name = trim(record_name)
        context%derived_types(type_index_new)%canonical_name = trim(canonical_name)
        context%derived_types(type_index_new)%canonical_identity = &
            trim(canonical_identity)
        context%derived_types(type_index_new)%component_count = 0
        context%derived_types(type_index_new)%parent_name = ''
        context%derived_types(type_index_new)%is_imported = .true.
        parent_index = 0
        if (allocated(dtype%parent_identity)) then
            if (len_trim(dtype%parent_identity) > 0) parent_index = &
                find_derived_type_identity(context, trim(dtype%parent_identity))
        end if
        if (parent_index > 0) then
            context%derived_types(type_index_new)%parent_name = &
                trim(context%derived_types(parent_index)%name)
        else if (allocated(dtype%parent_name)) then
            context%derived_types(type_index_new)%parent_name = &
                trim(dtype%parent_name)
        end if
        context%derived_type_count = type_index_new

        no_default = 0_c_int64_t
        offset = 0_c_int64_t
        do k = 1, size(dtype%components)
            associate (comp => dtype%components(k))
                if (comp%numeric_metadata_invalid) then
                    error_msg = 'corrupt .fmod derived layout: component '// &
                        trim(comp%name)//' of '//trim(dtype%name)// &
                        ' contains an unreadable numeric field'
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                if ((comp%is_pointer .and. (comp%is_allocatable .or. &
                                            comp%is_alloc_array)) .or. &
                    (comp%is_allocatable .and. comp%is_alloc_array)) then
                    error_msg = 'corrupt .fmod derived layout: component '// &
                        trim(comp%name)//' of '//trim(dtype%name)// &
                        ' has inconsistent component flags'
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                if (comp%alloc_rank < 0 .or. comp%alloc_rank > ARRAY_MAX_RANK) then
                    error_msg = 'corrupt .fmod derived layout: component '// &
                        trim(comp%name)//' of '//trim(dtype%name)// &
                        ' has an unsupported allocatable rank'
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                if (comp%is_alloc_array) then
                    if (comp%alloc_rank < 1) then
                        error_msg = 'corrupt .fmod derived layout: component '// &
                            trim(comp%name)//' of '//trim(dtype%name)// &
                            ' has an invalid allocatable array rank'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                else
                    if (comp%alloc_rank /= 0) then
                        error_msg = 'corrupt .fmod derived layout: component '// &
                            trim(comp%name)//' of '//trim(dtype%name)// &
                            ' has an allocatable rank without an array descriptor'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                end if
                if (int(comp%slot_offset, c_int64_t) /= offset) then
                    error_msg = 'corrupt .fmod derived layout: component '// &
                        trim(comp%name)//' of '//trim(dtype%name)// &
                        ' records slot offset '//integer_text(comp%slot_offset)// &
                        ' where its own component slot counts give '// &
                        integer64_text(offset)
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                if (offset > huge(offset) - int(comp%slot_count, c_int64_t)) then
                    error_msg = 'corrupt .fmod derived layout: slot offset accumulation overflow'
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                slot_product = int(comp%elem_count, c_int64_t) * &
                               int(comp%slot_width, c_int64_t)
                if (comp%slot_count < 1 .or. comp%elem_count < 1 .or. &
                    comp%slot_width < 1 .or. &
                    int(comp%slot_count, c_int64_t) /= slot_product) then
                    error_msg = 'corrupt .fmod derived layout: component '// &
                        trim(comp%name)//' of '//trim(dtype%name)// &
                        ' has inconsistent slot counts'
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                value_kind = fmod_component_value_kind(comp%kind)
                if (value_kind == 0) then
                    error_msg = 'unsupported .fmod component kind "'// &
                        trim(comp%kind)//'" for '//trim(comp%name)//' of '// &
                        trim(dtype%name)
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                nested = 0
                if (value_kind == VALUE_DERIVED) then
                    if (allocated(comp%type_identity)) then
                        if (len_trim(comp%type_identity) > 0) nested = &
                            find_derived_type_identity(context, &
                                                       trim(comp%type_identity))
                    end if
                    if (nested <= 0) nested = find_derived_type(context, &
                                                                  trim(comp%type_name))
                    if (nested <= 0) then
                        error_msg = 'unknown derived type "'// &
                            trim(comp%type_name)//'" for component '// &
                            trim(comp%name)//' of '//trim(dtype%name)// &
                            ': the .fmod does not define it'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                end if
                call add_derived_component(context, type_index_new, &
                    trim(comp%name), use_node%line, use_node%column, .false., &
                    no_default, comp%elem_count, value_kind, error_msg, &
                    comp_type_index=nested, &
                    is_allocatable=comp%is_allocatable, &
                    comp_char_length=comp%char_length, &
                    is_alloc_array=comp%is_alloc_array, &
                    alloc_rank=merge(max(1, comp%alloc_rank), 0, &
                                     comp%is_alloc_array), &
                    is_pointer=comp%is_pointer)
                if (len_trim(error_msg) > 0) then
                    context%derived_type_count = type_index_new - 1
                    return
                end if
                context%derived_types(type_index_new)%component_dim1(k) = &
                    comp%dim1
                offset = offset + int(comp%slot_count, c_int64_t)
            end associate
        end do
        ! Report identity omissions after structural validation. This keeps a
        ! more specific corruption diagnostic (unknown nested type, bad
        ! offsets, or inconsistent slot counts) useful for malformed artefacts,
        ! while still rejecting schema-14 records before they are promoted.
        if (require_identity) then
            if (.not. allocated(dtype%canonical_identity)) then
                error_msg = 'malformed .fmod derived type "'//trim(dtype%name)// &
                    '": missing canonical identity'
                context%derived_type_count = type_index_new - 1
                return
            end if
            if (len_trim(dtype%canonical_identity) == 0) then
                error_msg = 'malformed .fmod derived type "'//trim(dtype%name)// &
                    '": missing canonical identity'
                context%derived_type_count = type_index_new - 1
                return
            end if
            if (allocated(dtype%parent_name)) then
                if (len_trim(dtype%parent_name) > 0) then
                    if (.not. allocated(dtype%parent_identity)) then
                        error_msg = 'malformed .fmod derived type "'// &
                            trim(dtype%name)//'": missing parent identity'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                    if (len_trim(dtype%parent_identity) == 0) then
                        error_msg = 'malformed .fmod derived type "'// &
                            trim(dtype%name)//'": missing parent identity'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                end if
            end if
            if (allocated(dtype%components)) then
                do k = 1, size(dtype%components)
                    if (trim(dtype%components(k)%kind) /= 'derived') cycle
                    if (.not. allocated(dtype%components(k)%type_identity)) then
                        error_msg = 'malformed .fmod derived type "'// &
                            trim(dtype%name)//'": missing component identity'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                    if (len_trim(dtype%components(k)%type_identity) == 0) then
                        error_msg = 'malformed .fmod derived type "'// &
                            trim(dtype%name)//'": missing component identity'
                        context%derived_type_count = type_index_new - 1
                        return
                    end if
                end do
            end if
        end if
        if (allocated(dtype%bindings)) then
            if (size(dtype%bindings) > 0) then
                context%derived_types(type_index_new)%binding_count = &
                    size(dtype%bindings)
                allocate(context%derived_types(type_index_new)% &
                    binding_method_names(size(dtype%bindings)))
                allocate(context%derived_types(type_index_new)% &
                    binding_target_names(size(dtype%bindings)))
                allocate(context%derived_types(type_index_new)% &
                    binding_specific_names(size(dtype%bindings)))
                allocate(context%derived_types(type_index_new)% &
                    binding_pass_names(size(dtype%bindings)))
                allocate(context%derived_types(type_index_new)% &
                    binding_pass_args(size(dtype%bindings)))
                do k = 1, size(dtype%bindings)
                    context%derived_types(type_index_new)% &
                        binding_method_names(k) = dtype%bindings(k)%method_name
                    context%derived_types(type_index_new)% &
                        binding_target_names(k) = dtype%bindings(k)%target_name
                    context%derived_types(type_index_new)% &
                        binding_specific_names(k) = dtype%bindings(k)%specific_names
                    context%derived_types(type_index_new)% &
                        binding_pass_names(k) = dtype%bindings(k)%pass_name
                    context%derived_types(type_index_new)% &
                        binding_pass_args(k) = dtype%bindings(k)%pass_arg
                end do
            end if
        end if
        if (.not. same_name(registered_name, canonical_name)) then
            call register_derived_type_alias(context, registered_name, &
                                             type_index_new, error_msg)
        end if
    end subroutine import_fmod_derived_type

    function integer_text(value) result(text)
        integer, intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write (buffer, '(I0)') value
        text = trim(buffer)
    end function integer_text

    function integer64_text(value) result(text)
        integer(c_int64_t), intent(in) :: value
        character(len=:), allocatable :: text
        character(len=64) :: buffer

        write (buffer, '(I0)') value
        text = trim(buffer)
    end function integer64_text

    subroutine import_fmod_generic(use_node, module_name, gen, info, context, &
                                   error_msg)
        ! Register a use-associated generic interface so a call to its generic
        ! name resolves to the matching specific and links against the module
        ! object. Importing the generic also force-imports its specifics (they
        ! need not be named individually in an only-list) and records each
        ! specific's first-argument and return kind for resolution.
        type(use_statement_node), intent(in) :: use_node
        character(len=*), intent(in) :: module_name
        type(fmod_generic_t), intent(in) :: gen
        type(module_info_t), intent(in) :: info
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: rest, token
        character(len=:), allocatable :: local_name
        integer :: sep, ext_idx, next
        integer :: p
        logical :: imported

        call set_empty(error_msg)
        if (.not. allocated(gen%name)) return
        if (len_trim(gen%name) == 0) return
        call resolve_fmod_import_name(use_node, trim(gen%name), local_name, &
                                      imported)
        if (.not. imported) return
        if (.not. allocated(gen%specifics)) return
        if (find_generic(context, trim(local_name)) > 0) return
        call grow_generics(context)
        next = context%generic_count + 1
        context%generics(next)%generic_name = trim(local_name)
        context%generics(next)%specific_count = 0
        rest = trim(adjustl(gen%specifics))
        do while (len_trim(rest) > 0)
            sep = index(rest, ' ')
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
            if (len_trim(token) == 0) cycle
            do p = 1, size(info%procedures)
                if (.not. allocated(info%procedures(p)%name)) cycle
                if (same_name(info%procedures(p)%name, token)) then
                    call import_fmod_procedure(use_node, module_name, &
                        info%procedures(p), context, error_msg, force=.true., &
                        require_class_identities= &
                            info%fmod_schema == FMOD_SCHEMA_VERSION)
                    if (len_trim(error_msg) > 0) return
                    exit
                end if
            end do
            ext_idx = external_procedure_index(context, trim(token))
            if (ext_idx <= 0) cycle
            call add_generic_specific_from_external(context, next, trim(token), &
                                                    ext_idx)
        end do
        if (context%generics(next)%specific_count > 0) &
            context%generic_count = next
    end subroutine import_fmod_generic

 subroutine add_generic_specific_from_external(context, generic_idx, name, &
                                                   ext_idx)
        ! Append a specific to a generic entry, taking its signature and
        ! return kinds from an already-registered external procedure rather
        ! than from an AST node (the specific was imported from a .fmod, #284).
        ! Ranks come from the artefact, so an imported generic resolves a
        ! rank-1 or rank-2 specific exactly as a same-unit call does (#415).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: generic_idx
        character(len=*), intent(in) :: name
        integer, intent(in) :: ext_idx
        integer :: sc
        integer :: j

        if (context%generics(generic_idx)%specific_count >= &
            MAX_GENERIC_SPECIFICS) return
        context%generics(generic_idx)%specific_count = &
            context%generics(generic_idx)%specific_count + 1
        sc = context%generics(generic_idx)%specific_count
        context%generics(generic_idx)%specific_names(sc) = trim(name)
        context%generics(generic_idx)%return_kinds(sc) = &
            context%external_procedures(ext_idx)%return_value_kind
        context%generics(generic_idx)%specific_arg_counts(sc) = &
            context%external_procedures(ext_idx)%arg_count
        context%generics(generic_idx)%specific_arg_kinds(:, sc) = VALUE_I32
        context%generics(generic_idx)%specific_arg_ranks(:, sc) = 0
        do j = 1, min(context%external_procedures(ext_idx)%arg_count, &
            MAX_PROC_ARGS)
            context%generics(generic_idx)%specific_arg_kinds(j, sc) = &
                context%external_procedures(ext_idx)%arg_value_kinds(j)
            context%generics(generic_idx)%specific_arg_ranks(j, sc) = &
                context%external_procedures(ext_idx)%arg_ranks(j)
        end do
    end subroutine add_generic_specific_from_external

    subroutine import_fmod_procedure(use_node, module_name, proc, context, &
                                     error_msg, force, require_class_identities)
        ! Register a separately compiled procedure from a .fmod. Module
        ! procedures use __<module>_MOD_<name>; public plain INTERFACE
        ! procedures retain their unmangled external symbol (#284). The
        ! procedure passes arguments by reference; scalar functions and
        ! subroutines whose arguments are all supported are recorded in the
        ! .fmod, carrying each argument's kind so the call materialises the
        ! actual with the right type. force=.true. imports
        ! the specific of a use-associated generic even when it is not named in
        ! the only-list.
        type(use_statement_node), intent(in) :: use_node
        character(len=*), intent(in) :: module_name
        type(fmod_procedure_t), intent(in) :: proc
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in), optional :: force
        logical, intent(in), optional :: require_class_identities
        character(len=:), allocatable :: mangled
        character(len=:), allocatable :: local_name
        integer, allocatable :: arg_kinds(:)
        character(len=64), allocatable :: arg_names(:)
        logical, allocatable :: arg_optionals(:), arg_values(:)
        logical, allocatable :: arg_opaque(:)
        logical, allocatable :: arg_class(:)
        character(len=64), allocatable :: arg_class_types(:)
        character(len=:), allocatable :: arg_class_type_identities(:)
        integer, allocatable :: arg_intents(:), arg_ranks(:), arg_extents(:)
        integer :: return_kind
        logical :: forced, imported

        call set_empty(error_msg)
        forced = .false.
        if (present(force)) forced = force
        if (.not. allocated(proc%name)) return
        if (len_trim(proc%name) == 0) return
        if (.not. proc%callable) return
        if (forced) then
            local_name = trim(proc%name)
            imported = .true.
        else
            call resolve_fmod_import_name(use_node, trim(proc%name), &
                                          local_name, imported)
        end if
        if (.not. imported) return
        if (external_procedure_index(context, trim(local_name)) > 0) return
        if (contained_function_kind(context, trim(local_name)) /= 0) return
        if (trim(proc%kind) == 'subroutine') then
            return_kind = VALUE_SUBROUTINE
        else if (trim(proc%kind) == 'integer') then
            return_kind = VALUE_I32
        else if (trim(proc%kind) == 'integer8') then
            return_kind = VALUE_I64
        else if (trim(proc%kind) == 'integer1') then
            return_kind = VALUE_I8
        else if (trim(proc%kind) == 'integer2') then
            return_kind = VALUE_I16
        else if (trim(proc%kind) == 'real') then
            return_kind = VALUE_F32
        else if (trim(proc%kind) == 'real8') then
            return_kind = VALUE_F64
        else if (trim(proc%kind) == 'complex') then
            return_kind = VALUE_C4
        else if (trim(proc%kind) == 'complex8') then
            return_kind = VALUE_C8
        else
            return
        end if
        call validate_fmod_procedure_metadata(proc, error_msg, &
                                              require_class_identities)
        if (len_trim(error_msg) > 0) return
        call parse_fmod_arg_kinds(proc, arg_kinds, error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_fmod_arg_names(proc, arg_names)
        call parse_fmod_arg_flags(proc%arg_optionals, proc%nargs, arg_optionals, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_fmod_arg_flags(proc%arg_values, proc%nargs, arg_values, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_fmod_arg_opaque(proc, arg_opaque)
        call parse_fmod_arg_classes(proc, arg_class)
        call parse_fmod_arg_class_types(proc, arg_class_types)
        call parse_fmod_arg_class_type_identities(proc, &
                                                  arg_class_type_identities)
        call parse_fmod_arg_intents(proc, arg_intents)
        call parse_fmod_arg_integers(proc%arg_ranks, proc%nargs, arg_ranks, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_fmod_arg_integers(proc%arg_extents, proc%nargs, arg_extents, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        if (any(arg_ranks < 0) .or. any(arg_ranks > ARRAY_MAX_RANK)) then
            error_msg = 'malformed .fmod rank metadata'
            return
        end if
        if (any(arg_extents < 0)) then
            error_msg = 'malformed .fmod extent metadata'
            return
        end if
        if (proc%external_binding) then
            if (allocated(proc%external_name) .and. &
                len_trim(proc%external_name) > 0) then
                mangled = trim(proc%external_name)
            else
                mangled = trim(proc%name)
            end if
        else
            mangled = module_procedure_mangled(module_name, trim(proc%name))
        end if
        call add_external_procedure(context, trim(local_name), mangled, &
                                    return_kind, proc%nargs, .not. proc%external_binding, &
                                    arg_kinds, &
                                    arg_names, arg_optionals, arg_values, &
                                    arg_intents, arg_ranks, arg_extents, &
                                    is_bind_c=proc%external_binding, &
                                    arg_opaque=arg_opaque, &
                                    arg_class=arg_class, &
                                    arg_class_types=arg_class_types, &
                                    arg_class_type_identities= &
                                        arg_class_type_identities)
    end subroutine import_fmod_procedure

    subroutine validate_fmod_procedure_metadata(proc, error_msg, &
                                                require_class_identities)
        ! Schema 13+ class metadata is an ABI contract, not a hint. Reject a
        ! partial or contradictory pair before the importer can default a
        ! malformed class dummy to the raw-storage ABI.
        type(fmod_procedure_t), intent(in) :: proc
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in), optional :: require_class_identities
        logical :: have_classes, have_class_types
        logical :: have_class_identities
        logical :: require_identities
        logical, allocatable :: classes(:)
        character(len=64), allocatable :: class_types(:)
        character(len=:), allocatable :: class_identities(:)
        integer :: i

        call set_empty(error_msg)
        require_identities = .false.
        if (present(require_class_identities)) require_identities = &
            require_class_identities
        if (proc%nargs < 0 .or. proc%nargs > MAX_PROC_ARGS) then
            error_msg = 'malformed .fmod procedure "'//trim(proc%name)// &
                '" has an invalid argument count'
            return
        end if
        have_classes = .false.
        if (allocated(proc%arg_classes)) then
            if (len_trim(proc%arg_classes) > 0) have_classes = .true.
        end if
        have_class_types = .false.
        if (allocated(proc%arg_class_types)) then
            if (len_trim(proc%arg_class_types) > 0) have_class_types = .true.
        end if
        have_class_identities = .false.
        if (allocated(proc%arg_class_type_identities)) then
            if (len_trim(proc%arg_class_type_identities) > 0) &
                have_class_identities = .true.
        end if
        if (have_classes .neqv. have_class_types) then
            error_msg = 'malformed .fmod class metadata for procedure "'// &
                trim(proc%name)//'": flags and declared types must both be present'
            return
        end if
        if (.not. have_classes) return
        if (fmod_token_count(proc%arg_classes) /= proc%nargs .or. &
            fmod_token_count(proc%arg_class_types) /= proc%nargs) then
            error_msg = 'malformed .fmod class metadata for procedure "'// &
                trim(proc%name)//'": one value is required per argument'
            return
        end if
        if (have_class_identities) then
            if (fmod_token_count(proc%arg_class_type_identities) /= proc%nargs) then
                error_msg = 'malformed .fmod class metadata for procedure "'// &
                    trim(proc%name)//'": one identity is required per argument'
                return
            end if
        end if
        call validate_fmod_flag_tokens(proc%arg_classes, proc%nargs, &
                                        trim(proc%name), error_msg)
        if (len_trim(error_msg) > 0) return
        call parse_fmod_arg_classes(proc, classes)
        call parse_fmod_arg_class_types(proc, class_types)
        call parse_fmod_arg_class_type_identities(proc, class_identities)
        if (require_identities .and. any(classes)) then
            if (.not. have_class_identities) then
                error_msg = 'malformed .fmod class metadata for procedure "'// &
                    trim(proc%name)//'": canonical identities are required'
                return
            end if
        end if
        do i = 1, proc%nargs
            if (classes(i)) then
                if (len_trim(class_types(i)) == 0) then
                    error_msg = 'malformed .fmod class metadata for procedure "'// &
                        trim(proc%name)//'": a class argument needs a declared type'
                    return
                end if
                if (require_identities) then
                    if (len_trim(class_identities(i)) == 0) then
                        error_msg = 'malformed .fmod class metadata for procedure "'// &
                            trim(proc%name)//'": a class argument needs a canonical identity'
                        return
                    end if
                end if
            else
                if (len_trim(class_types(i)) > 0) then
                    error_msg = 'malformed .fmod class metadata for procedure "'// &
                        trim(proc%name)//'": a non-class argument has a class type'
                    return
                end if
            end if
        end do
    end subroutine validate_fmod_procedure_metadata

    subroutine validate_fmod_flag_tokens(text, nargs, procedure_name, error_msg)
        character(len=:), allocatable, intent(in) :: text
        integer, intent(in) :: nargs
        character(len=*), intent(in) :: procedure_name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: rest, token
        integer :: i, sep

        call set_empty(error_msg)
        rest = trim(adjustl(text))
        do i = 1, nargs
            sep = index(rest, ' ')
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = trim(rest(1:sep - 1))
                rest = adjustl(rest(sep + 1:))
            end if
            if (token /= '0' .and. token /= '1') then
                error_msg = 'malformed .fmod class metadata for procedure "'// &
                    trim(procedure_name)//'": flags must be 0 or 1'
                return
            end if
        end do
    end subroutine validate_fmod_flag_tokens

    integer function fmod_token_count(text) result(count)
        character(len=:), allocatable, intent(in) :: text
        character(len=:), allocatable :: rest
        integer :: sep

        count = 0
        if (.not. allocated(text)) return
        rest = trim(adjustl(text))
        do while (len_trim(rest) > 0)
            count = count + 1
            sep = index(rest, ' ')
            if (sep == 0) exit
            rest = adjustl(rest(sep + 1:))
        end do
    end function fmod_token_count

    subroutine parse_fmod_arg_opaque(proc, opaque)
        ! Opaque procedures may still carry precise per-dummy kinds. Preserve
        ! the old all-opaque interpretation only for artefacts whose tokens
        ! are all opaque; newer artefacts mark just unsupported dummies.
        type(fmod_procedure_t), intent(in) :: proc
        logical, allocatable, intent(out) :: opaque(:)
        character(len=:), allocatable :: rest, token
        integer :: i, sep

        allocate (opaque(max(proc%nargs, 0)))
        opaque = .false.
        if (proc%nargs <= 0) return
        if (.not. allocated(proc%arg_kinds)) then
            opaque = proc%opaque
            return
        end if
        rest = trim(adjustl(proc%arg_kinds))
        i = 0
        do while (len_trim(rest) > 0 .and. i < proc%nargs)
            sep = index(rest, ' ')
            i = i + 1
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
            opaque(i) = trim(token) == 'opaque'
        end do
        if (i /= proc%nargs) opaque = proc%opaque
    end subroutine parse_fmod_arg_opaque

    subroutine parse_fmod_arg_classes(proc, classes)
        type(fmod_procedure_t), intent(in) :: proc
        logical, allocatable, intent(out) :: classes(:)

        call parse_fmod_arg_flags(proc%arg_classes, proc%nargs, classes)
    end subroutine parse_fmod_arg_classes

    subroutine parse_fmod_arg_class_types(proc, class_types)
        type(fmod_procedure_t), intent(in) :: proc
        character(len=64), allocatable, intent(out) :: class_types(:)
        character(len=:), allocatable :: rest, token
        integer :: i, sep

        allocate(class_types(max(proc%nargs, 0)))
        class_types = ''
        if (proc%nargs <= 0) return
        if (.not. allocated(proc%arg_class_types)) return
        rest = trim(adjustl(proc%arg_class_types))
        i = 0
        do while (len_trim(rest) > 0 .and. i < proc%nargs)
            sep = index(rest, ' ')
            i = i + 1
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = trim(rest(1:sep - 1))
                rest = adjustl(rest(sep + 1:))
            end if
            if (trim(token) /= '-') class_types(i) = trim(token)
        end do
    end subroutine parse_fmod_arg_class_types

    subroutine parse_fmod_arg_class_type_identities(proc, identities)
        type(fmod_procedure_t), intent(in) :: proc
        character(len=:), allocatable, intent(out) :: identities(:)
        character(len=:), allocatable :: rest, token
        integer :: i, sep, identity_length

        identity_length = 1
        if (allocated(proc%arg_class_type_identities)) identity_length = max( &
            identity_length, len_trim(proc%arg_class_type_identities))
        allocate (character(len=identity_length) :: identities(max(proc%nargs, 0)))
        identities = ''
        if (proc%nargs <= 0) return
        if (.not. allocated(proc%arg_class_type_identities)) return
        rest = trim(adjustl(proc%arg_class_type_identities))
        i = 0
        do while (len_trim(rest) > 0 .and. i < proc%nargs)
            sep = index(rest, ' ')
            i = i + 1
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = trim(rest(1:sep - 1))
                rest = adjustl(rest(sep + 1:))
            end if
            if (trim(token) /= '-') identities(i) = trim(token)
        end do
    end subroutine parse_fmod_arg_class_type_identities

    subroutine parse_fmod_arg_flags(text, nargs, flags, error_msg)
        ! Split a .fmod per-dummy '0'/'1' attribute list (OPTIONAL, VALUE) into
        ! logicals. A shorter or absent list leaves the remaining dummies
        ! without the attribute (#397).
        character(len=:), allocatable, intent(in) :: text
        integer, intent(in) :: nargs
        logical, allocatable, intent(out) :: flags(:)
        character(len=:), allocatable, intent(out), optional :: error_msg
        character(len=:), allocatable :: rest
        character(len=:), allocatable :: token
        integer :: n, sep, value

        if (present(error_msg)) error_msg = ''
        allocate (flags(max(nargs, 0)))
        if (nargs <= 0) then
            if (present(error_msg)) then
                if (allocated(text)) then
                    if (len_trim(text) > 0) &
                        error_msg = 'malformed .fmod integer flag metadata'
                end if
            end if
            return
        end if
        flags = .false.
        if (.not. allocated(text)) return
        rest = trim(adjustl(text))
        n = 0
        do while (len_trim(rest) > 0)
            if (n >= nargs) then
                if (present(error_msg)) &
                    error_msg = 'malformed .fmod integer flag metadata'
                return
            end if
            sep = index(rest, ' ')
            n = n + 1
            if (sep == 0) then
                token = trim(rest)
                rest = ''
            else
                token = trim(rest(1:sep - 1))
                rest = adjustl(rest(sep + 1:))
            end if
            if (.not. parse_strict_integer(token, value) .or. &
                (value /= 0 .and. value /= 1)) then
                if (present(error_msg)) &
                    error_msg = 'malformed .fmod integer flag metadata'
                return
            end if
            flags(n) = value /= 0
        end do
    end subroutine parse_fmod_arg_flags

    subroutine parse_fmod_arg_integers(text, nargs, values, error_msg)
        ! Split a .fmod per-dummy integer list (ranks, extents) into values. An
        ! absent or short list leaves the remaining dummies at 0 (#415).
        character(len=:), allocatable, intent(in) :: text
        integer, intent(in) :: nargs
        integer, allocatable, intent(out) :: values(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: rest, token
        integer :: n, sep, value

        call set_empty(error_msg)
        allocate (values(max(nargs, 0)))
        if (nargs <= 0) then
            if (allocated(text)) then
                if (len_trim(text) > 0) &
                    error_msg = 'malformed .fmod integer argument metadata'
            end if
            return
        end if
        values = 0
        if (.not. allocated(text)) return
        rest = trim(adjustl(text))
        n = 0
        do while (len_trim(rest) > 0 .and. n < nargs)
            sep = index(rest, ' ')
            n = n + 1
            if (sep == 0) then
                token = trim(rest)
                rest = ''
            else
                token = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
            if (.not. parse_strict_integer(token, value)) then
                error_msg = 'malformed .fmod integer argument metadata'
                return
            end if
            values(n) = value
        end do
        if (len_trim(rest) > 0) &
            error_msg = 'malformed .fmod integer argument metadata'
    end subroutine parse_fmod_arg_integers

    subroutine parse_fmod_arg_intents(proc, intents)
        ! Split the .fmod per-dummy INTENT token list into intent codes (#397).
        type(fmod_procedure_t), intent(in) :: proc
        integer, allocatable, intent(out) :: intents(:)
        character(len=:), allocatable :: rest, token
        integer :: n, sep

        allocate (intents(max(proc%nargs, 0)))
        if (proc%nargs <= 0) return
        intents = ARG_INTENT_NONE
        if (.not. allocated(proc%arg_intents)) return
        rest = trim(adjustl(proc%arg_intents))
        n = 0
        do while (len_trim(rest) > 0 .and. n < proc%nargs)
            sep = index(rest, ' ')
            n = n + 1
            if (sep == 0) then
                token = trim(rest)
                rest = ''
            else
                token = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
            select case (trim(token))
            case ('in')
                intents(n) = ARG_INTENT_IN
            case ('out')
                intents(n) = ARG_INTENT_OUT
            case ('inout')
                intents(n) = ARG_INTENT_INOUT
            case default
                intents(n) = ARG_INTENT_NONE
            end select
        end do
    end subroutine parse_fmod_arg_intents

    subroutine parse_fmod_arg_names(proc, arg_names)
        ! Split the .fmod arg_names token string into per-dummy names, which a
        ! call site uses to place keyword actuals (#408). An artefact written
        ! before those names existed yields blanks, and keyword association
        ! against that signature is then rejected rather than guessed.
        type(fmod_procedure_t), intent(in) :: proc
        character(len=64), allocatable, intent(out) :: arg_names(:)
        character(len=:), allocatable :: rest
        integer :: n, sep

        allocate (arg_names(max(proc%nargs, 0)))
        if (proc%nargs <= 0) return
        arg_names = ''
        if (.not. allocated(proc%arg_names)) return
        rest = trim(adjustl(proc%arg_names))
        n = 0
        do while (len_trim(rest) > 0 .and. n < proc%nargs)
            sep = index(rest, ' ')
            n = n + 1
            if (sep == 0) then
                arg_names(n) = rest
                rest = ''
            else
                arg_names(n) = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
        end do
    end subroutine parse_fmod_arg_names

    subroutine parse_fmod_arg_kinds(proc, arg_kinds, error_msg)
        ! Split the .fmod arg_kinds token string into per-argument value kinds.
        ! An unrecognised or count-mismatched token list means the artefact does
        ! not describe a callable signature, so the procedure is left out (#284).
        type(fmod_procedure_t), intent(in) :: proc
        integer, allocatable, intent(out) :: arg_kinds(:)
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: rest, token
        integer :: n, value_kind, sep

        call set_empty(error_msg)
        allocate (arg_kinds(max(proc%nargs, 0)))
        if (proc%nargs <= 0) return
        if (.not. allocated(proc%arg_kinds)) then
            error_msg = 'missing argument kinds for module procedure '// &
                trim(proc%name)
            return
        end if
        rest = trim(adjustl(proc%arg_kinds))
        n = 0
        do while (len_trim(rest) > 0 .and. n < proc%nargs)
            sep = index(rest, ' ')
            if (sep == 0) then
                token = rest
                rest = ''
            else
                token = rest(1:sep - 1)
                rest = adjustl(rest(sep + 1:))
            end if
            if (trim(token) == 'opaque') then
                value_kind = 0
            else
                value_kind = value_kind_of_token(token)
            end if
            if (value_kind == 0 .and. trim(token) /= 'opaque') then
                error_msg = 'unrecognised argument kind "'//trim(token)// &
                    '" for module procedure '//trim(proc%name)
                return
            end if
            n = n + 1
            arg_kinds(n) = value_kind
        end do
        if (n /= proc%nargs) then
            error_msg = 'argument kind count mismatch for module procedure '// &
                trim(proc%name)
        end if
    end subroutine parse_fmod_arg_kinds

    logical function module_defined_in_unit(arena, module_lowered)
        ! Whether a module of this name is defined anywhere in this arena.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_lowered
        integer :: n

        module_defined_in_unit = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (module_node)
                if (allocated(node%name)) then
                    if (lowercase_text(node%name) == module_lowered) then
                        module_defined_in_unit = .true.
                        return
                    end if
                end if
            end select
        end do
    end function module_defined_in_unit

    logical function is_intrinsic_module(name)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: lowered

        lowered = lowercase_text(name)
        is_intrinsic_module = lowered == 'iso_c_binding' .or. &
                              lowered == 'iso_fortran_env' .or. &
                              lowered == 'ieee_arithmetic' .or. &
                              lowered == 'ieee_exceptions' .or. &
                              lowered == 'ieee_features' .or. &
                              lowered == 'omp_lib'
    end function is_intrinsic_module

    function find_fmod_on_path(context, name) result(path)
        ! First match of <dir>/<name>.fmod across the include paths and the
        ! current directory wins.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: path
        character(len=:), allocatable :: candidate
        logical :: exists
        integer :: i

        path = ''
        do i = 1, context%include_path_count
            candidate = trim(context%include_paths(i))//'/'//trim(name)//'.fmod'
            inquire (file=candidate, exist=exists)
            if (exists) then
                path = candidate
                return
            end if
        end do
        candidate = trim(name)//'.fmod'
        inquire (file=candidate, exist=exists)
        if (exists) path = candidate
    end function find_fmod_on_path

    subroutine import_fmod_parameter(use_node, param, context, error_msg)
        type(use_statement_node), intent(in) :: use_node
        type(fmod_parameter_t), intent(in) :: param
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: constant_value
        integer :: index, existing_idx
        character(len=:), allocatable :: local_name
        logical :: imported
        logical :: rebind, skip

        call set_empty(error_msg)
        if (.not. allocated(param%name)) return
        if (len_trim(param%name) == 0) return
        call resolve_fmod_import_name(use_node, trim(param%name), local_name, &
                                      imported)
        if (.not. imported) return
        ! Only integer named constants are importable today.
        if (.not. allocated(param%kind)) return
        if (trim(param%kind) /= 'integer') return
        ! A parameter whose initializer was not representable in the current
        ! .fmod schema is emitted with an empty value.  It remains a valid
        ! exported name, but there is no constant to bind in the using unit;
        ! defer that unsupported value instead of rejecting an otherwise
        ! independent USE list (#584).
        if (.not. allocated(param%value)) return
        if (len_trim(param%value) == 0) return
        if (.not. parse_strict_int64(param%value, constant_value)) then
            error_msg = 'malformed integer value in .fmod for parameter '// &
                        trim(param%name)
            return
        end if
        call check_use_rename_collision(context, trim(local_name), &
                                        trim(use_node%module_name), &
                                        trim(param%name), existing_idx, &
                                        rebind, error_msg)
        if (len_trim(error_msg) > 0) return
        call use_import_symbol_slot(context, existing_idx, rebind, index, skip)
        if (skip) return
        context%symbols(index)%name = trim(local_name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, &
                                                      constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        call record_use_rename_source(context, index, &
                                      trim(use_node%module_name), &
                                      trim(param%name))
    end subroutine import_fmod_parameter

    subroutine import_fmod_variable(use_node, var, context, error_msg)
        ! Add a module variable from a .fmod record to the symbol table (#249 B7a).
        ! The global `__<mod>_MOD_<name>` must already exist in the session
        ! (emitted when the defining module was compiled); we resolve it with
        ! lr_session_intern and set up an addressed symbol so that load/store
        ! work in the using unit.
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(use_statement_node), intent(in) :: use_node
        type(fmod_variable_t), intent(in) :: var
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: mangled
        character(len=:), allocatable :: local_name
        character(kind=c_char), allocatable :: c_name(:)
        integer(c_int32_t) :: global_id
        integer :: sym_idx, value_kind, existing_idx
        logical :: imported
        logical :: rebind, skip

        call set_empty(error_msg)
        if (.not. allocated(var%name)) return
        if (len_trim(var%name) == 0) return
        call resolve_fmod_import_name(use_node, trim(var%name), local_name, &
                                      imported)
        if (.not. imported) return
        value_kind = value_kind_of_token(trim(var%kind))
        if (value_kind == 0) return
        call check_use_rename_collision(context, trim(local_name), &
                                        trim(use_node%module_name), &
                                        trim(var%name), existing_idx, rebind, &
                                        error_msg)
        if (len_trim(error_msg) > 0) return
        if (existing_idx > 0) then
            if (.not. rebind) return
        end if
        if (allocated(var%c_name) .and. len_trim(var%c_name) > 0) then
            mangled = trim(var%c_name)
        else
            mangled = module_procedure_mangled( &
                trim(use_node%module_name), trim(var%name))
        end if
        call to_c_chars(mangled, c_name)
        global_id = lr_session_intern(context%session%handle, c_name)
        if (global_id < 0_c_int32_t) then
            error_msg = 'module variable global not found: '//trim(mangled)
            return
        end if
        call use_import_symbol_slot(context, existing_idx, rebind, sym_idx, skip)
        if (skip) return
        context%symbols(sym_idx)%name = trim(local_name)
        context%symbols(sym_idx)%value_kind = value_kind
        context%symbols(sym_idx)%has_address = .true.
        context%symbols(sym_idx)%is_reference = .true.
        context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
        context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
        context%symbols(sym_idx)%address%typ = lr_type_ptr_s(context%session%handle)
        context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
        call record_use_rename_source(context, sym_idx, &
                                      trim(use_node%module_name), &
                                      trim(var%name))
    end subroutine import_fmod_variable

    subroutine check_use_rename_collision(context, local_name, module_name, &
                                          remote_name, existing_idx, rebind, &
                                          error_msg)
        ! Classify a USE import of module_name::remote_name under local_name
        ! against the symbols already registered in this unit.
        !   existing_idx == 0                 the local name is free.
        !   existing_idx > 0, rebind = .true. the local name is held by a
        !       symbol with no recorded import origin; it adopts this remote
        !       binding and the caller rebinds that slot in place.
        !   existing_idx > 0, rebind = .false. the very same remote binding is
        !       already imported under this local name; the repeat is valid
        !       (F2018 19.5.2) and the caller skips it.
        ! A different remote binding under an already-bound local name is
        ! ambiguous and rejected.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: local_name
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: remote_name
        integer, intent(out) :: existing_idx
        logical, intent(out) :: rebind
        character(len=:), allocatable, intent(out) :: error_msg

        rebind = .false.
        call set_empty(error_msg)
        existing_idx = find_symbol_compat(context, local_name)
        if (existing_idx <= 0) return
        if (.not. allocated(context%symbols(existing_idx)%use_rename_module)) then
            rebind = .true.
            return
        end if
        if (.not. allocated(context%symbols(existing_idx)%use_rename_remote)) &
            return
        if (same_name(context%symbols(existing_idx)%use_rename_module, &
                      module_name)) then
            if (same_name(context%symbols(existing_idx)%use_rename_remote, &
                          remote_name)) return
        end if
        error_msg = 'ambiguous use rename local name: '//trim(local_name)
    end subroutine check_use_rename_collision

    subroutine use_import_symbol_slot(context, existing_idx, rebind, sym_idx, &
                                      skip)
        ! Pick the symbol table slot a USE import writes into: the adopted
        ! existing slot, or a fresh one. skip = .true. when the same binding is
        ! already imported and nothing must be written.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: existing_idx
        logical, intent(in) :: rebind
        integer, intent(out) :: sym_idx
        logical, intent(out) :: skip

        skip = .false.
        sym_idx = 0
        if (existing_idx > 0) then
            if (.not. rebind) then
                skip = .true.
                return
            end if
            sym_idx = existing_idx
            context%symbols(sym_idx) = symbol_t()
            return
        end if
        call grow_symbols(context)
        sym_idx = context%symbol_count + 1
        context%symbol_count = sym_idx
    end subroutine use_import_symbol_slot

    subroutine record_use_rename_source(context, sym_idx, module_name, &
                                        remote_name)
        ! Remember the remote binding a renamed local symbol came from so a
        ! later import of the same binding is recognised as a repeat.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym_idx
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: remote_name

        if (sym_idx <= 0) return
        context%symbols(sym_idx)%use_rename_module = trim(module_name)
        context%symbols(sym_idx)%use_rename_remote = trim(remote_name)
    end subroutine record_use_rename_source

    subroutine import_use_renames(arena, export_idx, use_node, context, error_msg)
        ! use m, only: local => remote binds module symbol `remote` under the
        ! local name. rename_list is a flat [local, remote, ...] array.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: j
        integer :: param_idx
        integer :: value_kind
        logical :: found
        integer(c_int64_t) :: constant_value
        integer :: init_index
        integer :: line
        integer :: column
        integer :: symbol_idx
        integer :: existing_idx
        logical :: rebind
        logical :: skip
        character(len=:), allocatable :: decl_name
        character(len=:), allocatable :: decl_type_name

        call set_empty(error_msg)
        if (.not. allocated(use_node%rename_list)) return

        do i = 1, size(use_node%rename_list) - 1, 2
            found = .false.
            ! Exported named constant: fold into an integer value symbol.
            do j = 1, context%module_exports(export_idx)%parameter_count
                param_idx = context%module_exports(export_idx)%parameter_indices(j)
                if (.not. node_exists(arena, param_idx)) cycle
                if (.not. is_declaration_node(arena, param_idx)) cycle
                call get_declaration_var_name(arena, param_idx, decl_name, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. same_name(decl_name, use_node%rename_list(i + 1)%s)) &
                    cycle
                call get_declaration_type_name(arena, param_idx, decl_type_name, &
                                               error_msg)
                if (len_trim(error_msg) > 0) return
                line = get_node_line(arena, param_idx)
                column = get_node_column(arena, param_idx)
                call type_name_value_kind(decl_type_name, line, column, &
                                          value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
                if (value_kind /= VALUE_I32) then
                    call unsupported_feature_error('renamed parameter', line, &
                                                   column, 'direct LIRIC '// &
                                                   'session only supports '// &
                                                   'integer scalar parameters', &
                                                   error_msg)
                    return
                end if
                if (.not. get_declaration_has_initializer(arena, param_idx)) then
                    error_msg = 'renamed parameter has no initializer: '// &
                                trim(decl_name)
                    return
                end if
                init_index = get_declaration_initializer_index(arena, param_idx)
                if (init_index <= 0) then
                    error_msg = 'renamed parameter has no initializer: '// &
                                trim(decl_name)
                    return
                end if
                call eval_i32_constant(context%arena, init_index, context, &
                                       constant_value, error_msg)
                if (len_trim(error_msg) > 0) return
                call check_use_rename_collision(context, &
                    trim(use_node%rename_list(i)%s), &
                    trim(context%module_exports(export_idx)%module_name), &
                    trim(use_node%rename_list(i + 1)%s), existing_idx, &
                    rebind, error_msg)
                if (len_trim(error_msg) > 0) return
                call use_import_symbol_slot(context, existing_idx, rebind, &
                                            symbol_idx, skip)
                if (skip) then
                    found = .true.
                    exit
                end if
                context%symbols(symbol_idx)%name = trim(use_node%rename_list(i)%s)
                context%symbols(symbol_idx)%value_kind = VALUE_I32
                context%symbols(symbol_idx)%value = i32_immediate(context%session, &
                                                                  constant_value)
                context%symbols(symbol_idx)%is_parameter = .true.
                context%symbols(symbol_idx)%has_i32_constant = .true.
                context%symbols(symbol_idx)%i32_constant = constant_value
                call record_use_rename_source(context, symbol_idx, &
                    trim(context%module_exports(export_idx)%module_name), &
                    trim(use_node%rename_list(i + 1)%s))
                found = .true.
                exit
            end do
            if (found) cycle
            ! Exported module variable: bind the shared global under the local
            ! name (#274).
            call import_renamed_variable(arena, export_idx, &
                trim(use_node%rename_list(i + 1)%s), &
                trim(use_node%rename_list(i)%s), context, found, error_msg)
            if (len_trim(error_msg) > 0) return
            if (found) cycle
            ! Exported module procedure: alias the call name (#274).
            call import_renamed_procedure(arena, &
                trim(context%module_exports(export_idx)%module_name), &
                trim(use_node%rename_list(i + 1)%s), &
                trim(use_node%rename_list(i)%s), context, found, error_msg)
            if (len_trim(error_msg) > 0) return
            if (found) cycle
            error_msg = 'use rename target is not an exported parameter of '// &
                trim(use_node%module_name)//': '// &
                trim(use_node%rename_list(i + 1)%s)
            return
        end do
    end subroutine import_use_renames

    subroutine import_renamed_variable(arena, export_idx, remote_name, &
                                       local_name, context, found, error_msg)
        ! Bind an exported module variable to a local name (#274). Mirrors
        ! bind_one_module_variable but interns the global under remote_name and
        ! registers the symbol under local_name.
        use, intrinsic :: iso_c_binding, only: c_char, c_int32_t, c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        character(len=*), intent(in) :: remote_name
        character(len=*), intent(in) :: local_name
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: found
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, var_idx, value_kind, status, sym_idx, existing_idx
        logical :: rebind, skip
        integer(c_int32_t) :: global_id
        character(len=:), allocatable :: var_name, mangled, name_err
        character(kind=c_char), allocatable :: c_name(:)

        found = .false.
        call set_empty(error_msg)
        do i = 1, context%module_exports(export_idx)%variable_count
            var_idx = context%module_exports(export_idx)%variable_indices(i)
            if (.not. node_exists(arena, var_idx)) cycle
            call get_declaration_var_name(arena, var_idx, var_name, name_err)
            if (len_trim(name_err) > 0) cycle
            if (.not. same_name(var_name, remote_name)) cycle
            call module_variable_kind(arena, var_idx, value_kind, status, &
                                      context, error_msg)
            if (status /= MODVAR_OK) then
                call set_empty(error_msg)
                return
            end if
            call check_use_rename_collision(context, local_name, &
                trim(context%module_exports(export_idx)%module_name), &
                remote_name, existing_idx, rebind, error_msg)
            if (len_trim(error_msg) > 0) return
            if (existing_idx > 0) then
                if (.not. rebind) then
                    found = .true.
                    return
                end if
            end if
            mangled = module_procedure_mangled( &
                trim(context%module_exports(export_idx)%module_name), &
                trim(var_name))
            call to_c_chars(mangled, c_name)
            global_id = lr_session_intern(context%session%handle, c_name)
            if (global_id < 0_c_int32_t) then
                error_msg = 'module variable global not found: '//trim(mangled)
                return
            end if
            if (value_kind == VALUE_DERIVED) then
                call bind_one_derived_module_variable(arena, var_idx, local_name, &
                                                      global_id, context, error_msg)
                if (len_trim(error_msg) > 0) return
                found = .true.
                return
            end if
            call use_import_symbol_slot(context, existing_idx, rebind, &
                                        sym_idx, skip)
            if (skip) then
                found = .true.
                return
            end if
            context%symbols(sym_idx)%name = local_name
            context%symbols(sym_idx)%value_kind = value_kind
            context%symbols(sym_idx)%has_address = .true.
            context%symbols(sym_idx)%is_reference = .true.
            context%symbols(sym_idx)%address%kind = LR_OP_KIND_GLOBAL
            context%symbols(sym_idx)%address%payload = int(global_id, c_int64_t)
            context%symbols(sym_idx)%address%typ = &
                lr_type_ptr_s(context%session%handle)
            context%symbols(sym_idx)%address%global_offset = 0_c_int64_t
            call record_use_rename_source(context, sym_idx, &
                trim(context%module_exports(export_idx)%module_name), &
                remote_name)
            found = .true.
            return
        end do
    end subroutine import_renamed_variable

    subroutine import_renamed_procedure(arena, module_name, remote_name, &
                                        local_name, context, found, error_msg)
        ! Alias an exported module procedure under a local name (#274). The
        ! procedure node is already in the function table from
        ! register_module_procedures; this adds a second entry sharing the same
        ! node index, value kind, and parameter count.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: remote_name
        character(len=*), intent(in) :: local_name
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: found
        character(len=:), allocatable, intent(out) :: error_msg

        found = .false.
        call set_empty(error_msg)
        if (.not. module_exports_procedure(arena, module_name, remote_name)) return
        if (contained_function_kind(context, local_name) /= 0) then
            error_msg = 'duplicate use rename local name: '//local_name
            return
        end if
        call register_procedure_alias(context, remote_name, local_name, error_msg)
        if (len_trim(error_msg) > 0) return
        found = .true.
    end subroutine import_renamed_procedure

    subroutine import_renamed_parameter(decl, local_name, value_kind, context, &
                                        error_msg)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: local_name
        integer, intent(in) :: value_kind
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int64_t) :: constant_value
        integer :: index

        if (value_kind /= VALUE_I32) then
            call unsupported_feature_error('renamed parameter', decl%line, &
                decl%column, 'direct LIRIC session only supports integer '// &
                'scalar parameters', error_msg)
            return
        end if
        if (find_symbol_compat(context, local_name) > 0) then
            error_msg = 'duplicate use rename local name: '//trim(local_name)
            return
        end if
        if (.not. decl%has_initializer .or. decl%initializer_index <= 0) then
            error_msg = 'renamed parameter has no initializer: '//trim(decl%var_name)
            return
        end if
        call eval_i32_constant(context%arena, decl%initializer_index, context, &
                               constant_value, error_msg)
        if (len_trim(error_msg) > 0) return
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(local_name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%value = i32_immediate(context%session, &
                                                     constant_value)
        context%symbols(index)%is_parameter = .true.
        context%symbols(index)%has_i32_constant = .true.
        context%symbols(index)%i32_constant = constant_value
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine import_renamed_parameter

    logical function use_only_wants(use_node, name)
        ! Whether a plain USE imports an exported name. A USE without only:
        ! imports everything; a USE with only: imports just the listed names.
        type(use_statement_node), intent(in) :: use_node
        character(len=*), intent(in) :: name
        integer :: i

        use_only_wants = .true.
        if (.not. use_node%has_only) return
        use_only_wants = .false.
        if (.not. allocated(use_node%only_list)) return
        do i = 1, size(use_node%only_list)
            if (same_name(use_node%only_list(i)%s, name)) then
                use_only_wants = .true.
                return
            end if
        end do
    end function use_only_wants

    subroutine resolve_fmod_import_name(use_node, remote_name, local_name, &
                                        imported)
        ! Resolve the local spelling of a .fmod export. A rename maps the
        ! remote export to its local name; an ONLY list admits unrenamed names
        ! only when they are explicitly listed. A renamed remote name is not
        ! also imported under its original spelling.
        type(use_statement_node), intent(in) :: use_node
        character(len=*), intent(in) :: remote_name
        character(len=:), allocatable, intent(out) :: local_name
        logical, intent(out) :: imported
        integer :: i

        local_name = trim(remote_name)
        imported = .true.
        if (allocated(use_node%rename_list)) then
            do i = 1, size(use_node%rename_list) - 1, 2
                if (.not. allocated(use_node%rename_list(i)%s)) cycle
                if (.not. allocated(use_node%rename_list(i + 1)%s)) cycle
                if (same_name(use_node%rename_list(i + 1)%s, remote_name)) then
                    local_name = trim(use_node%rename_list(i)%s)
                    return
                end if
            end do
        end if
        if (.not. use_node%has_only) return

        imported = .false.
        if (.not. allocated(use_node%only_list)) return
        do i = 1, size(use_node%only_list)
            if (.not. allocated(use_node%only_list(i)%s)) cycle
            if (same_name(use_node%only_list(i)%s, remote_name)) then
                imported = .true.
                return
            end if
        end do
    end subroutine resolve_fmod_import_name

    subroutine validate_fmod_use_names(use_node, info, error_msg)
        ! Validate every remote name mentioned by ONLY or a rename against the
        ! exports recorded in the .fmod. Without this check an unrecognised
        ! name would disappear during import and fail later as an unrelated
        ! undeclared identifier.
        type(use_statement_node), intent(in) :: use_node
        type(module_info_t), intent(in) :: info
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        character(len=:), allocatable :: name

        call set_empty(error_msg)
        if (use_node%has_only .and. allocated(use_node%only_list)) then
            do i = 1, size(use_node%only_list)
                if (.not. allocated(use_node%only_list(i)%s)) cycle
                name = trim(use_node%only_list(i)%s)
                if (.not. fmod_exports_name(info, name)) then
                    error_msg = 'use only: name is not exported by module '// &
                        trim(use_node%module_name)//': '//name
                    return
                end if
            end do
        end if
        if (.not. allocated(use_node%rename_list)) return
        do i = 2, size(use_node%rename_list), 2
            if (.not. allocated(use_node%rename_list(i)%s)) cycle
            name = trim(use_node%rename_list(i)%s)
            if (.not. fmod_exports_name(info, name)) then
                error_msg = 'use only: name is not exported by module '// &
                    trim(use_node%module_name)//': '//name
                return
            end if
        end do
    end subroutine validate_fmod_use_names

    logical function fmod_exports_name(info, name)
        type(module_info_t), intent(in) :: info
        character(len=*), intent(in) :: name
        integer :: i

        fmod_exports_name = .false.
        if (allocated(info%parameters)) then
            do i = 1, size(info%parameters)
                if (allocated(info%parameters(i)%name)) then
                    if (same_name(info%parameters(i)%name, name)) then
                        fmod_exports_name = .true.
                        return
                    end if
                end if
            end do
        end if
        if (allocated(info%derived_types)) then
            do i = 1, size(info%derived_types)
                if (allocated(info%derived_types(i)%name)) then
                    if (same_name(info%derived_types(i)%name, name)) then
                        fmod_exports_name = .true.
                        return
                    end if
                end if
            end do
        end if
        if (allocated(info%variables)) then
            do i = 1, size(info%variables)
                if (allocated(info%variables(i)%name)) then
                    if (same_name(info%variables(i)%name, name)) then
                        fmod_exports_name = .true.
                        return
                    end if
                end if
            end do
        end if
        if (allocated(info%procedures)) then
            do i = 1, size(info%procedures)
                if (allocated(info%procedures(i)%name)) then
                    if (same_name(info%procedures(i)%name, name)) then
                        fmod_exports_name = .true.
                        return
                    end if
                end if
            end do
        end if
        if (allocated(info%generics)) then
            do i = 1, size(info%generics)
                if (allocated(info%generics(i)%name)) then
                    if (same_name(info%generics(i)%name, name)) then
                        fmod_exports_name = .true.
                        return
                    end if
                end if
            end do
        end if
    end function fmod_exports_name

    subroutine validate_use_only_names(arena, export_idx, use_node, context, &
                                       error_msg)
        ! Every name in only: must be a symbol the module actually exports.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (.not. use_node%has_only) return
        if (.not. allocated(use_node%only_list)) return
        do i = 1, size(use_node%only_list)
            if (.not. module_exports_name(arena, export_idx, context, &
                                          use_node%only_list(i)%s)) then
                error_msg = 'use only: name is not exported by module '// &
                    trim(use_node%module_name)//': '// &
                    trim(use_node%only_list(i)%s)
                return
            end if
        end do
    end subroutine validate_use_only_names

    logical function module_exports_name(arena, export_idx, context, name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i
        character(len=:), allocatable :: derived_name, decl_name
        character(len=:), allocatable :: error_msg

        module_exports_name = .false.
        do i = 1, context%module_exports(export_idx)%derived_type_count
            if (.not. is_derived_type_node(arena, &
                    context%module_exports(export_idx)%derived_type_indices(i))) cycle
            call get_derived_type_name(arena, &
                                       context%module_exports(export_idx)% &
                                           derived_type_indices(i), &
                                       derived_name, error_msg)
            if (len_trim(error_msg) > 0) cycle
            if (same_name(derived_name, name)) then
                module_exports_name = .true.
                return
            end if
        end do
        do i = 1, context%module_exports(export_idx)%parameter_count
            if (.not. is_declaration_node(arena, &
                    context%module_exports(export_idx)%parameter_indices(i))) cycle
            call get_declaration_var_name(arena, &
                                          context%module_exports(export_idx)% &
                                              parameter_indices(i), &
                                          decl_name, error_msg)
            if (len_trim(error_msg) > 0) cycle
            if (same_name(decl_name, name)) then
                module_exports_name = .true.
                return
            end if
        end do
        do i = 1, context%module_exports(export_idx)%variable_count
            if (.not. is_declaration_node(arena, &
                    context%module_exports(export_idx)%variable_indices(i))) cycle
            call get_declaration_var_name(arena, &
                                          context%module_exports(export_idx)% &
                                              variable_indices(i), &
                                          decl_name, error_msg)
            if (len_trim(error_msg) > 0) cycle
            if (same_name(decl_name, name)) then
                module_exports_name = .true.
                return
            end if
        end do
        ! A generic interface declared by the module is an export too: it is
        ! registered under its generic name when the module is lowered in this
        ! unit, so ONLY may name it (#371).
        if (find_generic(context, name) > 0) then
            module_exports_name = .true.
            return
        end if
        ! Operator interfaces are kept in the operator table rather than the
        ! named-generic table. Resolve the export against the defining module
        ! so `use m, only: operator(/)` is validated by the same ownership rule
        ! as a named generic.
        if (module_exports_operator(arena, &
                context%module_exports(export_idx)%module_name, name)) then
            module_exports_name = .true.
            return
        end if
        ! Module procedures are not recorded in module_exports_t (only data
        ! symbols are), so resolve a procedure name against the module body
        ! itself (#263). A USE-only of a contained function or subroutine is a
        ! valid export.
        if (module_exports_procedure(arena, &
                context%module_exports(export_idx)%module_name, name)) then
            module_exports_name = .true.
            return
        end if
    end function module_exports_name

    logical function module_exports_operator(arena, module_name, name)
        ! Whether a module declares the operator interface named by a USE
        ! ONLY item such as `operator(/)`. Operator interfaces do not have a
        ! named-generic entry, so inspect the module declaration list directly
        ! instead of treating any globally registered operator as exported.
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: requested, op_token
        integer :: n, d, open_pos

        module_exports_operator = .false.
        requested = trim(lowercase_text(name))
        if (len_trim(requested) < 11) return
        if (requested(1:9) /= 'operator(') return
        if (requested(len_trim(requested):len_trim(requested)) /= ')') return
        open_pos = 9
        op_token = requested(open_pos + 1:len_trim(requested) - 1)
        if (len_trim(op_token) == 0) return

        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (mod_node => arena%entries(n)%node)
            type is (module_node)
                if (.not. allocated(mod_node%name)) cycle
                if (.not. same_name(mod_node%name, module_name)) cycle
                if (module_symbol_is_private(arena, mod_node, requested)) return
                if (.not. allocated(mod_node%declaration_indices)) return
                do d = 1, size(mod_node%declaration_indices)
                    if (.not. node_exists(arena, &
                            mod_node%declaration_indices(d))) cycle
                    select type (block => &
                            arena%entries(mod_node%declaration_indices(d))%node)
                    type is (interface_block_node)
                        if (.not. allocated(block%kind)) cycle
                        if (trim(lowercase_text(block%kind)) /= 'operator') cycle
                        if (.not. allocated(block%operator)) cycle
                        if (same_name(block%operator, op_token)) then
                            module_exports_operator = .true.
                            return
                        end if
                    end select
                end do
                return
            end select
        end do
    end function module_exports_operator

    logical function module_exports_procedure(arena, module_name, name)
        ! Whether the named module contains a public function or subroutine
        ! called `name`. Used to validate USE-only of a module procedure (#263).
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: module_lowered
        integer :: n

        module_exports_procedure = .false.
        module_lowered = lowercase_text(module_name)
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (node => arena%entries(n)%node)
            type is (module_node)
                if (.not. allocated(node%name)) cycle
                if (lowercase_text(node%name) /= module_lowered) cycle
                if (allocated(node%procedure_indices)) then
                    if (module_procedure_named(arena, node, name)) then
                        module_exports_procedure = .true.
                        return
                    end if
                end if
                ! A module procedure declared only by its interface body
                ! (module function/subroutine) is a public export; its body is
                ! defined in a submodule (#292).
                if (module_declares_interface_procedure(arena, node, name)) &
                    module_exports_procedure = .true.
                return
            end select
        end do
    end function module_exports_procedure

    logical function module_declares_interface_procedure(arena, mod_node, name)
        ! Whether the module declares `name` as a module procedure through an
        ! interface body (module function/subroutine), regardless of where the
        ! implementation lives.
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        character(len=*), intent(in) :: name
        integer :: d, p

        module_declares_interface_procedure = .false.
        if (.not. allocated(mod_node%declaration_indices)) return
        do d = 1, size(mod_node%declaration_indices)
            if (.not. node_exists(arena, mod_node%declaration_indices(d))) cycle
            select type (iblock => &
                arena%entries(mod_node%declaration_indices(d))%node)
            type is (interface_block_node)
                if (iblock%is_abstract) cycle
                if (.not. allocated(iblock%procedure_indices)) cycle
                do p = 1, size(iblock%procedure_indices)
                    if (procedure_node_named(arena, iblock%procedure_indices(p), &
                                             name)) then
                        module_declares_interface_procedure = .true.
                        return
                    end if
                end do
            end select
        end do
    end function module_declares_interface_procedure

    logical function module_procedure_named(arena, mod_node, name)
        ! Whether `mod_node` holds a non-private function/subroutine `name`.
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: proc_name
        integer :: p, idx

        module_procedure_named = .false.
        do p = 1, size(mod_node%procedure_indices)
            idx = mod_node%procedure_indices(p)
            if (.not. node_exists(arena, idx)) cycle
            proc_name = ''
            select type (proc => arena%entries(idx)%node)
            type is (function_def_node)
                if (allocated(proc%name)) proc_name = proc%name
            type is (subroutine_def_node)
                if (allocated(proc%name)) proc_name = proc%name
            end select
            if (len_trim(proc_name) == 0) cycle
            if (.not. same_name(proc_name, name)) cycle
            if (module_symbol_is_private(arena, mod_node, proc_name)) cycle
            module_procedure_named = .true.
            return
        end do
    end function module_procedure_named

    subroutine import_module_constants(arena, export_idx, use_node, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: param_idx
        integer :: symbol_idx
        integer :: value_kind
        integer(c_int64_t) :: constant_value
        integer :: init_index
        integer :: line
        integer :: column
        character(len=:), allocatable :: decl_name
        character(len=:), allocatable :: decl_type_name

        call set_empty(error_msg)
        do i = 1, context%module_exports(export_idx)%parameter_count
            param_idx = context%module_exports(export_idx)%parameter_indices(i)
            if (.not. node_exists(arena, param_idx)) cycle
            if (.not. is_declaration_node(arena, param_idx)) cycle
            call get_declaration_var_name(arena, param_idx, decl_name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. use_only_wants(use_node, trim(decl_name))) cycle
            if (find_symbol_compat(context, decl_name) > 0) cycle
            call get_declaration_type_name(arena, param_idx, decl_type_name, &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
            line = get_node_line(arena, param_idx)
            column = get_node_column(arena, param_idx)
            call type_name_value_kind(decl_type_name, line, column, value_kind, &
                                      error_msg, context, param_idx)
            if (len_trim(error_msg) > 0) return
            if (.not. get_declaration_has_initializer(arena, param_idx)) then
                error_msg = 'parameter requires a compile-time initializer: '// &
                            trim(decl_name)
                return
            end if
            init_index = get_declaration_initializer_index(arena, param_idx)
            if (init_index <= 0) then
                error_msg = 'parameter requires a compile-time initializer: '// &
                            trim(decl_name)
                return
            end if
            if (value_kind /= VALUE_I32) then
                select type (decl_node => arena%entries(param_idx)%node)
                type is (declaration_node)
                    call lower_nonint_constant_declaration(decl_node, context, &
                                                           value_kind, error_msg)
                class default
                    error_msg = 'module parameter declaration is not a declaration: '// &
                                trim(decl_name)
                end select
                if (len_trim(error_msg) > 0) return
                cycle
            end if
            call eval_i32_constant(context%arena, init_index, context, &
                                   constant_value, error_msg)
            if (len_trim(error_msg) > 0) return
            call grow_symbols(context)
            symbol_idx = context%symbol_count + 1
            context%symbols(symbol_idx)%name = trim(decl_name)
            context%symbols(symbol_idx)%value_kind = VALUE_I32
            context%symbols(symbol_idx)%value = i32_immediate(context%session, &
                                                              constant_value)
            context%symbols(symbol_idx)%is_parameter = .true.
            context%symbols(symbol_idx)%has_i32_constant = .true.
            context%symbols(symbol_idx)%i32_constant = constant_value
            context%symbol_count = symbol_idx
        end do
    end subroutine import_module_constants

    subroutine import_module_variables(arena, export_idx, use_node, context, &
                                       error_msg)
        ! Import module-level variables from an in-unit module into the calling
        ! program's symbol table (#263). Each supported scalar variable binds to
        ! the shared global as an addressed reference, honouring use, only:.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, var_idx, value_kind, status

        call set_empty(error_msg)
        do i = 1, context%module_exports(export_idx)%variable_count
            var_idx = context%module_exports(export_idx)%variable_indices(i)
            if (.not. node_exists(arena, var_idx)) cycle
            call module_variable_kind(arena, var_idx, value_kind, status, &
                                      context, error_msg)
            ! An unsupported module variable already failed in
            ! lower_module_variables; the using unit just skips it here.
            if (status /= MODVAR_OK) then
                call set_empty(error_msg)
                cycle
            end if
            block
                character(len=64), allocatable :: names(:)
                integer :: nm
                call module_declaration_var_names(arena, var_idx, names)
                do nm = 1, size(names)
                    if (.not. use_only_wants(use_node, trim(names(nm)))) cycle
                    call bind_one_module_variable(arena, var_idx, value_kind, &
                        trim(names(nm)), &
                        trim(context%module_exports(export_idx)%module_name), &
                        context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end block
        end do
    end subroutine import_module_variables

    subroutine add_derived_type_from_arena(arena, type_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index_new

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

      call grow_derived_types(context)

        type_index_new = context%derived_type_count + 1

        ! Use the existing lower_derived_type_definition pattern but with arena index
        call import_derived_type_from_arena(arena, type_index, context, &
            type_index_new, error_msg)
        if (len_trim(error_msg) > 0) return

        context%derived_type_count = type_index_new
    end subroutine add_derived_type_from_arena

subroutine import_derived_type_from_arena(arena, type_index, context, &
        type_index_new, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index_new
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: type_name

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        context%derived_types(type_index_new)%component_count = 0

        call get_derived_type_name(arena, type_index, type_name, error_msg)
        if (len_trim(error_msg) > 0) return
        context%derived_types(type_index_new)%name = trim(type_name)

        call collect_derived_components_via_arena(arena, type_index, context, &
            type_index_new, error_msg)
    end subroutine import_derived_type_from_arena

    subroutine collect_derived_components_via_arena(arena, type_index, &
        context, type_index_new, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index_new
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        ! The key insight: we need to access derived_type_node%component_indices
        ! which is an allocatable integer array. When we do select type on
        ! arena%entries(type_index)%node (which is class(ast_node), allocatable),
        ! the allocatable arrays in the derived type should be accessible.
        !
        ! The issue we've been seeing is that component_indices appears unallocated.
        ! This might be because the arena stores nodes differently than expected.
        !
        ! Let's try using context%arena instead of arena to see if that helps.
        call collect_derived_components_from_context_arena(arena, type_index, context, &
            type_index_new, error_msg)
    end subroutine collect_derived_components_via_arena

    subroutine collect_derived_components_from_context_arena(arena, type_index, &
        context, type_index_new, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: type_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index_new
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: parent_line
        integer :: parent_column
        integer, allocatable :: component_indices(:)

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        if (.not. is_derived_type_node(arena, type_index)) then
            error_msg = 'node at type_index is not a derived_type_node'
            return
        end if
        call get_derived_type_components(arena, type_index, component_indices)
        parent_line = get_node_line(arena, type_index)
        parent_column = get_node_column(arena, type_index)
        if (allocated(component_indices)) then
            do i = 1, size(component_indices)
                call collect_component_declaration(component_indices(i), &
                                                   parent_line, parent_column, &
                                                   context, type_index_new, &
                                                   error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
    end subroutine collect_derived_components_from_context_arena

    subroutine lower_program_body(arena, program, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(program_node), intent(in) :: program
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        value = i32_immediate(context%session, 0_c_int64_t)
        context%current_block_terminated = .false.

        if (.not. allocated(program%body_indices)) return
        do i = 1, size(program%body_indices)
            if (is_internal_procedure_entry(arena, program%body_indices(i))) cycle
            call lower_statement(arena, program%body_indices(i), context, &
                                 value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%current_block_terminated) return
        end do
    end subroutine lower_program_body

    subroutine find_module_in_arena(arena, module_lowered, module_index)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: module_lowered
        integer, intent(out) :: module_index
        integer :: i

        module_index = 0
        if (len_trim(module_lowered) == 0) return

        do i = 1, arena%size
            if (.not. allocated(arena%entries(i)%node)) cycle
            select type (node => arena%entries(i)%node)
            type is (module_node)
                if (.not. allocated(node%name)) cycle
                if (lowercase_text(node%name) == module_lowered) then
                    module_index = i
                    return
                end if
            end select
        end do
    end subroutine find_module_in_arena
