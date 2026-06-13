    subroutine collect_program_parameters(arena, root_index, context, error_msg)
        ! Pre-collect program-scope integer scalar parameters so derived-type
        ! array component bounds (integer :: a(n)) can resolve them. Parameters
        ! that are not compile-time integer scalars are skipped here and handled
        ! (or rejected) when the body declaration is lowered.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, value_kind
        character(len=:), allocatable :: kind_err, decl_err

        call set_empty(error_msg)
        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (declaration_node)
                    if (.not. node%is_parameter) cycle
                    if (node%is_array .or. node%is_multi_declaration) cycle
                    if (.not. allocated(node%var_name)) cycle
                    call declaration_value_kind(node, value_kind, kind_err)
                    if (len_trim(kind_err) > 0) cycle
                    if (value_kind /= VALUE_I32) cycle
                    if (.not. node%has_initializer) cycle
                    ! Best effort: ignore errors here; body lowering reports them.
                    call lower_constant_declaration(node, context, VALUE_I32, &
                                                    decl_err)
                end select
            end do
        end select
    end subroutine collect_program_parameters

    subroutine collect_derived_type_definitions(arena, root_index, context, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        do i = 1, size(context%derived_types)
            context%derived_types(i)%name = ''
            context%derived_types(i)%component_count = 0
            if (allocated(context%derived_types(i)%component_names)) then
                deallocate(context%derived_types(i)%component_names)
            end if
        end do
        context%derived_type_count = 0
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (derived_type_node)
                    call lower_derived_type_definition(node, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call emit_type_info_constant(context, &
                                                 context%derived_type_count, &
                                                 error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end do
        end select
    end subroutine collect_derived_type_definitions

    subroutine emit_type_info_constant(context, type_index, error_msg)
        ! Emit the per-derived-type compile-time type_info_t constant (#140).
        ! The id is the dense per-unit type index; size is the storage in bytes
        ! (each component occupies a 4-byte i32 slot in the MVP layout).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg

        if (type_index <= 0) then
            call set_empty(error_msg)
            return
        end if
        call create_type_info_global(context%session, &
            '__ffc_type_info_'//trim(context%derived_types(type_index)%name), &
            int(type_index, c_int64_t), &
            int(total_component_slots(context, type_index) * 4, c_int64_t), &
            error_msg)
    end subroutine emit_type_info_constant

    subroutine collect_module_exports(arena, root_index, context, &
                                      error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: j
        integer :: export_idx
        integer :: type_idx
        integer :: comp_idx

        do i = 1, size(context%module_exports)
            context%module_exports(i)%module_name = ''
            if (allocated(context%module_exports(i)%derived_type_indices)) then
                deallocate(context%module_exports(i)%derived_type_indices)
            end if
            context%module_exports(i)%derived_type_count = 0
            if (allocated(context%module_exports(i)%parameter_indices)) then
                deallocate(context%module_exports(i)%parameter_indices)
            end if
            context%module_exports(i)%parameter_count = 0
            if (allocated(context%module_exports(i)%variable_indices)) then
                deallocate(context%module_exports(i)%variable_indices)
            end if
            context%module_exports(i)%variable_count = 0
        end do
        context%module_export_count = 0
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (mod_node => arena%entries(program%body_indices(i))%node)
                type is (module_node)
                    call collect_one_module_exports(arena, mod_node, context)
                end select
            end do
        type is (module_node)
            ! A module-only compilation unit (ffc -c <module>.f90).
            call collect_one_module_exports(arena, program, context)
        end select
    end subroutine collect_module_exports

    subroutine collect_one_module_exports(arena, mod_node, context)
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        type(lowering_context_t), intent(inout) :: context
        integer :: j
        integer :: export_idx

        if (.not. allocated(mod_node%name)) return
        if (.not. allocated(mod_node%declaration_indices)) return
        call grow_module_exports(context)

        export_idx = context%module_export_count + 1
        context%module_exports(export_idx)%module_name = trim(mod_node%name)
        context%module_exports(export_idx)%derived_type_count = 0
        allocate(context%module_exports(export_idx)%derived_type_indices(8))
        context%module_exports(export_idx)%parameter_count = 0
        allocate(context%module_exports(export_idx)%parameter_indices(8))
        context%module_exports(export_idx)%variable_count = 0
        allocate(context%module_exports(export_idx)%variable_indices(8))

        do j = 1, size(mod_node%declaration_indices)
            select type (decl => &
                arena%entries(mod_node%declaration_indices(j))%node)
            type is (derived_type_node)
                if (.not. allocated(decl%name)) cycle
                ! A private symbol is not exported through use (#149).
                if (module_symbol_is_private(arena, mod_node, trim(decl%name))) cycle
                call grow_module_export_derived_indices(context, export_idx)
                context%module_exports(export_idx)%derived_type_indices( &
                    context%module_exports(export_idx)%derived_type_count + 1) = &
                    mod_node%declaration_indices(j)
                context%module_exports(export_idx)%derived_type_count = &
                    context%module_exports(export_idx)%derived_type_count + 1
            type is (declaration_node)
                if (.not. allocated(decl%var_name)) cycle
                if (decl%is_parameter) then
                    if (module_symbol_is_private(arena, mod_node, trim(decl%var_name))) &
                        cycle
                    call grow_module_export_parameter_indices(context, export_idx)
                    context%module_exports(export_idx)%parameter_indices( &
                        context%module_exports(export_idx)%parameter_count + 1) = &
                        mod_node%declaration_indices(j)
                    context%module_exports(export_idx)%parameter_count = &
                        context%module_exports(export_idx)%parameter_count + 1
                else if (.not. decl%is_array) then
                    ! Non-parameter scalar: export as module variable (#249 B7a).
                    if (module_symbol_is_private(arena, mod_node, trim(decl%var_name))) &
                        cycle
                    call grow_module_export_variable_indices(context, export_idx)
                    context%module_exports(export_idx)%variable_indices( &
                        context%module_exports(export_idx)%variable_count + 1) = &
                        mod_node%declaration_indices(j)
                    context%module_exports(export_idx)%variable_count = &
                        context%module_exports(export_idx)%variable_count + 1
                end if
            end select
        end do

        context%module_export_count = export_idx
    end subroutine collect_one_module_exports

    logical function module_symbol_is_private(arena, mod_node, name)
        ! Effective accessibility of a module symbol (#149): a name listed in
        ! `private :: ...` is private; one listed in `public :: ...` is public;
        ! otherwise the module default applies (public unless a bare `private`
        ! statement flipped it). The attribute form (`integer, private ::`) is
        ! also honoured when FortFront records it on the declaration.
        type(ast_arena_t), intent(in) :: arena
        type(module_node), intent(in) :: mod_node
        character(len=*), intent(in) :: name
        logical :: default_private
        integer :: j, k

        module_symbol_is_private = .false.
        default_private = .false.
        ! Pass 1: explicit per-name lists win regardless of position.
        do j = 1, size(mod_node%declaration_indices)
            if (.not. node_exists(arena, mod_node%declaration_indices(j))) cycle
            select type (vis => arena%entries(mod_node%declaration_indices(j))%node)
            type is (visibility_statement_node)
                if (vis%has_list .and. allocated(vis%names)) then
                    do k = 1, size(vis%names)
                        if (allocated(vis%names(k)%s)) then
                            if (trim(lowercase_text(vis%names(k)%s)) == &
                                trim(lowercase_text(name))) then
                                module_symbol_is_private = vis%is_private
                                return
                            end if
                        end if
                    end do
                else if (vis%is_private) then
                    default_private = .true.
                else
                    default_private = .false.
                end if
            type is (declaration_node)
                ! Attribute form: integer, private, parameter :: name.
                if (decl_names_match(vis, name)) then
                    if (allocated(vis%accessibility)) then
                        if (trim(lowercase_text(vis%accessibility)) == 'private') &
                            then
                            module_symbol_is_private = .true.
                            return
                        end if
                    end if
                end if
            end select
        end do
        module_symbol_is_private = default_private
    end function module_symbol_is_private

    logical function decl_names_match(decl, name)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: name
        integer :: k

        decl_names_match = .false.
        if (allocated(decl%var_name)) then
            if (trim(decl%var_name) == trim(name)) decl_names_match = .true.
        end if
        if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
            do k = 1, size(decl%var_names)
                if (trim(decl%var_names(k)) == trim(name)) decl_names_match = .true.
            end do
        end if
    end function decl_names_match

    integer function find_module_export_index(context, module_name) &
        result(idx)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: module_name
        integer :: i

        idx = 0
        do i = 1, context%module_export_count
            if (same_name(context%module_exports(i)%module_name, module_name)) then
                idx = i
                return
            end if
        end do
    end function find_module_export_index

    subroutine lower_derived_type_definition(node, context, error_msg)
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: type_index

        if (.not. allocated(node%name)) then
            error_msg = 'derived type definition did not expose a name'
            return
        end if

        type_index = find_derived_type(context, node%name)
        if (type_index > 0) then
            call set_empty(error_msg)
            return
        end if

        if (node%has_parameters .or. allocated(node%param_indices)) then
            call unsupported_feature_error('derived type parameters', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'non-parameterized derived types', &
                                           error_msg)
            return
        end if
        if (.not. allocated(node%component_indices) .and. &
            .not. allocated(node%extends_parent)) then
            call unsupported_feature_error('derived type definition', &
                                           node%line, node%column, &
                                           'derived types need at least one '// &
                                           'integer component', error_msg)
            return
        end if

        call grow_derived_types(context)

        type_index = context%derived_type_count + 1
        context%derived_types(type_index)%name = trim(node%name)
        context%derived_types(type_index)%binding_count = 0
        if (allocated(node%extends_parent)) then
            call inherit_parent_members(node, context, type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (allocated(node%component_indices)) then
            call collect_derived_components(node, context, type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (context%derived_types(type_index)%component_count <= 0) then
            call unsupported_feature_error('derived type definition', &
                                           node%line, node%column, &
                                           'derived types need at least one '// &
                                           'integer component', error_msg)
            return
        end if
        call collect_type_bindings(node, context, type_index, error_msg)
        if (len_trim(error_msg) > 0) return

        context%derived_type_count = type_index
        call set_empty(error_msg)
    end subroutine lower_derived_type_definition

    subroutine collect_type_bindings(node, context, type_index, error_msg)
        ! Record type-bound procedure bindings (procedure :: m => fn) as
        ! compile-time aliases. When inherited bindings are already present
        ! (from inherit_parent_bindings), child declarations either override
        ! an inherited name or append a new one (#248 B6b).
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, b, inherited, new_total, slot
        character(len=64) :: method_name_buf, target_name_buf, pass_name_buf
        logical :: overrides
        character(len=64), allocatable :: old_methods(:), old_targets(:), old_passes(:)

        call set_empty(error_msg)
        if (.not. allocated(node%binding_indices)) return

        inherited = context%derived_types(type_index)%binding_count
        if (inherited == 0) then
            ! No inherited bindings: allocate fresh arrays sized to own bindings.
            allocate(context%derived_types(type_index)% &
                binding_method_names(size(node%binding_indices)))
            allocate(context%derived_types(type_index)% &
                binding_target_names(size(node%binding_indices)))
            allocate(context%derived_types(type_index)% &
                binding_pass_names(size(node%binding_indices)))
            context%derived_types(type_index)%binding_pass_names = ''
        else
            ! Inherited bindings exist: grow arrays to hold inherited + new own.
            new_total = inherited + size(node%binding_indices)
            old_methods = context%derived_types(type_index)%binding_method_names
            old_targets = context%derived_types(type_index)%binding_target_names
            old_passes  = context%derived_types(type_index)%binding_pass_names
            deallocate(context%derived_types(type_index)%binding_method_names)
            deallocate(context%derived_types(type_index)%binding_target_names)
            deallocate(context%derived_types(type_index)%binding_pass_names)
            allocate(context%derived_types(type_index)%binding_method_names(new_total))
            allocate(context%derived_types(type_index)%binding_target_names(new_total))
            allocate(context%derived_types(type_index)%binding_pass_names(new_total))
            context%derived_types(type_index)%binding_method_names = ''
            context%derived_types(type_index)%binding_target_names = ''
            context%derived_types(type_index)%binding_pass_names = ''
            context%derived_types(type_index)%binding_method_names(1:inherited) = &
                old_methods(1:inherited)
            context%derived_types(type_index)%binding_target_names(1:inherited) = &
                old_targets(1:inherited)
            context%derived_types(type_index)%binding_pass_names(1:inherited) = &
                old_passes(1:inherited)
        end if

        do i = 1, size(node%binding_indices)
            if (.not. node_exists(context%arena, node%binding_indices(i))) cycle
            select type (binding => &
                context%arena%entries(node%binding_indices(i))%node)
            type is (type_binding_node)
                if (binding%is_generic .or. binding%is_final .or. &
                    binding%is_deferred) then
                    call unsupported_feature_error('type-bound procedure', &
                        node%line, node%column, &
                        'direct LIRIC session only supports plain '// &
                        '(procedure :: m => fn) bindings, not generic, '// &
                        'final, or deferred', error_msg)
                    return
                end if
                if (.not. allocated(binding%binding_name)) cycle
                method_name_buf = trim(binding%binding_name)
                if (allocated(binding%implementation) .and. &
                    len_trim(binding%implementation) > 0) then
                    target_name_buf = trim(binding%implementation)
                else
                    target_name_buf = trim(binding%binding_name)
                end if
                pass_name_buf = ''
                if (allocated(binding%pass_name)) then
                    if (len_trim(binding%pass_name) > 0) &
                        pass_name_buf = trim(binding%pass_name)
                end if
                ! Override inherited binding with same method name, or append.
                overrides = .false.
                do b = 1, inherited
                    if (trim(context%derived_types(type_index)% &
                              binding_method_names(b)) == trim(method_name_buf)) then
                        context%derived_types(type_index)%binding_target_names(b) = &
                            target_name_buf
                        context%derived_types(type_index)%binding_pass_names(b) = &
                            pass_name_buf
                        overrides = .true.
                        exit
                    end if
                end do
                if (.not. overrides) then
                    slot = context%derived_types(type_index)%binding_count + 1
                    context%derived_types(type_index)%binding_method_names(slot) = &
                        method_name_buf
                    context%derived_types(type_index)%binding_target_names(slot) = &
                        target_name_buf
                    context%derived_types(type_index)%binding_pass_names(slot) = &
                        pass_name_buf
                    context%derived_types(type_index)%binding_count = slot
                end if
            end select
        end do
    end subroutine collect_type_bindings

    subroutine collect_derived_components(node, context, type_index, error_msg)
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        ! Own components append after any inherited (parent-first) components,
        ! so the child layout is parent components then own (#248 B6a).
        do i = 1, size(node%component_indices)
            call collect_component_declaration( &
                node%component_indices(i), node%line, node%column, context, &
                type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine collect_derived_components

    subroutine inherit_parent_members(node, context, type_index, error_msg)
        ! Copy the parent type's components (parent-first layout) and its
        ! static type-bound bindings into the extending child (#248 B6a). The
        ! parent must already be collected; types are collected in source order
        ! and Fortran requires the parent to be defined before it is extended.
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: parent_index, c

        call set_empty(error_msg)
        parent_index = find_derived_type(context, node%extends_parent)
        if (parent_index <= 0) then
            call unsupported_feature_error('derived type inheritance', &
                node%line, node%column, 'parent type "'// &
                trim(node%extends_parent)//'" must be defined before the type '// &
                'that extends it', error_msg)
            return
        end if

        associate (parent => context%derived_types(parent_index))
            do c = 1, parent%component_count
                call add_derived_component(context, type_index, &
                    parent%component_names(c), node%line, node%column, &
                    parent%component_has_default(c), &
                    parent%component_default_value(c), &
                    parent%component_array_size(c), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end associate
        call inherit_parent_bindings(context, type_index, parent_index)
    end subroutine inherit_parent_members

    subroutine inherit_parent_bindings(context, type_index, parent_index)
        ! Copy the parent's static type-bound bindings into the child so a
        ! parent method is callable on a child instance (static dispatch;
        ! overriding and dynamic dispatch are later #248 slices).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        integer, intent(in) :: parent_index
        integer :: pc, slot

        pc = context%derived_types(parent_index)%binding_count
        if (pc <= 0) return
        if (.not. allocated(context%derived_types(parent_index)% &
                            binding_method_names)) return
        allocate(context%derived_types(type_index)%binding_method_names(pc))
        allocate(context%derived_types(type_index)%binding_target_names(pc))
        allocate(context%derived_types(type_index)%binding_pass_names(pc))
        do slot = 1, pc
            context%derived_types(type_index)%binding_method_names(slot) = &
                context%derived_types(parent_index)%binding_method_names(slot)
            context%derived_types(type_index)%binding_target_names(slot) = &
                context%derived_types(parent_index)%binding_target_names(slot)
            context%derived_types(type_index)%binding_pass_names(slot) = &
                context%derived_types(parent_index)%binding_pass_names(slot)
        end do
        context%derived_types(type_index)%binding_count = pc
    end subroutine inherit_parent_bindings

subroutine collect_component_declaration(component_index, parent_line, &
                                              parent_column, context, &
                                              type_index, error_msg)
        integer, intent(in) :: component_index
        integer, intent(in) :: parent_line
        integer, intent(in) :: parent_column
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        logical :: has_default
        integer(c_int64_t) :: default_value
        integer :: array_size

        if (.not. node_exists(context%arena, component_index)) then
            error_msg = 'derived type component index does not reference an AST node'
            return
        end if

        select type (component => context%arena%entries(component_index)%node)
        type is (declaration_node)
            call validate_component_declaration(component, parent_line, &
                                                parent_column, context, &
                                                error_msg)
            if (len_trim(error_msg) > 0) return
            ! A component default initialiser must be a compile-time integer.
            has_default = .false.
            default_value = 0_c_int64_t
            array_size = 1
            if (component%is_array) then
                call component_array_count(component, context, array_size, &
                                           error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (component%has_initializer .and. component%initializer_index > 0) then
                if (component%is_array) then
                    call unsupported_feature_error( &
                        'derived type array component initializer', &
                        component%line, component%column, 'array components '// &
                        'cannot carry a default initialiser', error_msg)
                    return
                end if
                call parse_i32_constant(context%arena, component%initializer_index, &
                                        default_value, 'component default', &
                                        error_msg)
                if (len_trim(error_msg) > 0) then
                    call unsupported_feature_error( &
                        'derived type component initializer', component%line, &
                        component%column, 'only a compile-time integer default '// &
                        'is supported', error_msg)
                    return
                end if
                has_default = .true.
            end if
            if (component%is_multi_declaration .and. &
                allocated(component%var_names)) then
                do i = 1, size(component%var_names)
                    call add_derived_component(context, type_index, &
                                               component%var_names(i), &
                                               component%line, component%column, &
                                               has_default, default_value, &
                                               array_size, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(component%var_name)) then
                call add_derived_component(context, type_index, component%var_name, &
                                           component%line, component%column, &
                                           has_default, default_value, array_size, &
                                           error_msg)
            else
                error_msg = 'derived type component declaration did not expose a name'
                return
            end if
        class default
            call unsupported_feature_error('derived type component', &
                                           parent_line, parent_column, &
                                           'direct LIRIC session only supports '// &
                                           'integer component declarations', &
                                           error_msg)
            return
        end select
        call set_empty(error_msg)
    end subroutine collect_component_declaration

    subroutine validate_component_declaration(component, parent_line, &
                                              parent_column, context, &
                                              error_msg)
        type(declaration_node), intent(in) :: component
        integer, intent(in) :: parent_line
        integer, intent(in) :: parent_column
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: nested_index
        integer :: comp_kind

        if (.not. allocated(component%type_name)) then
            call set_empty(error_msg)
            return
        end if

        nested_index = declaration_derived_type_index(context, component)
        if (nested_index > 0) then
            call unsupported_feature_error('nested derived type component', &
                                           component%line, component%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar integer components', error_msg)
            return
        end if
        ! Accept any component that resolves to an integer-sized slot, including
        ! interop kinds (integer(c_int32_t)) and the opaque c_ptr. Other kinds
        ! (real, character, deferred-length) are rejected: the MVP derived-type
        ! layout stores only i32-sized components.
        call type_name_value_kind(component%type_name, component%line, &
                                  component%column, comp_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (comp_kind /= VALUE_I32 .and. comp_kind /= VALUE_C_PTR) then
            call unsupported_feature_error('derived type component', &
                                           component%line, component%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar integer (or c_ptr) components', &
                                           error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine validate_component_declaration

    subroutine component_array_count(component, context, array_size, error_msg)
        ! Compile-time element count of an array component (literal or
        ! parameter bound). Only a default lower bound of 1 is assumed.
        type(declaration_node), intent(in) :: component
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: array_lower_bound

        call get_array_bounds(component, context, array_lower_bound, array_size, &
                              error_msg)
        if (len_trim(error_msg) > 0) return
        if (array_size <= 0) then
            call unsupported_feature_error('derived type array component', &
                component%line, component%column, 'array components need a '// &
                'positive compile-time size', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine component_array_count

    subroutine add_derived_component(context, type_index, name, line, column, &
                                     has_default, default_value, array_size, &
                                     error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: name
        integer, intent(in) :: line, column
        logical, intent(in) :: has_default
        integer(c_int64_t), intent(in) :: default_value
        integer, intent(in) :: array_size
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: component_index

        if (find_derived_component(context, type_index, name) > 0) then
            error_msg = 'duplicate derived type component: '//trim(name)
            return
        end if

        call grow_derived_type_components(context, type_index)

        component_index = context%derived_types(type_index)%component_count + 1
        context%derived_types(type_index)%component_names(component_index) = &
            trim(name)
        context%derived_types(type_index)%component_has_default(component_index) = &
            has_default
        context%derived_types(type_index)%component_default_value(component_index) = &
            default_value
        context%derived_types(type_index)%component_array_size(component_index) = &
            array_size
        context%derived_types(type_index)%component_count = component_index
        call set_empty(error_msg)
    end subroutine add_derived_component

    subroutine grow_derived_type_components(context, type_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=64), allocatable :: old_names(:)
        integer :: new_size, old_size

        old_size = context%derived_types(type_index)%component_count
        if (old_size > 0 .and. allocated(context%derived_types(type_index)%component_names)) then
            if (old_size < size(context%derived_types(type_index)%component_names)) return
        else
            allocate(context%derived_types(type_index)%component_names(8))
            allocate(context%derived_types(type_index)%component_has_default(8))
            allocate(context%derived_types(type_index)%component_default_value(8))
            allocate(context%derived_types(type_index)%component_array_size(8))
            context%derived_types(type_index)%component_has_default = .false.
            context%derived_types(type_index)%component_default_value = 0_c_int64_t
            context%derived_types(type_index)%component_array_size = 1
            context%derived_types(type_index)%component_count = 0
            return
        end if

        old_names = context%derived_types(type_index)%component_names
        new_size = 2 * size(context%derived_types(type_index)%component_names)
        deallocate(context%derived_types(type_index)%component_names)
        allocate(context%derived_types(type_index)%component_names(new_size))
        context%derived_types(type_index)%component_names(1:old_size) = old_names
        deallocate(old_names)
        block
            logical, allocatable :: old_hd(:)
            integer(c_int64_t), allocatable :: old_dv(:)
            integer, allocatable :: old_as(:)
            associate (dt => context%derived_types(type_index))
                old_hd = dt%component_has_default
                old_dv = dt%component_default_value
                old_as = dt%component_array_size
                deallocate(dt%component_has_default)
                deallocate(dt%component_default_value)
                deallocate(dt%component_array_size)
                allocate(dt%component_has_default(new_size))
                allocate(dt%component_default_value(new_size))
                allocate(dt%component_array_size(new_size))
                dt%component_has_default = .false.
                dt%component_default_value = 0_c_int64_t
                dt%component_array_size = 1
                dt%component_has_default(1:old_size) = old_hd
                dt%component_default_value(1:old_size) = old_dv
                dt%component_array_size(1:old_size) = old_as
            end associate
        end block
    end subroutine grow_derived_type_components

    subroutine grow_module_export_derived_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%derived_type_count
        if (old_size > 0 .and. allocated(context%module_exports(export_idx)%derived_type_indices)) then
            if (old_size < size(context%module_exports(export_idx)%derived_type_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%derived_type_indices
        new_size = 2 * size(context%module_exports(export_idx)%derived_type_indices)
        deallocate(context%module_exports(export_idx)%derived_type_indices)
        allocate(context%module_exports(export_idx)%derived_type_indices(new_size))
        context%module_exports(export_idx)%derived_type_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_derived_indices

    subroutine grow_module_export_parameter_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%parameter_count
        if (old_size > 0 .and. &
            allocated(context%module_exports(export_idx)%parameter_indices)) then
            if (old_size < &
                size(context%module_exports(export_idx)%parameter_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%parameter_indices
        new_size = 2 * size(context%module_exports(export_idx)%parameter_indices)
        deallocate(context%module_exports(export_idx)%parameter_indices)
        allocate(context%module_exports(export_idx)%parameter_indices(new_size))
        context%module_exports(export_idx)%parameter_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_parameter_indices

    subroutine grow_module_export_variable_indices(context, export_idx)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: export_idx
        integer, allocatable :: old_indices(:)
        integer :: new_size, old_size

        old_size = context%module_exports(export_idx)%variable_count
        if (old_size > 0 .and. &
            allocated(context%module_exports(export_idx)%variable_indices)) then
            if (old_size < &
                size(context%module_exports(export_idx)%variable_indices)) return
        else
            return
        end if

        old_indices = context%module_exports(export_idx)%variable_indices
        new_size = 2 * size(context%module_exports(export_idx)%variable_indices)
        deallocate(context%module_exports(export_idx)%variable_indices)
        allocate(context%module_exports(export_idx)%variable_indices(new_size))
        context%module_exports(export_idx)%variable_indices(1:old_size) = old_indices
        deallocate(old_indices)
    end subroutine grow_module_export_variable_indices

    subroutine lower_module_variables(arena, root_index, context, error_msg)
        ! Emit a zero-initialised i32 global for each scalar integer module
        ! variable collected in the export table (#249 B7a). The global is named
        ! __<module>_MOD_<var> (gfortran convention). A using unit resolves the
        ! same name via lr_session_intern in import_module_variables.
        use, intrinsic :: iso_c_binding, only: c_bool, c_char, c_int32_t, &
                                               c_null_ptr, c_ptr, c_size_t
        use liric_session_bindings, only: lr_type_i32_s, lr_session_global, to_c_chars
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: m, i, var_idx, value_kind
        integer(c_int32_t) :: global_id
        character(len=:), allocatable :: mangled, var_name, type_name, err2
        character(kind=c_char), allocatable :: c_name(:)
        type(c_ptr) :: i32_type
        logical(c_bool) :: c_false_local

        call set_empty(error_msg)
        c_false_local = .false.
        do m = 1, context%module_export_count
            do i = 1, context%module_exports(m)%variable_count
                var_idx = context%module_exports(m)%variable_indices(i)
                if (.not. node_exists(arena, var_idx)) cycle
                if (.not. is_declaration_node(arena, var_idx)) cycle
                call get_declaration_type_name(arena, var_idx, type_name, err2)
                if (len_trim(err2) > 0) cycle
                call type_name_value_kind(type_name, 0, 0, value_kind, err2)
                if (len_trim(err2) > 0) cycle
                if (value_kind /= VALUE_I32) cycle
                call get_declaration_var_name(arena, var_idx, var_name, err2)
                if (len_trim(err2) > 0) cycle
                mangled = module_procedure_mangled( &
                    trim(context%module_exports(m)%module_name), trim(var_name))
                call to_c_chars(mangled, c_name)
                i32_type = lr_type_i32_s(context%session%handle)
                global_id = lr_session_global(context%session%handle, c_name, &
                                              i32_type, c_false_local, &
                                              c_null_ptr, 0_c_size_t)
                if (global_id < 0_c_int32_t) then
                    error_msg = 'LIRIC could not create module variable global: '// &
                                trim(mangled)
                    return
                end if
            end do
        end do
    end subroutine lower_module_variables
