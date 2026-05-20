    subroutine collect_derived_type_definitions(arena, root_index, context, &
                                                error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        context%derived_types = derived_type_info_t()
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
                end select
            end do
        end select
    end subroutine collect_derived_type_definitions

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

        context%module_exports = module_exports_t()
        context%module_export_count = 0
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (mod_node => arena%entries(program%body_indices(i))%node)
                type is (module_node)
                    if (.not. allocated(mod_node%name)) cycle
                    if (.not. allocated(mod_node%declaration_indices)) cycle
                    if (context%module_export_count >= MAX_MODULE_NAMES) then
                        error_msg = 'too many modules for ffc direct-session lowering'
                        return
                    end if

                    export_idx = context%module_export_count + 1
                    context%module_exports(export_idx)%module_name = &
                        trim(mod_node%name)
                    context%module_exports(export_idx)%derived_type_count = 0

                     do j = 1, size(mod_node%declaration_indices)
                        select type (decl => &
                            arena%entries(mod_node%declaration_indices(j))%node)
                        type is (derived_type_node)
                            if (.not. allocated(decl%name)) cycle

                            context%module_exports(export_idx)% &
                                derived_type_indices( &
                                context%module_exports(export_idx)% &
                                derived_type_count + 1) = mod_node%declaration_indices(j)
                            context%module_exports(export_idx)% &
                                derived_type_count = &
                                context%module_exports(export_idx)% &
                                derived_type_count + 1
                        end select
                    end do

                    context%module_export_count = export_idx
                end select
            end do
        end select
    end subroutine collect_module_exports

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

        if (allocated(node%extends_parent)) then
            call unsupported_feature_error('derived type inheritance', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'plain derived types', error_msg)
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
        if (node%has_contains .or. allocated(node%binding_indices)) then
            call unsupported_feature_error('type-bound procedure', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support type-bound procedures', &
                                           error_msg)
            return
        end if
        if (.not. allocated(node%component_indices)) then
            call unsupported_feature_error('derived type definition', &
                                           node%line, node%column, &
                                           'derived types need at least one '// &
                                           'integer component', error_msg)
            return
        end if

        if (context%derived_type_count >= MAX_DERIVED_TYPES) then
            error_msg = 'too many derived types for ffc direct-session lowering'
            return
        end if

        type_index = context%derived_type_count + 1
        context%derived_types(type_index)%name = trim(node%name)
        call collect_derived_components(node, context, type_index, error_msg)
        if (len_trim(error_msg) > 0) return

        context%derived_type_count = type_index
        call set_empty(error_msg)
    end subroutine lower_derived_type_definition

    subroutine collect_derived_components(node, context, type_index, error_msg)
        type(derived_type_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        context%derived_types(type_index)%component_count = 0
        do i = 1, size(node%component_indices)
            call collect_component_declaration( &
                node%component_indices(i), node, context, type_index, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        if (context%derived_types(type_index)%component_count <= 0) then
            call unsupported_feature_error('derived type definition', &
                                           node%line, node%column, &
                                           'derived types need at least one '// &
                                           'integer component', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine collect_derived_components

subroutine collect_component_declaration(component_index, parent, context, &
                                              type_index, error_msg)
        integer, intent(in) :: component_index
        type(derived_type_node), intent(in) :: parent
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (.not. node_exists(context%arena, component_index)) then
            error_msg = 'derived type component index does not reference an AST node'
            return
        end if

        select type (component => context%arena%entries(component_index)%node)
        type is (declaration_node)
            call validate_component_declaration(component, parent, context, &
                                                error_msg)
            if (len_trim(error_msg) > 0) return
            if (component%is_multi_declaration .and. &
                allocated(component%var_names)) then
                do i = 1, size(component%var_names)
                    call add_derived_component(context, type_index, &
                                               component%var_names(i), &
                                               component%line, component%column, &
                                               error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            else if (allocated(component%var_name)) then
                call add_derived_component(context, type_index, component%var_name, &
                                           component%line, component%column, &
                                           error_msg)
            else
                error_msg = 'derived type component declaration did not expose a name'
                return
            end if
        class default
            call unsupported_feature_error('derived type component', &
                                           parent%line, parent%column, &
                                           'direct LIRIC session only supports '// &
                                           'integer component declarations', &
                                           error_msg)
            return
        end select
        call set_empty(error_msg)
    end subroutine collect_component_declaration

    subroutine validate_component_declaration(component, parent, context, &
                                              error_msg)
        type(declaration_node), intent(in) :: component
        type(derived_type_node), intent(in) :: parent
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: nested_index

        if (component%is_array) then
            call unsupported_feature_error('derived type component', &
                                           component%line, component%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar integer components', error_msg)
            return
        end if
        if (component%has_initializer) then
            call unsupported_feature_error('derived type component initializer', &
                                           component%line, component%column, &
                                           'component default initialization is '// &
                                           'not supported', error_msg)
            return
        end if
        if (.not. allocated(component%type_name)) then
            call set_empty(error_msg)
            return
        end if

        nested_index = declaration_derived_type_index(context, component)
        if (nested_index > 0 .or. is_derived_type_spec(component%type_name)) then
            call unsupported_feature_error('nested derived type component', &
                                           component%line, component%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar integer components', error_msg)
            return
        end if
        if (trim(lowercase_text(component%type_name)) /= 'integer') then
            call unsupported_feature_error('derived type component', &
                                           component%line, component%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar integer components', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine validate_component_declaration

    subroutine add_derived_component(context, type_index, name, line, column, &
                                     error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: type_index
        character(len=*), intent(in) :: name
        integer, intent(in) :: line, column
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: component_index

        if (find_derived_component(context, type_index, name) > 0) then
            error_msg = 'duplicate derived type component: '//trim(name)
            return
        end if
        if (context%derived_types(type_index)%component_count >= &
            MAX_DERIVED_COMPONENTS) then
            error_msg = 'too many derived type components for ffc direct-session lowering'
            return
        end if

        component_index = context%derived_types(type_index)%component_count + 1
        context%derived_types(type_index)%component_names(component_index) = &
            trim(name)
        context%derived_types(type_index)%component_count = component_index
        call set_empty(error_msg)
    end subroutine add_derived_component
