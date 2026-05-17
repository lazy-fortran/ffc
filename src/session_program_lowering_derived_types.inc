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
            error_msg = 'too many derived types for direct LIRIC session MVP'
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

        if (.not. context%arena%has_node_at(component_index)) then
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
            error_msg = 'too many derived type components for direct LIRIC session MVP'
            return
        end if

        component_index = context%derived_types(type_index)%component_count + 1
        context%derived_types(type_index)%component_names(component_index) = &
            trim(name)
        context%derived_types(type_index)%component_count = component_index
        call set_empty(error_msg)
    end subroutine add_derived_component

    subroutine lower_derived_type_declaration(node, context, derived_type_index, &
                                              error_msg)
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_parameter) then
            call unsupported_feature_error('derived type parameter declaration', &
                                           node%line, node%column, &
                                           'derived type parameters are not '// &
                                           'supported by direct LIRIC session', &
                                           error_msg)
            return
        end if
        if (node%is_array) then
            call unsupported_feature_error('derived type array declaration', &
                                           node%line, node%column, &
                                           'direct LIRIC session only supports '// &
                                           'scalar derived type variables', &
                                           error_msg)
            return
        end if
        if (node%has_initializer) then
            call unsupported_feature_error('derived type constructor', &
                                           node%line, node%column, &
                                           'direct LIRIC session does not '// &
                                           'support derived type constructors', &
                                           error_msg)
            return
        end if

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call define_derived_symbol(context, node, node%var_names(i), &
                                           derived_type_index, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call define_derived_symbol(context, node, node%var_name, &
                                       derived_type_index, error_msg)
        else
            error_msg = 'derived type declaration did not expose a variable name'
        end if
    end subroutine lower_derived_type_declaration

    subroutine define_derived_symbol(context, node, name, derived_type_index, &
                                     error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: name
        integer, intent(in) :: derived_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: component_count
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate derived type declaration: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many symbols for direct LIRIC session MVP'
            return
        end if

        component_count = context%derived_types(derived_type_index)%component_count
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_DERIVED
        context%symbols(index)%is_derived = .true.
        context%symbols(index)%derived_type_index = derived_type_index
        context%symbols(index)%array_size = component_count

        if (.not. context%session%emit_i32_array_alloca( &
            component_count, context%symbols(index)%element_address, &
            error_msg)) then
            error_msg = 'derived_alloca['//trim(name)//']: '//error_msg
            return
        end if

        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_derived_symbol

    subroutine lower_derived_component_assignment(arena, node, target, context, &
                                                  value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(component_access_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address

        call lower_i32_expression(arena, node%value_index, context, value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_derived_component_address(arena, target, context, &
                                             component_address, error_msg, &
                                             'derived type component assignment target')
        if (len_trim(error_msg) > 0) return

        if (.not. context%session%emit_i32_store(value, component_address, &
                                                 error_msg)) then
            error_msg = 'component_store: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_assignment

    subroutine lower_derived_component_load(arena, node, context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: component_address

        call lower_derived_component_address(arena, node, context, &
                                             component_address, error_msg, &
                                             'derived type component expression')
        if (len_trim(error_msg) > 0) return

        if (.not. context%session%emit_i32_load(component_address, value, &
                                                error_msg)) then
            error_msg = 'component_load: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_load

    subroutine lower_derived_component_address(arena, node, context, &
                                               component_address, error_msg, &
                                               feature)
        type(ast_arena_t), intent(in) :: arena
        type(component_access_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: component_address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: feature
        character(len=:), allocatable :: base_name
        integer :: component_index
        integer :: symbol_index
        integer :: type_index
        type(lr_operand_desc_t) :: offset

        call component_base_name(arena, node%base_expr_index, base_name, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           trim(error_msg), error_msg)
            return
        end if

        symbol_index = find_symbol(context, base_name)
        if (symbol_index <= 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           'component base was not declared: '// &
                                           trim(base_name), error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_derived) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           'component base is not a derived type: '// &
                                           trim(base_name), error_msg)
            return
        end if

        type_index = context%symbols(symbol_index)%derived_type_index
        component_index = find_derived_component(context, type_index, &
                                                 node%component_name)
        if (component_index <= 0) then
            call unsupported_feature_error(feature, node%line, node%column, &
                                           'component is not declared for type '// &
                                           trim(context%derived_types(type_index)%name)// &
                                           ': '//trim(node%component_name), &
                                           error_msg)
            return
        end if

        offset = context%session%i32_immediate(int(component_index - 1, c_int64_t))
        if (.not. context%session%emit_i32_array_element_addr( &
            context%derived_types(type_index)%component_count, &
            context%symbols(symbol_index)%element_address, offset, &
            component_address, error_msg)) then
            error_msg = 'component_gep: '//error_msg
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_derived_component_address

    subroutine component_base_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. arena%has_node_at(node_index)) then
            error_msg = 'component base index does not reference an AST node'
            call set_empty(name)
            return
        end if

        select type (base => arena%entries(node_index)%node)
        type is (identifier_node)
            name = base%name
            call set_empty(error_msg)
        class default
            error_msg = 'nested component access is not supported'
            call set_empty(name)
        end select
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
        if (.not. arena%has_node_at(node_index)) return
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
    end function declaration_derived_type_index

    subroutine extracted_derived_type_name(type_spec, type_name)
        character(len=*), intent(in) :: type_spec
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable :: lowered
        integer :: close_pos
        integer :: open_pos

        lowered = trim(lowercase_text(type_spec))
        if (len_trim(lowered) >= 5) then
            if (lowered(1:5) == 'type(') then
                open_pos = index(lowered, '(')
                close_pos = index(lowered, ')', back=.true.)
                if (close_pos > open_pos + 1) then
                    type_name = lowered(open_pos + 1:close_pos - 1)
                else
                    type_name = ''
                end if
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
    end function find_derived_type

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
