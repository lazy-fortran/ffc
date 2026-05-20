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
        character(len=64) :: type_name

        call set_empty(error_msg)

        if (.not. allocated(use_node%module_name)) return

        module_lowered = lowercase_text(use_node%module_name)
        export_idx = find_module_export_index(context, module_lowered)
        if (export_idx <= 0) return

        do i = 1, context%module_exports(export_idx)%derived_type_count
            type_idx = context%module_exports(export_idx)% &
                derived_type_indices(i)
            if (.not. node_exists(arena, type_idx)) then
                error_msg = 'invalid derived type arena index in module '// &
                    trim(context%module_exports(export_idx)%module_name)
                return
            end if
            select type (dtype => arena%entries(type_idx)%node)
            type is (derived_type_node)
                if (.not. allocated(dtype%name)) cycle
                type_name = trim(dtype%name)
            class default
                cycle
            end select
            type_idx_in_context = find_derived_type(context, type_name)
            if (type_idx_in_context > 0) then
                cycle
            end if
            call add_derived_type_from_arena(arena, type_idx, context, &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine import_module_derived_types

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

        if (context%derived_type_count >= MAX_DERIVED_TYPES) then
            error_msg = 'too many derived types for ffc direct-session lowering'
            return
        end if

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

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        context%derived_types(type_index_new)%component_count = 0

        ! Get the derived type name from the node
        select type (dtype => arena%entries(type_index)%node)
        type is (derived_type_node)
            context%derived_types(type_index_new)%name = trim(dtype%name)
        end select

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

        call set_empty(error_msg)

        if (.not. node_exists(arena, type_index)) then
            error_msg = 'derived type index does not reference an AST node'
            return
        end if

        select type (dtype => arena%entries(type_index)%node)
        type is (derived_type_node)
            if (allocated(dtype%component_indices)) then
                do i = 1, size(dtype%component_indices)
                    call collect_component_declaration( &
                        dtype%component_indices(i), dtype, context, type_index_new, error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
        class default
            error_msg = 'node at type_index is not a derived_type_node'
        end select
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
