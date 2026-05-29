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
            if (.not. use_only_wants(use_node, trim(type_name))) cycle
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
        call validate_use_only_names(arena, export_idx, use_node, context, &
                                     error_msg)
        if (len_trim(error_msg) > 0) return
        call import_use_renames(arena, export_idx, use_node, context, error_msg)
    end subroutine import_module_derived_types

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

        call set_empty(error_msg)
        if (.not. allocated(use_node%rename_list)) return

        do i = 1, size(use_node%rename_list) - 1, 2
            found = .false.
            do j = 1, context%module_exports(export_idx)%parameter_count
                param_idx = context%module_exports(export_idx)%parameter_indices(j)
                if (.not. node_exists(arena, param_idx)) cycle
                select type (decl => arena%entries(param_idx)%node)
                type is (declaration_node)
                    if (.not. allocated(decl%var_name)) cycle
                    if (.not. same_name(decl%var_name, &
                                        use_node%rename_list(i + 1)%s)) cycle
                    call declaration_value_kind(decl, value_kind, error_msg)
                    if (len_trim(error_msg) > 0) return
                    call import_renamed_parameter(decl, &
                        trim(use_node%rename_list(i)%s), value_kind, context, &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                    found = .true.
                end select
                if (found) exit
            end do
            if (.not. found) then
                error_msg = 'use rename target is not an exported parameter of '// &
                    trim(use_node%module_name)//': '// &
                    trim(use_node%rename_list(i + 1)%s)
                return
            end if
        end do
    end subroutine import_use_renames

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
        if (find_symbol(context, local_name) > 0) then
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

        module_exports_name = .false.
        do i = 1, context%module_exports(export_idx)%derived_type_count
            select type (dtype => arena%entries( &
                context%module_exports(export_idx)%derived_type_indices(i))%node)
            type is (derived_type_node)
                if (allocated(dtype%name)) then
                    if (same_name(dtype%name, name)) then
                        module_exports_name = .true.
                        return
                    end if
                end if
            end select
        end do
        do i = 1, context%module_exports(export_idx)%parameter_count
            select type (decl => arena%entries( &
                context%module_exports(export_idx)%parameter_indices(i))%node)
            type is (declaration_node)
                if (allocated(decl%var_name)) then
                    if (same_name(decl%var_name, name)) then
                        module_exports_name = .true.
                        return
                    end if
                end if
            end select
        end do
    end function module_exports_name

    subroutine import_module_constants(arena, export_idx, use_node, context, &
                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: export_idx
        type(use_statement_node), intent(in) :: use_node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: param_idx
        integer :: value_kind

        call set_empty(error_msg)
        do i = 1, context%module_exports(export_idx)%parameter_count
            param_idx = context%module_exports(export_idx)%parameter_indices(i)
            if (.not. node_exists(arena, param_idx)) cycle
            select type (decl => arena%entries(param_idx)%node)
            type is (declaration_node)
                if (.not. use_only_wants(use_node, trim(decl%var_name))) cycle
                if (find_symbol(context, decl%var_name) > 0) cycle
                call declaration_value_kind(decl, value_kind, error_msg)
                if (len_trim(error_msg) > 0) return
                ! lower_constant_declaration rejects non-integer parameters with
                ! a targeted diagnostic (real/character exports are follow-ups).
                call lower_constant_declaration(decl, context, value_kind, &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end select
        end do
    end subroutine import_module_constants

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
