    subroutine collect_internal_function_names(arena, root_index, context, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        context%function_count = 0
        if (allocated(context%function_names)) deallocate(context%function_names)
        if (allocated(context%function_value_kinds)) deallocate(context%function_value_kinds)
        if (allocated(context%function_param_counts)) &
            deallocate(context%function_param_counts)
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        allocate(context%function_param_counts(8))
        context%function_value_kinds = VALUE_I32
        context%function_param_counts = 0
        call set_empty(error_msg)

        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (function_def_node)
                    call register_internal_function_name(node, context, &
                                                         error_msg)
                    if (len_trim(error_msg) > 0) return
                type is (subroutine_def_node)
                    call register_internal_subroutine_name(node, context, &
                                                           error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end do
        end select
    end subroutine collect_internal_function_names

    subroutine register_internal_subroutine_name(node, context, error_msg)
        type(subroutine_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: next_index

        call set_empty(error_msg)
        if (.not. allocated(node%name)) return
        call grow_function_names(context)
        if (contained_function_kind(context, node%name) /= 0) return
        next_index = context%function_count + 1
        context%function_names(next_index) = trim(node%name)
        context%function_value_kinds(next_index) = VALUE_SUBROUTINE
        if (allocated(node%param_indices)) then
            context%function_param_counts(next_index) = size(node%param_indices)
        else
            context%function_param_counts(next_index) = 0
        end if
        context%function_count = next_index
    end subroutine register_internal_subroutine_name

    integer function proc_param_count(context, name) result(count)
        ! Declared parameter count of a contained procedure, or -1 if unknown.
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        count = -1
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                count = context%function_param_counts(i)
                return
            end if
        end do
    end function proc_param_count

    subroutine register_internal_function_name(node, context, error_msg)
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: next_index
        integer :: value_kind
        integer :: derived_type_index

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
       call grow_function_names(context)
        if (contained_function_kind(context, node%name) /= 0) then
            error_msg = 'duplicate contained function: '//trim(node%name)
            return
        end if
        call function_value_kind(node, context, value_kind, derived_type_index, &
                                 error_msg)
        if (len_trim(error_msg) > 0) return

        next_index = context%function_count + 1
        context%function_names(next_index) = trim(node%name)
        context%function_value_kinds(next_index) = value_kind
        if (allocated(node%param_indices)) then
            context%function_param_counts(next_index) = size(node%param_indices)
        else
            context%function_param_counts(next_index) = 0
        end if
        context%function_count = next_index
        call set_empty(error_msg)
    end subroutine register_internal_function_name

  subroutine function_value_kind(node, context, value_kind, derived_type_index, &
                                   error_msg)
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: derived_type_index
        integer :: char_len
        character(len=:), allocatable :: derived_name
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
        derived_type_index = 0
        call set_empty(error_msg)
        if (.not. allocated(node%return_type)) return

        select case (trim(node%return_type))
        case ('integer')
            value_kind = VALUE_I32
        case ('real')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case default
            if (is_character_type_name(node%return_type)) then
                call parse_character_length(node%return_type, &
                                            char_len, error_msg)
                if (len_trim(error_msg) > 0) return
                if (char_len <= 0) then
                    value_kind = VALUE_DEFERRED_CHARACTER_RESULT
                else
                    call unsupported_feature_error('function result type', &
                                                    node%line, node%column, &
                                                    'direct LIRIC session '// &
                                                    'supports deferred-length '// &
                                                    'character, integer, real, '// &
                                                    'and logical contained '// &
                                                    'functions', error_msg)
                end if
                return
            end if
            call extracted_derived_type_name(node%return_type, derived_name)
            if (len_trim(derived_name) > 0) &
                derived_type_index = find_derived_type(context, derived_name)
            ! The result type is often declared in the body (result(p) with
            ! type(t) :: p), not on the function statement.
            if (derived_type_index == 0) &
                derived_type_index = result_var_derived_type(node, context)
            if (derived_type_index > 0) then
                value_kind = VALUE_DERIVED
            else
                call unsupported_feature_error('function result type', &
                                                node%line, node%column, &
                                                'direct LIRIC session supports '// &
                                                'integer, real, logical, '// &
                                                'deferred-length character, and '// &
                                                'derived-type contained functions', &
                                                error_msg)
            end if
        end select
    end subroutine function_value_kind

    integer function result_var_derived_type(node, context) result(type_index)
        ! Find the derived type of the function result variable from its body
        ! declaration (result(p) with type(t) :: p inside the body).
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: result_name
        integer :: i, k

        type_index = 0
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) &
                result_name = trim(node%result_variable)
        end if
        if (.not. allocated(node%body_indices)) return
        do i = 1, size(node%body_indices)
            if (.not. node_exists(context%arena, node%body_indices(i))) cycle
            select type (decl => context%arena%entries(node%body_indices(i))%node)
            type is (declaration_node)
                if (allocated(decl%var_name)) then
                    if (trim(decl%var_name) == result_name) then
                        type_index = declaration_derived_type_index(context, decl)
                        if (type_index > 0) return
                    end if
                end if
                if (decl%is_multi_declaration .and. allocated(decl%var_names)) then
                    do k = 1, size(decl%var_names)
                        if (trim(decl%var_names(k)) == result_name) then
                            type_index = declaration_derived_type_index(context, &
                                                                        decl)
                            if (type_index > 0) return
                        end if
                    end do
                end if
            end select
        end do
    end function result_var_derived_type

    subroutine copy_internal_function_names(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        if (.not. allocated(destination%function_names)) then
            allocate(destination%function_names(8))
        end if
        if (.not. allocated(destination%function_value_kinds)) then
            allocate(destination%function_value_kinds(8))
        end if
        if (.not. allocated(destination%function_param_counts)) then
            allocate(destination%function_param_counts(8))
            destination%function_param_counts = 0
        end if
        do i = 1, source%function_count
            call ensure_function_names_capacity(destination, i)
            destination%function_names(i) = source%function_names(i)
            destination%function_value_kinds(i) = source%function_value_kinds(i)
            destination%function_param_counts(i) = source%function_param_counts(i)
        end do
        destination%function_count = source%function_count
    end subroutine copy_internal_function_names

    subroutine copy_derived_type_table(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        do i = 1, source%derived_type_count
            call ensure_derived_types_capacity(destination, i)
            destination%derived_types(i) = source%derived_types(i)
        end do
        destination%derived_type_count = source%derived_type_count
    end subroutine copy_derived_type_table

    logical function is_contained_i32_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_i32_function = &
            contained_function_kind(context, name) == VALUE_I32
    end function is_contained_i32_function

    logical function is_contained_f64_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_f64_function = &
            contained_function_kind(context, name) == VALUE_F64
    end function is_contained_f64_function

    logical function is_contained_logical_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_logical_function = &
            contained_function_kind(context, name) == VALUE_LOGICAL
    end function is_contained_logical_function

    logical function is_contained_derived_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_derived_function = &
            contained_function_kind(context, name) == VALUE_DERIVED
    end function is_contained_derived_function

    integer function contained_function_kind(context, name) result(value_kind)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        value_kind = 0
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                value_kind = context%function_value_kinds(i)
                return
            end if
        end do
   end function contained_function_kind

    logical function is_contained_deferred_char_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name

        is_contained_deferred_char_function = &
            contained_function_kind(context, name) == &
            VALUE_DEFERRED_CHARACTER_RESULT
    end function is_contained_deferred_char_function

    subroutine lower_internal_functions(arena, root_index, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        select type (program => arena%entries(root_index)%node)
        type is (program_node)
            if (.not. allocated(program%body_indices)) return
            do i = 1, size(program%body_indices)
                select type (node => arena%entries(program%body_indices(i))%node)
                type is (function_def_node)
                    call lower_scalar_function(arena, node, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                type is (subroutine_def_node)
                    call lower_void_subroutine(arena, node, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                end select
            end do
        end select
    end subroutine lower_internal_functions

    logical function is_internal_procedure_entry(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_internal_procedure_entry = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (function_def_node)
            is_internal_procedure_entry = .true.
        type is (subroutine_def_node)
            is_internal_procedure_entry = .true.
        end select
        if (.not. is_internal_procedure_entry) then
            is_internal_procedure_entry = &
                trim(get_node_type_at(arena, node_index)) == 'contains'
        end if
    end function is_internal_procedure_entry

   subroutine lower_scalar_function(arena, node, parent_context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        integer :: result_index
        integer :: value_kind
        integer :: result_derived_index
        logical :: terminated
        character(len=:), allocatable :: result_name

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        call function_value_kind(node, parent_context, value_kind, &
                                 result_derived_index, error_msg)
        if (len_trim(error_msg) > 0) return
        result_name = trim(node%name)
        if (allocated(node%result_variable)) then
            if (len_trim(node%result_variable) > 0) then
                result_name = trim(node%result_variable)
            end if
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
  context%arena = arena
        allocate(context%symbols(16))
        allocate(context%derived_types(8))
        allocate(context%module_exports(4))
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        context%function_value_kinds = VALUE_I32
        call copy_internal_function_names(parent_context, context)
        call copy_derived_type_table(parent_context, context)
        context%in_internal_function = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (value_kind == VALUE_F64) then
            if (.not. begin_liric_f64_function(context%session, node%name, &
                                               param_count, error_msg)) return
        else if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                 value_kind == VALUE_DERIVED) then
            if (.not. begin_void_subroutine(context%session, node%name, &
                                            param_count + 1, error_msg)) return
        else
            if (.not. begin_i32_function(context%session, node%name, param_count, &
                                                          error_msg)) return
        end if

        if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT) then
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            context%symbols(context%current_function_result_index)%name = &
                trim(result_name)
            context%symbols(context%current_function_result_index)%value_kind = &
                VALUE_CHARACTER
            context%symbols(context%current_function_result_index)%&
                is_deferred_character = .true.
            context%symbols(context%current_function_result_index)%&
                has_character_value = .false.
            if (.not. emit_ptr_offset(context%session, &
                ptr_param(context%session, 0), 0_c_int64_t, &
                context%symbols(context%current_function_result_index)%deferred_data, &
                error_msg)) return
            if (.not. emit_ptr_offset(context%session, &
                ptr_param(context%session, 0), 8_c_int64_t, &
                context%symbols(context%current_function_result_index)%deferred_length, &
                error_msg)) return
            context%symbol_count = context%current_function_result_index
        else if (value_kind == VALUE_DERIVED) then
            call grow_symbols(context)
            context%current_function_result_index = context%symbol_count + 1
            associate (rsym => context%symbols(context%current_function_result_index))
                rsym%name = trim(result_name)
                rsym%value_kind = VALUE_DERIVED
                rsym%is_derived = .true.
                rsym%derived_type_index = result_derived_index
                rsym%array_size = total_component_slots(context, &
                                                        result_derived_index)
                rsym%element_address = ptr_param(context%session, 0)
            end associate
            context%symbol_count = context%current_function_result_index
        else
            call define_symbol(context, result_name, value_kind, error_msg)
            if (len_trim(error_msg) > 0) return
            context%current_function_result_index = find_symbol(context, &
                                                                result_name)
        end if
        if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
            value_kind == VALUE_DERIVED) then
            call define_reference_parameters(arena, node%param_indices, context, &
                                             error_msg, 1)
        else
            call define_reference_parameters(arena, node%param_indices, context, &
                                             error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        result_index = find_symbol(context, result_name)
        if (result_index <= 0) then
            error_msg = 'function result symbol was not declared: '// &
                        trim(result_name)
            return
        end if
        if (.not. context%current_block_terminated) then
            if (value_kind == VALUE_DEFERRED_CHARACTER_RESULT .or. &
                value_kind == VALUE_DERIVED) then
                if (.not. emit_ret_void(context%session, error_msg)) return
            else
                value = context%symbols(result_index)%value
                if (.not. emit_ret_i32_operand(context%session, value, error_msg)) return
            end if
        end if
        if (.not. finish_function(context%session, error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_scalar_function

    subroutine lower_void_subroutine(arena, node, parent_context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(subroutine_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        logical :: terminated

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session subroutine requires a name'
            return
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
        context%arena = arena
        allocate(context%symbols(16))
        allocate(context%derived_types(8))
        allocate(context%module_exports(4))
        allocate(context%function_names(8))
        allocate(context%function_value_kinds(8))
        context%function_value_kinds = VALUE_I32
        call copy_internal_function_names(parent_context, context)
        call copy_derived_type_table(parent_context, context)
        context%in_internal_function = .true.
        context%in_internal_subroutine = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (.not. begin_void_subroutine(context%session, node%name, param_count, &
                                                        error_msg)) return

        call define_reference_parameters(arena, node%param_indices, context, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (.not. context%current_block_terminated) then
            if (.not. emit_ret_void(context%session, error_msg)) return
        end if
        if (.not. finish_function(context%session, error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_void_subroutine

    subroutine define_reference_parameters(arena, param_indices, context, &
                                           error_msg, param_offset)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, intent(in), optional :: param_offset
        character(len=:), allocatable :: name
        integer :: offset
        integer :: i

        call set_empty(error_msg)
        offset = 0
        if (present(param_offset)) offset = param_offset
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            call define_parameter_symbol(context, name, i - 1 + offset, &
                                         error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine define_reference_parameters

    subroutine parameter_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'parameter index does not reference an AST node'
            call set_empty(name)
            return
        end if

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            return
        end if

        select type (node => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            name = node%name
            call set_empty(error_msg)
        class default
            error_msg = 'direct LIRIC session function parameter needs a name'
            call set_empty(name)
        end select
    end subroutine parameter_name

    subroutine define_parameter_symbol(context, name, param_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate function parameter: '//trim(name)
            return
        end if
       call grow_symbols(context)

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%address = ptr_param(context%session, param_index)
        context%symbols(index)%has_address = .true.
        context%symbols(index)%is_reference = .true.
        context%symbols(index)%value = i32_immediate(context%session, 0_c_int64_t)
        context%symbols(index)%is_parameter = .true.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_parameter_symbol

    subroutine ensure_function_names_capacity(context, required_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: required_index

        if (required_index <= size(context%function_names)) return
        call grow_function_names(context)
        if (required_index <= size(context%function_names)) return
        allocate(context%function_names(required_index), &
                 source=context%function_names)
        allocate(context%function_value_kinds(required_index), &
                 source=context%function_value_kinds)
        allocate(context%function_param_counts(required_index), &
                 source=context%function_param_counts)
    end subroutine ensure_function_names_capacity

    subroutine ensure_derived_types_capacity(context, required_index)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: required_index

        if (required_index <= size(context%derived_types)) return
        call grow_derived_types(context)
        if (required_index <= size(context%derived_types)) return
        allocate(context%derived_types(required_index), &
                 source=context%derived_types)
    end subroutine ensure_derived_types_capacity

    subroutine ensure_all_context_arrays_allocated(context, min_symbols, &
                                                    min_derived_types, &
                                                    min_module_exports, &
                                                    min_functions)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: min_symbols
        integer, intent(in) :: min_derived_types
        integer, intent(in) :: min_module_exports
        integer, intent(in) :: min_functions

        if (.not. allocated(context%symbols)) then
            allocate(context%symbols(max(min_symbols, 16)))
        end if
        if (.not. allocated(context%derived_types)) then
            allocate(context%derived_types(max(min_derived_types, 8)))
        end if
        if (.not. allocated(context%module_exports)) then
            allocate(context%module_exports(max(min_module_exports, 4)))
        end if
        if (.not. allocated(context%function_names)) then
            allocate(context%function_names(max(min_functions, 8)))
            allocate(context%function_value_kinds(max(min_functions, 8)))
            allocate(context%function_param_counts(max(min_functions, 8)))
            context%function_value_kinds = VALUE_I32
            context%function_param_counts = 0
        end if
    end subroutine ensure_all_context_arrays_allocated
