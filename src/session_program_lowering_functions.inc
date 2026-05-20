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
        allocate(context%function_names(MAX_PROCEDURES))
        allocate(context%function_value_kinds(MAX_PROCEDURES))
        context%function_value_kinds = VALUE_I32
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
                end select
            end do
        end select
    end subroutine collect_internal_function_names

    subroutine register_internal_function_name(node, context, error_msg)
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: next_index
        integer :: value_kind

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
       call grow_function_names(context)
        if (contained_function_kind(context, node%name) /= 0) then
            error_msg = 'duplicate contained function: '//trim(node%name)
            return
        end if
        call function_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return

        next_index = context%function_count + 1
        context%function_names(next_index) = trim(node%name)
        context%function_value_kinds(next_index) = value_kind
        context%function_count = next_index
        call set_empty(error_msg)
    end subroutine register_internal_function_name

    subroutine function_value_kind(node, value_kind, error_msg)
        type(function_def_node), intent(in) :: node
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg

        value_kind = VALUE_I32
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
            call unsupported_feature_error('function result type', &
                                           node%line, node%column, &
                                           'direct LIRIC session supports '// &
                                           'integer, real, and logical '// &
                                           'contained functions', error_msg)
        end select
    end subroutine function_value_kind

    subroutine copy_internal_function_names(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        if (.not. allocated(destination%function_names)) then
            allocate(destination%function_names(MAX_PROCEDURES))
        end if
        if (.not. allocated(destination%function_value_kinds)) then
            allocate(destination%function_value_kinds(MAX_PROCEDURES))
        end if
        do i = 1, source%function_count
            destination%function_names(i) = source%function_names(i)
            destination%function_value_kinds(i) = source%function_value_kinds(i)
        end do
        destination%function_count = source%function_count
    end subroutine copy_internal_function_names

    subroutine copy_derived_type_table(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination
        integer :: i

        if (.not. allocated(destination%derived_types)) then
            allocate(destination%derived_types(source%derived_type_count))
        end if
        do i = 1, source%derived_type_count
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
        logical :: terminated

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        call function_value_kind(node, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%f64_print_format_id = parent_context%f64_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
        context%arena = arena
        allocate(context%symbols(MAX_SYMBOLS))
        call copy_internal_function_names(parent_context, context)
        call copy_derived_type_table(parent_context, context)
        context%in_internal_function = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (value_kind == VALUE_F64) then
            if (.not. begin_liric_f64_function(context%session, node%name, &
                                               param_count, error_msg)) return
        else
            if (.not. begin_i32_function(context%session, node%name, param_count, &
                                                         error_msg)) return
        end if

        call define_symbol(context, node%name, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        context%current_function_result_index = find_symbol(context, node%name)
        call define_reference_parameters(arena, node%param_indices, context, &
                                         error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        result_index = find_symbol(context, node%name)
        if (result_index <= 0) then
            error_msg = 'function result symbol was not declared: '// &
                        trim(node%name)
            return
        end if
        value = context%symbols(result_index)%value
        if (.not. context%current_block_terminated) then
            if (.not. emit_ret_i32_operand(context%session, value, error_msg)) return
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
        context%f64_print_format_id = parent_context%f64_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
        context%arena = arena
        allocate(context%symbols(MAX_SYMBOLS))
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

    subroutine define_reference_parameters(arena, param_indices, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: param_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(param_indices)) return
        do i = 1, size(param_indices)
            call parameter_name(arena, param_indices(i), name, error_msg)
            if (len_trim(error_msg) > 0) return
            call define_parameter_symbol(context, name, i - 1, error_msg)
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

        select type (node => arena%entries(node_index)%node)
        type is (parameter_declaration_node)
            name = node%name
            call set_empty(error_msg)
        type is (identifier_node)
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
