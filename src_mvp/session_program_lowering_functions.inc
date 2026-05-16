    subroutine collect_internal_function_names(arena, root_index, context, &
                                               error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: root_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        context%integer_function_count = 0
        context%integer_function_names = ''
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

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        if (context%integer_function_count >= MAX_PROCEDURES) then
            error_msg = 'too many contained functions for direct LIRIC session MVP'
            return
        end if
        if (is_contained_i32_function(context, node%name)) then
            error_msg = 'duplicate contained function: '//trim(node%name)
            return
        end if

        next_index = context%integer_function_count + 1
        context%integer_function_names(next_index) = trim(node%name)
        context%integer_function_count = next_index
        call set_empty(error_msg)
    end subroutine register_internal_function_name

    subroutine copy_internal_function_names(source, destination)
        type(lowering_context_t), intent(in) :: source
        type(lowering_context_t), intent(inout) :: destination

        destination%integer_function_names = source%integer_function_names
        destination%integer_function_count = source%integer_function_count
    end subroutine copy_internal_function_names

    logical function is_contained_i32_function(context, name)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        is_contained_i32_function = .false.
        do i = 1, context%integer_function_count
            if (same_name(context%integer_function_names(i), name)) then
                is_contained_i32_function = .true.
                return
            end if
        end do
    end function is_contained_i32_function

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
                    call lower_i32_function(arena, node, context, error_msg)
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
        if (.not. arena%has_node_at(node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (function_def_node)
            is_internal_procedure_entry = .true.
        type is (subroutine_def_node)
            is_internal_procedure_entry = .true.
        end select
        if (.not. is_internal_procedure_entry) then
            if (allocated(arena%entries(node_index)%node_type)) then
                is_internal_procedure_entry = &
                    trim(arena%entries(node_index)%node_type) == 'contains'
            end if
        end if
    end function is_internal_procedure_entry

    subroutine lower_i32_function(arena, node, parent_context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: parent_context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lowering_context_t) :: context
        type(lr_operand_desc_t) :: value
        integer :: param_count
        logical :: terminated

        if (.not. allocated(node%name)) then
            error_msg = 'direct LIRIC session function requires a name'
            return
        end if
        if (allocated(node%return_type)) then
            if (trim(node%return_type) /= 'integer') then
                error_msg = 'direct LIRIC session MVP only supports integer functions'
                return
            end if
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%f64_print_format_id = parent_context%f64_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%string_literal_count = parent_context%string_literal_count
        call copy_internal_function_names(parent_context, context)
        context%in_internal_function = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (.not. context%session%begin_i32_function(node%name, param_count, &
                                                     error_msg)) return

        call define_i32_symbol(context, node%name, error_msg)
        if (len_trim(error_msg) > 0) return
        call define_i32_parameters(arena, node%param_indices, context, error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        value = context%symbols(find_symbol(context, node%name))%value
        if (.not. context%current_block_terminated) then
            if (.not. context%session%emit_ret_i32_operand(value, error_msg)) return
        end if
        if (.not. context%session%finish_function(error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_i32_function

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
        call copy_internal_function_names(parent_context, context)
        context%in_internal_function = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (.not. context%session%begin_void_subroutine(node%name, param_count, &
                                                        error_msg)) return

        call define_i32_parameters(arena, node%param_indices, context, error_msg)
        if (len_trim(error_msg) > 0) return

        if (allocated(node%body_indices)) then
            call lower_statement_list(arena, node%body_indices, context, value, &
                                      terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (.not. context%current_block_terminated) then
            if (.not. context%session%emit_ret_void(error_msg)) return
        end if
        if (.not. context%session%finish_function(error_msg)) return

        parent_context%session = context%session
        parent_context%string_literal_count = context%string_literal_count
        call set_empty(error_msg)
    end subroutine lower_void_subroutine

    subroutine define_i32_parameters(arena, param_indices, context, error_msg)
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
            call define_i32_parameter_symbol(context, name, i - 1, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine define_i32_parameters

    subroutine parameter_name(arena, node_index, name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable, intent(out) :: error_msg

        if (.not. arena%has_node_at(node_index)) then
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

    subroutine define_i32_parameter_symbol(context, name, param_index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        integer, intent(in) :: param_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate function parameter: '//trim(name)
            return
        end if
        if (context%symbol_count >= MAX_SYMBOLS) then
            error_msg = 'too many scalar symbols for direct LIRIC session MVP'
            return
        end if

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%address = context%session%ptr_param(param_index)
        context%symbols(index)%has_address = .true.
        context%symbols(index)%is_reference = .true.
        context%symbols(index)%value = context%session%i32_immediate(0_c_int64_t)
        context%symbols(index)%is_parameter = .true.
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i32_parameter_symbol
