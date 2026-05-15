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
                end select
            end do
        end select
    end subroutine lower_internal_functions

    logical function is_function_def(arena, node_index)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_function_def = .false.
        if (.not. arena%has_node_at(node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (function_def_node)
            is_function_def = .true.
        end select
    end function is_function_def

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
        if (allocated(node%return_type) .and. trim(node%return_type) /= 'integer') then
            error_msg = 'direct LIRIC session MVP only supports integer functions'
            return
        end if

        context%session = parent_context%session
        context%i32_print_format_id = parent_context%i32_print_format_id
        context%f64_print_format_id = parent_context%f64_print_format_id
        context%str_print_format_id = parent_context%str_print_format_id
        context%in_internal_function = .true.

        param_count = 0
        if (allocated(node%param_indices)) param_count = size(node%param_indices)
        if (.not. context%session%begin_i32_function(node%name, param_count, &
                                                     error_msg)) return

        call define_i32_symbol(context, node%name, error_msg)
        if (len_trim(error_msg) > 0) return
        call define_i32_parameters(arena, node, context, error_msg)
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
        call set_empty(error_msg)
    end subroutine lower_i32_function

    subroutine define_i32_parameters(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(function_def_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%param_indices)) return
        do i = 1, size(node%param_indices)
            call parameter_name(arena, node%param_indices(i), name, error_msg)
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
        context%symbols(index)%value = context%session%i32_param(param_index)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_i32_parameter_symbol
