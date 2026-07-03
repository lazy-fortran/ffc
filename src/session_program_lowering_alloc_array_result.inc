    logical function is_alloc_array_result_call(arena, value_index, context)
        ! A bare call to a contained function whose result is an allocatable
        ! array (the descriptor-sret ABI).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(in) :: context

        is_alloc_array_result_call = .false.
        if (.not. node_exists(arena, value_index)) return
        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            if (allocated(v%name) .and. v%base_expr_index == 0 .and. &
                .not. v%is_array_access) &
                is_alloc_array_result_call = &
                    contained_function_kind(context, v%name) == &
                    VALUE_ALLOC_ARRAY_RESULT
        end select
    end function is_alloc_array_result_call

    subroutine lower_alloc_array_result_assignment(arena, value_index, &
                                                   dest_symbol_index, context, &
                                                   error_msg)
        ! lhs = alloc_fn(...): pass a zeroed temporary descriptor as the hidden
        ! sret pointer, let the callee allocate into it, then move that descriptor
        ! into lhs (freeing lhs's prior storage first).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        integer, intent(in) :: dest_symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(call_or_subscript_node) :: call_node
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        type(lr_operand_desc_t) :: temp_desc, dest_desc, old_data, zero
        type(lr_operand_desc_t) :: moved_ptr, field
        integer :: i
        integer(c_int64_t) :: off

        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            call_node = v
        class default
            error_msg = 'expected function call for allocatable array result'
            return
        end select
        func_name = call_node%name

        if (allocated(call_node%arg_indices)) then
            call prepare_reference_args(arena, call_node%arg_indices, context, &
                                        VALUE_I32, call_node%name, user_args, &
                                        copyback_indices, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            allocate (user_args(0))
            allocate (copyback_indices(0))
        end if

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 40_c_int64_t), temp_desc, &
                error_msg)) return
        zero = i64_immediate(context%session, 0_c_int64_t)
        do i = 0, 4
            if (.not. emit_i64_store_at(context%session, zero, temp_desc, &
                    int(i * 8, c_int64_t), error_msg)) return
        end do

        allocate (all_args(size(user_args) + 1))
        all_args(1) = temp_desc
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do

        if (.not. emit_void_call(context%session, &
                                 call_emit_name(arena, trim(func_name)), &
                                 all_args, error_msg)) return
        call copy_back_reference_args(context, user_args, copyback_indices, &
                                      error_msg)
        if (len_trim(error_msg) > 0) return

        dest_desc = context%symbols(dest_symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, dest_desc, old_data, error_msg)) &
            return
        if (.not. emit_free(context%session, old_data, error_msg)) return

        if (.not. emit_ptr_load(context%session, temp_desc, moved_ptr, error_msg)) &
            return
        if (.not. emit_ptr_store(context%session, moved_ptr, dest_desc, &
                error_msg)) return
        do i = 1, 4
            off = int(i * 8, c_int64_t)
            if (.not. emit_i64_load_at(context%session, temp_desc, off, field, &
                    error_msg)) return
            if (.not. emit_i64_store_at(context%session, field, dest_desc, off, &
                    error_msg)) return
        end do
        context%symbols(dest_symbol_index)%allocatable_static_size = &
            alloc_array_result_static_size(arena, context, func_name)
        call set_empty(error_msg)
    end subroutine lower_alloc_array_result_assignment

    integer function alloc_array_result_static_size(arena, context, name) &
            result(static_size)
        ! Compile-time element count of a rank-1 allocatable array result, read
        ! from the last size-fixing statement in the function body (an array
        ! constructor assignment or a constant-extent allocate). 0 when the
        ! extent is only known at run time, which keeps whole-array print/index
        ! of the destination on their existing unsupported diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        type(function_def_node), pointer :: fn_node
        character(len=:), allocatable :: result_name, id_name, id_err
        integer, allocatable :: flat(:)
        integer(c_int64_t) :: n
        integer :: i, node_index, size_index1, size_index2

        static_size = 0
        node_index = 0
        do i = 1, context%function_count
            if (same_name(context%function_names(i), name)) then
                node_index = context%function_node_indices(i)
                exit
            end if
        end do
        if (node_index <= 0 .or. .not. node_exists(arena, node_index)) return
        fn_node => get_node_as_function_def(arena, node_index)
        if (.not. associated(fn_node)) return
        result_name = trim(fn_node%name)
        if (allocated(fn_node%result_variable)) then
            if (len_trim(fn_node%result_variable) > 0) &
                result_name = trim(fn_node%result_variable)
        end if
        if (.not. allocated(fn_node%body_indices)) return

        do i = 1, size(fn_node%body_indices)
            if (.not. node_exists(arena, fn_node%body_indices(i))) cycle
            select type (stmt => arena%entries(fn_node%body_indices(i))%node)
            type is (assignment_node)
                if (.not. is_identifier(arena, stmt%target_index)) cycle
                call get_identifier_name(arena, stmt%target_index, id_name, id_err)
                if (len_trim(id_err) > 0) cycle
                if (trim(id_name) /= result_name) cycle
                if (.not. node_exists(arena, stmt%value_index)) cycle
                select type (rhs => arena%entries(stmt%value_index)%node)
                type is (array_literal_node)
                    if (.not. allocated(rhs%element_indices)) cycle
                    allocate (flat(0))
                    call flatten_constructor_elements(arena, rhs%element_indices, &
                                                      flat, id_err)
                    if (len_trim(id_err) == 0) static_size = size(flat)
                    deallocate (flat)
                end select
            type is (allocate_statement_node)
                if (.not. allocated(stmt%var_indices)) cycle
                if (size(stmt%var_indices) /= 1) cycle
                call allocate_target(arena, stmt, id_name, size_index1, &
                                     size_index2, id_err)
                if (len_trim(id_err) > 0) cycle
                if (trim(id_name) /= result_name) cycle
                if (size_index1 == 0 .or. size_index2 /= 0) cycle
                call eval_i32_constant(arena, size_index1, context, n, id_err)
                if (len_trim(id_err) == 0 .and. n >= 0_c_int64_t) &
                    static_size = int(n)
            end select
        end do
    end function alloc_array_result_static_size
