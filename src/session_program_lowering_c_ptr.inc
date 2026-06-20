    function null_ptr_operand(context) result(op)
        ! A null c_ptr value: an 8-byte pointer-typed zero.
        type(lowering_context_t), intent(in) :: context
        type(lr_operand_desc_t) :: op

        op%kind = LR_OP_KIND_IMM_I64
        op%payload = 0_c_int64_t
        op%typ = lr_type_ptr_s(context%session%handle)
        op%global_offset = 0_c_int64_t
    end function null_ptr_operand

    subroutine define_c_ptr_symbol(context, name, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate c_ptr declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)
        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_C_PTR
        context%symbols(index)%value = null_ptr_operand(context)
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine define_c_ptr_symbol

    subroutine call_or_subscript_arg_indices(arena, node_index, arg_indices, &
                                              error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, allocatable, intent(out) :: arg_indices(:)
        character(len=:), allocatable, intent(out) :: error_msg

        allocate (arg_indices(0))
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'call index does not reference an AST node'
            return
        end if
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (allocated(node%arg_indices)) then
                if (size(node%arg_indices) > 0) arg_indices = node%arg_indices
            end if
            call set_empty(error_msg)
        class default
            error_msg = 'expected a call expression'
        end select
    end subroutine call_or_subscript_arg_indices

    logical function is_named_call(arena, node_index, name)
        ! A scalar call_or_subscript to the intrinsic named `name`.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: name

        is_named_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. node%is_array_access .and. allocated(node%name)) &
                is_named_call = same_name(node%name, name)
        end select
    end function is_named_call

    recursive subroutine lower_c_ptr_expression(arena, node_index, context, &
                                                value, error_msg)
        ! Lower an expression that yields a c_ptr: c_null_ptr, a c_ptr variable,
        ! or c_loc(x).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'c_ptr expression does not reference an AST node'
            return
        end if
        if (is_named_call(arena, node_index, 'c_loc')) then
            block
                integer, allocatable :: c_loc_args(:)
                call call_or_subscript_arg_indices(arena, node_index, &
                                                   c_loc_args, error_msg)
                if (len_trim(error_msg) > 0) return
                call lower_c_loc(arena, c_loc_args, context, value, error_msg)
            end block
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (same_name(name, 'c_null_ptr')) then
                value = null_ptr_operand(context)
                call set_empty(error_msg)
                return
            end if
            symbol_index = find_symbol(context, name)
            if (symbol_index <= 0) then
                error_msg = 'c_ptr identifier was not declared: '//trim(name)
                return
            end if
            if (context%symbols(symbol_index)%value_kind /= VALUE_C_PTR) then
                error_msg = 'expected a c_ptr value: '//trim(name)
                return
            end if
            if (context%symbols(symbol_index)%has_address .and. &
                context%symbols(symbol_index)%is_reference) then
                if (.not. emit_ptr_load(context%session, &
                    context%symbols(symbol_index)%address, value, error_msg)) &
                    return
            else
                value = context%symbols(symbol_index)%value
                call set_empty(error_msg)
            end if
            return
        end if
        error_msg = 'direct LIRIC session c_ptr expression supports c_null_ptr, '// &
                    'a c_ptr variable, or c_loc(x)'
    end subroutine lower_c_ptr_expression

    subroutine lower_c_loc(arena, arg_indices, context, value, error_msg)
        ! c_loc(x): the address of x. A scalar local is materialised in memory
        ! (alloca + store) to obtain a stable address.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'c_loc requires exactly one argument'
            return
        end if
        if (size(arg_indices) /= 1) then
            error_msg = 'c_loc requires exactly one argument'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = 'c_loc argument must be a scalar variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'c_loc argument was not declared: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%has_address) then
            value = context%symbols(symbol_index)%address
            call set_empty(error_msg)
            return
        end if
        call make_reference_argument(context, &
                                     context%symbols(symbol_index)%value_kind, &
                                     context%symbols(symbol_index)%value, value, &
                                     error_msg)
    end subroutine lower_c_loc

    subroutine lower_c_associated(arena, arg_indices, context, value, error_msg)
        ! c_associated(p): 1 when p is non-null, 0 otherwise (an i32 boolean).
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ptr_value

        call set_empty(error_msg)
        if (.not. allocated(arg_indices)) then
            error_msg = 'c_associated requires exactly one argument'
            return
        end if
        if (size(arg_indices) /= 1) then
            error_msg = 'c_associated requires exactly one argument'
            return
        end if
        call lower_c_ptr_expression(arena, arg_indices(1), context, ptr_value, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, ptr_value, &
                null_ptr_operand(context), value, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_c_associated

    subroutine lower_c_f_pointer(arena, arg_indices, context, error_msg)
        ! c_f_pointer(cptr, fptr): the MVP stores cptr into fptr's c_ptr slot.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: ptr_value
        character(len=:), allocatable :: name
        integer :: symbol_index

        if (.not. allocated(arg_indices)) then
            error_msg = 'c_f_pointer requires two arguments'
            return
        end if
        if (size(arg_indices) < 2) then
            error_msg = 'c_f_pointer requires at least two arguments'
            return
        end if
        if (size(arg_indices) > 2) then
            error_msg = 'c_f_pointer with a shape argument is not yet supported'
            return
        end if
        call lower_c_ptr_expression(arena, arg_indices(1), context, ptr_value, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. is_identifier(arena, arg_indices(2))) then
            error_msg = 'c_f_pointer target must be a scalar pointer variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(2), name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'c_f_pointer target was not declared: '//trim(name)
            return
        end if
        ! Bind the Fortran pointer to the C address: the pointer adopts cp's
        ! address as its storage, so a later read of p dereferences that address
        ! (emit_*_load) and yields the value the C pointer refers to.
        context%symbols(symbol_index)%value = ptr_value
        context%symbols(symbol_index)%address = ptr_value
        context%symbols(symbol_index)%has_address = .true.
        context%symbols(symbol_index)%is_reference = .true.
        context%symbols(symbol_index)%is_pointer = .true.
        context%symbols(symbol_index)%is_associated = .true.
        call set_empty(error_msg)
    end subroutine lower_c_f_pointer
