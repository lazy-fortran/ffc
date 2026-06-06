    subroutine lower_allocatable_declaration(node, context, error_msg)
        ! integer, allocatable :: a(:) [, b(:)]
        type(declaration_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        if (node%is_multi_declaration .and. allocated(node%var_names)) then
            do i = 1, size(node%var_names)
                call declare_allocatable_i32_1d(context, node%var_names(i), &
                                                error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        else if (allocated(node%var_name)) then
            call declare_allocatable_i32_1d(context, node%var_name, error_msg)
        else
            error_msg = 'allocatable declaration did not expose a variable name'
        end if
    end subroutine lower_allocatable_declaration

    subroutine lower_allocate_statement(arena, node, context, error_msg)
        ! allocate(a(N)) for one integer 1-D allocatable. N may be a runtime
        ! integer expression. Multi-variable and lower:upper forms are rejected.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: size_index, symbol_index

        if (.not. allocated(node%var_indices) .or. size(node%var_indices) /= 1) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'direct LIRIC session supports allocating exactly '// &
                'one variable per allocate', error_msg)
            return
        end if

        call allocate_target(arena, node, name, size_index, error_msg)
        if (len_trim(error_msg) > 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, trim(error_msg), error_msg)
            return
        end if

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('allocate statement', node%line, &
                node%column, 'allocate target is not an allocatable array: '// &
                trim(name), error_msg)
            return
        end if

        call lower_allocate_i32_1d(arena, size_index, symbol_index, context, &
                                   error_msg)
    end subroutine lower_allocate_statement

    subroutine allocate_target(arena, node, name, size_index, error_msg)
        ! Extract the variable name and the (single) size expression. allocate
        ! parses a(N) either as a call_or_subscript in var_indices or as an
        ! identifier in var_indices with the size in shape_indices.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        character(len=:), allocatable, intent(out) :: name
        integer, intent(out) :: size_index
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        size_index = 0
        if (.not. node_exists(arena, node%var_indices(1))) then
            error_msg = 'allocate target is not an AST node'
            return
        end if
        if (is_identifier(arena, node%var_indices(1))) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. allocated(node%shape_indices) .or. &
                size(node%shape_indices) /= 1) then
                error_msg = 'only one-dimensional allocate is supported'
                return
            end if
            size_index = node%shape_indices(1)
            return
        end if
        select type (v => arena%entries(node%var_indices(1))%node)
        type is (call_or_subscript_node)
            if (.not. allocated(v%name) .or. .not. allocated(v%arg_indices)) then
                error_msg = 'allocate target must name an array with a size'
                return
            end if
            if (size(v%arg_indices) /= 1) then
                error_msg = 'only one-dimensional allocate is supported'
                return
            end if
            name = v%name
            size_index = v%arg_indices(1)
        class default
            error_msg = 'allocate target must be a scalar array name'
        end select
    end subroutine allocate_target

    subroutine lower_allocate_i32_1d(arena, size_index, symbol_index, context, &
                                     error_msg)
        ! data = malloc(N*4); descriptor.data = data; .lower = 1; .upper = N
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: size_index
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: n_i32, bytes_i32, bytes_i64, n_i64, data_ptr
        type(lr_operand_desc_t) :: descriptor

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        call lower_i32_expression(arena, size_index, context, n_i32, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. emit_i32_binary(context%session, LR_OP_MUL, n_i32, &
                i32_immediate(context%session, 4_c_int64_t), bytes_i32, &
                error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, bytes_i32, bytes_i64, &
                error_msg)) return
        if (.not. emit_malloc(context%session, bytes_i64, data_ptr, error_msg)) &
            return
        if (.not. emit_liric_i32_to_i64(context%session, n_i32, n_i64, error_msg)) &
            return

        if (.not. emit_ptr_store(context%session, data_ptr, descriptor, &
                error_msg)) return
        if (.not. emit_i64_store_at(context%session, &
                i64_immediate(context%session, 1_c_int64_t), descriptor, &
                8_c_int64_t, error_msg)) return
        if (.not. emit_i64_store_at(context%session, n_i64, descriptor, &
                16_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocate_i32_1d

    subroutine lower_deallocate_statement(arena, node, context, error_msg)
        ! deallocate(a) for one integer 1-D allocatable.
        type(ast_arena_t), intent(in) :: arena
        type(deallocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index

        if (.not. allocated(node%var_indices) .or. size(node%var_indices) /= 1) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'direct LIRIC session supports deallocating exactly '// &
                'one variable per deallocate', error_msg)
            return
        end if
        if (.not. node_exists(arena, node%var_indices(1))) then
            error_msg = 'deallocate target is not an AST node'
            return
        end if
        if (is_identifier(arena, node%var_indices(1))) then
            call get_identifier_name(arena, node%var_indices(1), name, error_msg)
            if (len_trim(error_msg) > 0) return
        else
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target must be a scalar array name', &
                error_msg)
            return
        end if

        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target is not declared: '//trim(name), &
                error_msg)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            call unsupported_feature_error('deallocate statement', node%line, &
                node%column, 'deallocate target is not an allocatable array: '// &
                trim(name), error_msg)
            return
        end if

        call lower_deallocate_i32_1d(symbol_index, context, error_msg)
    end subroutine lower_deallocate_statement

    subroutine lower_deallocate_i32_1d(symbol_index, context, error_msg)
        ! free(descriptor.data); descriptor.data = 0; descriptor.upper = 0.
        ! free(NULL) is a no-op, so an unallocated variable is handled safely.
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, zero

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, &
                error_msg)) return
        if (.not. emit_free(context%session, data_ptr, error_msg)) return

        zero = i64_immediate(context%session, 0_c_int64_t)
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                0_c_int64_t, error_msg)) return
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                16_c_int64_t, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_deallocate_i32_1d

    logical function is_allocatable_element_ref(node, context) result(is_ref)
        ! a(i) where a is an allocatable and i is a subscript, even when FortFront
        ! did not flag the call_or_subscript node as an array access.
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_ref = .false.
        if (.not. allocated(node%name)) return
        if (.not. allocated(node%arg_indices)) return
        if (size(node%arg_indices) /= 1) return
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) return
        is_ref = context%symbols(symbol_index)%is_allocatable
    end function is_allocatable_element_ref

    subroutine lower_allocatable_element_address(arena, node, symbol_index, &
                                                 context, element_address, error_msg)
        ! Runtime element address of a(i) for a 1-D integer allocatable.
        ! data = descriptor.data (offset 0); lower = descriptor.lower (offset 8).
        ! address = data + (i - lower) * 4 bytes.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: element_address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, data_ptr, lower_i64
        type(lr_operand_desc_t) :: index_i32, index_i64, offset_i64, byte_offset
        integer :: arg_count

        arg_count = 0
        if (allocated(node%arg_indices)) arg_count = size(node%arg_indices)
        if (arg_count /= 1) then
            call unsupported_feature_error('allocatable element access', node%line, &
                node%column, 'direct LIRIC session supports one subscript on a '// &
                'rank-1 integer allocatable', error_msg)
            return
        end if

        call validate_i32_array_subscript(arena, node%arg_indices(1), context, &
                                          error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_i32_expression(arena, node%arg_indices(1), context, index_i32, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return

        descriptor = context%symbols(symbol_index)%allocatable_descriptor_address
        if (.not. emit_ptr_load(context%session, descriptor, data_ptr, error_msg)) &
            return
        if (.not. emit_i64_load_at(context%session, descriptor, 8_c_int64_t, &
                lower_i64, error_msg)) return
        if (.not. emit_liric_i32_to_i64(context%session, index_i32, index_i64, &
                error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_SUB, index_i64, lower_i64, &
                offset_i64, error_msg)) return
        if (.not. emit_i64_binary(context%session, LR_OP_MUL, offset_i64, &
                i64_immediate(context%session, 4_c_int64_t), byte_offset, &
                error_msg)) return
        if (.not. emit_ptr_offset_dyn(context%session, data_ptr, byte_offset, &
                element_address, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_allocatable_element_address

    subroutine declare_allocatable_i32_1d(context, name, error_msg)
        ! Stack-allocate a 24-byte descriptor {i32* data; i64 lower; i64 upper}
        ! and zero it. data == 0 marks the array unallocated. allocate/deallocate
        ! and element access are lowered by later issues.
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor, zero
        integer :: index

        if (find_symbol(context, name) > 0) then
            error_msg = 'duplicate allocatable declaration: '//trim(name)
            return
        end if
        call grow_symbols(context)

        if (.not. emit_alloca_bytes(context%session, &
                i64_immediate(context%session, 24_c_int64_t), descriptor, &
                error_msg)) return
        zero = i64_immediate(context%session, 0_c_int64_t)
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                0_c_int64_t, error_msg)) return
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                8_c_int64_t, error_msg)) return
        if (.not. emit_i64_store_at(context%session, zero, descriptor, &
                16_c_int64_t, error_msg)) return

        index = context%symbol_count + 1
        context%symbols(index)%name = trim(name)
        context%symbols(index)%value_kind = VALUE_I32
        context%symbols(index)%is_allocatable = .true.
        context%symbols(index)%allocatable_descriptor_address = descriptor
        context%symbol_count = index
        call set_empty(error_msg)
    end subroutine declare_allocatable_i32_1d
