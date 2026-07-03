    subroutine lower_allocated(arena, arg_indices, context, value, error_msg)
        ! allocated(x), W11. A scalar allocatable folds the compile-time
        ! association state (has_address), mirroring associated()'s
        ! straight-line-only limitation. An array allocatable gets a real
        ! runtime check: its descriptor's data pointer (offset 0) compared
        ! against null.
        type(ast_arena_t), intent(in) :: arena
        integer, allocatable, intent(in) :: arg_indices(:)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: symbol_index
        type(lr_operand_desc_t) :: data_ptr

        call set_empty(error_msg)
        if (.not. allocated(arg_indices) .or. size(arg_indices) /= 1) then
            error_msg = 'allocated requires exactly one argument'
            return
        end if
        if (.not. is_identifier(arena, arg_indices(1))) then
            error_msg = 'allocated argument must be an allocatable variable'
            return
        end if
        call get_identifier_name(arena, arg_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol(context, name)
        if (symbol_index <= 0) then
            error_msg = 'allocated argument was not declared: '//trim(name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_allocatable) then
            error_msg = 'allocated argument is not allocatable: '//trim(name)
            return
        end if
        if (context%symbols(symbol_index)%array_rank == 0) then
            if (context%symbols(symbol_index)%has_address) then
                value = i32_immediate(context%session, 1_c_int64_t)
            else
                value = i32_immediate(context%session, 0_c_int64_t)
            end if
            return
        end if
        if (.not. emit_ptr_load(context%session, &
                context%symbols(symbol_index)%allocatable_descriptor_address, &
                data_ptr, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, data_ptr, &
                null_ptr_operand(context), value, error_msg)) return
    end subroutine lower_allocated
