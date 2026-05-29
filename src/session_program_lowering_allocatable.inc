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
