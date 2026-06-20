    ! Complex assignment and load helpers.
    ! Storage: re in symbol%address (f32 or f64 alloca), im in element_address.
    ! symbol%value holds the last-loaded re SSA value (used as fallback).

    subroutine lower_c4_assignment(arena, rhs_index, sym_idx, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float, c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index, sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val
        character(len=:), allocatable :: src_name
        integer :: src_idx
        real(c_double) :: re_f64, im_f64

        call set_empty(error_msg)
        if (.not. node_exists(arena, rhs_index)) then
            error_msg = 'complex(4) assignment: rhs index does not exist'
            return
        end if

        select type (rhs => arena%entries(rhs_index)%node)
        type is (complex_literal_node)
            ! (re_expr, im_expr) literal
            call lower_f32_expression(arena, rhs%real_index, context, re_val, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f32_expression(arena, rhs%imag_index, context, im_val, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            ! Identifier: copy from another complex(4) variable.
            if (is_identifier(arena, rhs_index)) then
                call get_identifier_name(arena, rhs_index, src_name, error_msg)
                if (len_trim(error_msg) > 0) return
                src_idx = find_symbol(context, src_name)
                if (src_idx <= 0) then
                    error_msg = 'complex(4) rhs not declared: '//trim(src_name)
                    return
                end if
                if (context%symbols(src_idx)%value_kind /= VALUE_C4) then
                    error_msg = 'complex(4) rhs is not complex(4): '//trim(src_name)
                    return
                end if
                if (.not. emit_liric_f32_load(context%session, &
                        context%symbols(src_idx)%address, re_val, error_msg)) return
                if (.not. emit_liric_f32_load(context%session, &
                        context%symbols(src_idx)%element_address, im_val, &
                        error_msg)) return
            else
                error_msg = 'complex(4) assignment: unsupported rhs form'
                return
            end if
        end select

        if (.not. emit_liric_f32_store(context%session, re_val, &
                context%symbols(sym_idx)%address, error_msg)) return
        if (.not. emit_liric_f32_store(context%session, im_val, &
                context%symbols(sym_idx)%element_address, error_msg)) return
        context%symbols(sym_idx)%value = re_val
        call set_empty(error_msg)
    end subroutine lower_c4_assignment

    subroutine lower_c8_assignment(arena, rhs_index, sym_idx, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_float, c_double
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: rhs_index, sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val
        character(len=:), allocatable :: src_name
        integer :: src_idx

        call set_empty(error_msg)
        if (.not. node_exists(arena, rhs_index)) then
            error_msg = 'complex(8) assignment: rhs index does not exist'
            return
        end if

        select type (rhs => arena%entries(rhs_index)%node)
        type is (complex_literal_node)
            call lower_f64_expression(arena, rhs%real_index, context, re_val, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_f64_expression(arena, rhs%imag_index, context, im_val, &
                                      error_msg)
            if (len_trim(error_msg) > 0) return
        class default
            if (is_identifier(arena, rhs_index)) then
                call get_identifier_name(arena, rhs_index, src_name, error_msg)
                if (len_trim(error_msg) > 0) return
                src_idx = find_symbol(context, src_name)
                if (src_idx <= 0) then
                    error_msg = 'complex(8) rhs not declared: '//trim(src_name)
                    return
                end if
                if (context%symbols(src_idx)%value_kind /= VALUE_C8) then
                    error_msg = 'complex(8) rhs is not complex(8): '//trim(src_name)
                    return
                end if
                if (.not. emit_liric_f64_load(context%session, &
                        context%symbols(src_idx)%address, re_val, error_msg)) return
                if (.not. emit_liric_f64_load(context%session, &
                        context%symbols(src_idx)%element_address, im_val, &
                        error_msg)) return
            else
                error_msg = 'complex(8) assignment: unsupported rhs form'
                return
            end if
        end select

        if (.not. emit_liric_f64_store(context%session, re_val, &
                context%symbols(sym_idx)%address, error_msg)) return
        if (.not. emit_liric_f64_store(context%session, im_val, &
                context%symbols(sym_idx)%element_address, error_msg)) return
        context%symbols(sym_idx)%value = re_val
        call set_empty(error_msg)
    end subroutine lower_c8_assignment

    subroutine lower_print_complex4(sym_idx, context, error_msg)
        integer, intent(in) :: sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val

        if (.not. emit_liric_f32_load(context%session, &
                context%symbols(sym_idx)%address, re_val, error_msg)) return
        if (.not. emit_liric_f32_load(context%session, &
                context%symbols(sym_idx)%element_address, im_val, error_msg)) return
        if (.not. emit_complex4_print_call(context%session, re_val, im_val, &
                                           error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_print_complex4

    logical function is_contained_complex_function(context, name) result(is_c)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: kind

        kind = contained_function_kind(context, name)
        is_c = kind == VALUE_C4 .or. kind == VALUE_C8
    end function is_contained_complex_function

    subroutine lower_print_complex_result_call(arena, node_index, context, &
                                               error_msg)
        ! print *, cfunc(): emit the sret call into a scratch complex buffer,
        ! then load and print re/im. The callee writes re@+0, im@+4 (c4)/+8 (c8).
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: buf, re_addr, im_addr, re_val, im_val
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        integer :: kind, i
        integer(c_int64_t) :: nbytes, im_offset

        call set_empty(error_msg)
        select type (call_node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            func_name = trim(call_node%name)
            kind = contained_function_kind(context, func_name)
            if (kind == VALUE_C8) then
                nbytes = 16_c_int64_t
                im_offset = 8_c_int64_t
            else
                nbytes = 8_c_int64_t
                im_offset = 4_c_int64_t
            end if
            if (.not. emit_alloca_bytes(context%session, &
                    i64_immediate(context%session, nbytes), buf, error_msg)) return
            if (allocated(call_node%arg_indices)) then
                call prepare_reference_args(arena, call_node%arg_indices, context, &
                                            VALUE_I32, func_name, user_args, &
                                            copyback_indices, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                allocate (user_args(0)); allocate (copyback_indices(0))
            end if
            allocate (all_args(size(user_args) + 1))
            all_args(1) = buf
            do i = 1, size(user_args)
                all_args(i + 1) = user_args(i)
            end do
            if (.not. emit_void_call(context%session, func_name, all_args, &
                                     error_msg)) return
            call copy_back_reference_args(context, user_args, copyback_indices, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_ptr_offset(context%session, buf, 0_c_int64_t, &
                                      re_addr, error_msg)) return
            if (.not. emit_ptr_offset(context%session, buf, im_offset, &
                                      im_addr, error_msg)) return
            if (kind == VALUE_C8) then
                if (.not. emit_liric_f64_load(context%session, re_addr, re_val, &
                                              error_msg)) return
                if (.not. emit_liric_f64_load(context%session, im_addr, im_val, &
                                              error_msg)) return
                if (.not. emit_complex8_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
            else
                if (.not. emit_liric_f32_load(context%session, re_addr, re_val, &
                                              error_msg)) return
                if (.not. emit_liric_f32_load(context%session, im_addr, im_val, &
                                              error_msg)) return
                if (.not. emit_complex4_print_call(context%session, re_val, &
                                                   im_val, error_msg)) return
            end if
        class default
            error_msg = 'expected function call for complex result print'
        end select
    end subroutine lower_print_complex_result_call

    subroutine lower_print_complex8(sym_idx, context, error_msg)
        integer, intent(in) :: sym_idx
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val

        if (.not. emit_liric_f64_load(context%session, &
                context%symbols(sym_idx)%address, re_val, error_msg)) return
        if (.not. emit_liric_f64_load(context%session, &
                context%symbols(sym_idx)%element_address, im_val, error_msg)) return
        if (.not. emit_complex8_print_call(context%session, re_val, im_val, &
                                           error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_print_complex8
