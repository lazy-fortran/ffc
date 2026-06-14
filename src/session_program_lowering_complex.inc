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
