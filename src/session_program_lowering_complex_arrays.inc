    ! Fixed-size complex(4)/complex(8) array support: element_address holds
    ! the real-part [N x f32|f64] array base pointer, address holds the
    ! matching imaginary-part array base pointer (define_declared_array_symbol
    ! in session_program_lowering_arrays.inc allocates both). Element access
    ! reuses the generic rank-1/2 linear-index GEP machinery, applied twice
    ! (once per component array) with the same subscripts.

    subroutine complex_array_element_addresses(arena, node, context, vk, &
                                                re_address, im_address, &
                                                error_msg, feature)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: re_address, im_address
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in), optional :: feature
        character(len=:), allocatable :: diagnostic_feature
        integer :: symbol_index
        type(lr_operand_desc_t) :: linear_index

        if (.not. allocated(node%name)) then
            error_msg = 'complex array access requires an array name'
            return
        end if
        symbol_index = find_symbol(context, node%name)
        if (symbol_index <= 0) then
            error_msg = 'complex array identifier was not declared: '// &
                        trim(node%name)
            return
        end if
        if (.not. context%symbols(symbol_index)%is_array) then
            error_msg = trim(node%name)//' is not an array'
            return
        end if

        diagnostic_feature = 'complex array expression'
        if (present(feature)) diagnostic_feature = feature
        call lower_array_element_linear_index(arena, node, context, symbol_index, &
                                              linear_index, error_msg, &
                                              diagnostic_feature)
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_C4) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, re_address, error_msg)) then
                error_msg = 'c4_arr_gep_re: '//error_msg
                return
            end if
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%address, &
                    linear_index, im_address, error_msg)) then
                error_msg = 'c4_arr_gep_im: '//error_msg
                return
            end if
        else
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, &
                    linear_index, re_address, error_msg)) then
                error_msg = 'c8_arr_gep_re: '//error_msg
                return
            end if
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%address, &
                    linear_index, im_address, error_msg)) then
                error_msg = 'c8_arr_gep_im: '//error_msg
                return
            end if
        end if
        call set_empty(error_msg)
    end subroutine complex_array_element_addresses

    subroutine load_complex_array_element(arena, node, context, vk, re_val, &
                                          im_val, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_addr, im_addr

        call complex_array_element_addresses(arena, node, context, vk, re_addr, &
                                             im_addr, error_msg, &
                                             'complex array element')
        if (len_trim(error_msg) > 0) return
        if (vk == VALUE_C4) then
            if (.not. emit_liric_f32_load(context%session, re_addr, re_val, &
                                          error_msg)) return
            if (.not. emit_liric_f32_load(context%session, im_addr, im_val, &
                                          error_msg)) return
        else
            if (.not. emit_liric_f64_load(context%session, re_addr, re_val, &
                                          error_msg)) return
            if (.not. emit_liric_f64_load(context%session, im_addr, im_val, &
                                          error_msg)) return
        end if
    end subroutine load_complex_array_element

    subroutine lower_complex_array_element_assignment(arena, node, target, &
                                                       context, sym_idx, vk, &
                                                       error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        type(call_or_subscript_node), intent(in) :: target
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym_idx
        integer, intent(in) :: vk
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val, re_addr, im_addr

        call set_empty(error_msg)
        if (sym_idx <= 0) then
            error_msg = 'complex array assignment target was not declared'
            return
        end if

        if (vk == VALUE_C4) then
            call eval_c4_rhs(arena, node%value_index, context, re_val, im_val, &
                             error_msg)
        else
            call eval_c8_rhs(arena, node%value_index, context, re_val, im_val, &
                             error_msg)
        end if
        if (len_trim(error_msg) > 0) return

        call complex_array_element_addresses(arena, target, context, vk, &
                                             re_addr, im_addr, error_msg, &
                                             'complex array assignment target')
        if (len_trim(error_msg) > 0) return

        if (vk == VALUE_C4) then
            if (.not. emit_liric_f32_store(context%session, re_val, re_addr, &
                                           error_msg)) return
            if (.not. emit_liric_f32_store(context%session, im_val, im_addr, &
                                           error_msg)) return
        else
            if (.not. emit_liric_f64_store(context%session, re_val, re_addr, &
                                           error_msg)) return
            if (.not. emit_liric_f64_store(context%session, im_val, im_addr, &
                                           error_msg)) return
        end if
        call set_empty(error_msg)
    end subroutine lower_complex_array_element_assignment
