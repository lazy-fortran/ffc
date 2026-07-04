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

    ! Element addresses in the re/im component arrays for a compile-time linear
    ! index (re array base = element_address, im array base = address).
    subroutine complex_array_linear_addresses(context, symbol_index, &
                                               linear_index, vk, re_addr, &
                                               im_addr, error_msg)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: symbol_index
        integer, intent(in) :: linear_index
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: re_addr, im_addr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: idx

        call set_empty(error_msg)
        idx = i32_immediate(context%session, int(linear_index, c_int64_t))
        if (vk == VALUE_C4) then
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, idx, &
                    re_addr, error_msg)) return
            if (.not. emit_f32_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%address, idx, &
                    im_addr, error_msg)) return
        else
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%element_address, idx, &
                    re_addr, error_msg)) return
            if (.not. emit_f64_array_element_addr(context%session, &
                    context%symbols(symbol_index)%array_size, &
                    context%symbols(symbol_index)%address, idx, &
                    im_addr, error_msg)) return
        end if
    end subroutine complex_array_linear_addresses

    ! True when node is a bare identifier naming a fixed-size complex array;
    ! returns its symbol index for elementwise loads.
    logical function operand_is_whole_complex_array(arena, node_index, context, &
                                                     sym) result(is_arr)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: sym
        character(len=:), allocatable :: nm, err

        is_arr = .false.
        sym = 0
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, nm, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, nm)
        if (sym <= 0) return
        if (.not. context%symbols(sym)%is_array) then
            sym = 0
            return
        end if
        if (context%symbols(sym)%value_kind == VALUE_C4 .or. &
            context%symbols(sym)%value_kind == VALUE_C8) is_arr = .true.
        if (.not. is_arr) sym = 0
    end function operand_is_whole_complex_array

    ! A scalar/self-indexed complex leaf broadcasts per element; decline forms
    ! that fold a whole complex array we cannot index elementwise here (e.g.
    ! conjg(a) over a whole array).
    logical function leaf_folds_whole_complex_array(arena, node_index, context) &
            result(folds)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, allocatable :: arg_indices(:)
        character(len=:), allocatable :: err
        integer :: k, sym

        folds = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (n%is_array_access) return
        class default
            return
        end select
        call call_or_subscript_arg_indices(arena, node_index, arg_indices, err)
        if (len_trim(err) > 0) return
        do k = 1, size(arg_indices)
            if (operand_is_whole_complex_array(arena, arg_indices(k), context, &
                                               sym)) then
                folds = .true.
                return
            end if
        end do
    end function leaf_folds_whole_complex_array

    ! Evaluate the RHS complex value of a whole-array assignment at one linear
    ! element index. Binary +/-/*// recurse; a whole complex-array operand loads
    ! its element; scalar/self-indexed leaves broadcast via eval_cN_rhs.
    recursive subroutine eval_complex_array_element(arena, node_index, &
                                                    target_sym, linear_index, &
                                                    context, vk, re_val, im_val, &
                                                    error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: target_sym
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: re_val, im_val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lre, lim, rre, rim, re_addr, im_addr
        character(len=:), allocatable :: op, kind_tag
        integer :: li, ri, opc, ln, cl, sym

        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'complex whole-array expression references no AST node'
            return
        end if

        kind_tag = 'complex(4)'
        if (vk == VALUE_C8) kind_tag = 'complex(8)'

        if (is_binary_op(arena, node_index)) then
            call complex_binary_op(arena, node_index, op, li, ri, opc, ln, cl, &
                                   kind_tag, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_complex_array_element(arena, li, target_sym, linear_index, &
                                            context, vk, lre, lim, error_msg)
            if (len_trim(error_msg) > 0) return
            call eval_complex_array_element(arena, ri, target_sym, linear_index, &
                                            context, vk, rre, rim, error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk == VALUE_C4) then
                call complex_combine_c4(context, op, opc, lre, lim, rre, rim, &
                                        re_val, im_val, error_msg)
            else
                call complex_combine_c8(context, op, opc, lre, lim, rre, rim, &
                                        re_val, im_val, error_msg)
            end if
            return
        end if

        if (operand_is_whole_complex_array(arena, node_index, context, sym)) then
            if (context%symbols(sym)%value_kind /= vk) then
                call unsupported_feature_error('complex whole-array assignment', &
                    get_node_line(arena, node_index), &
                    get_node_column(arena, node_index), &
                    'operand kind does not match the assignment target', &
                    error_msg)
                return
            end if
            call ensure_array_shapes_match(context, target_sym, sym, error_msg)
            if (len_trim(error_msg) > 0) return
            call complex_array_linear_addresses(context, sym, linear_index, vk, &
                                                re_addr, im_addr, error_msg)
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
            return
        end if

        if (leaf_folds_whole_complex_array(arena, node_index, context)) then
            call unsupported_feature_error('complex whole-array assignment', &
                get_node_line(arena, node_index), &
                get_node_column(arena, node_index), &
                'direct LIRIC session supports elementwise +, -, *, / over '// &
                'complex arrays with scalar or complex-array operands only', &
                error_msg)
            return
        end if

        if (vk == VALUE_C4) then
            call eval_c4_rhs(arena, node_index, context, re_val, im_val, error_msg)
        else
            call eval_c8_rhs(arena, node_index, context, re_val, im_val, error_msg)
        end if
    end subroutine eval_complex_array_element

    ! Whole-array assignment to a fixed-size complex array: evaluate the RHS
    ! elementwise and store each re/im pair into the component arrays.
    subroutine lower_complex_array_whole_assignment(arena, node, symbol_index, &
                                                     context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(assignment_node), intent(in) :: node
        integer, intent(in) :: symbol_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: re_val, im_val, re_addr, im_addr
        integer :: i, array_size, vk

        call set_empty(error_msg)
        vk = context%symbols(symbol_index)%value_kind
        array_size = context%symbols(symbol_index)%array_size
        if (array_size <= 0) then
            error_msg = 'complex whole-array assignment requires a positive '// &
                        'array size'
            return
        end if
        do i = 1, array_size
            call eval_complex_array_element(arena, node%value_index, &
                                            symbol_index, i - 1, context, vk, &
                                            re_val, im_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call complex_array_linear_addresses(context, symbol_index, i - 1, &
                                                vk, re_addr, im_addr, error_msg)
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
        end do
        call set_empty(error_msg)
    end subroutine lower_complex_array_whole_assignment
