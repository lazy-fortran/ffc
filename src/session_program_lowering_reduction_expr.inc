    ! sum() (and friends) over a general array-valued expression argument:
    ! a binary-op combination of arrays/sections (sum(a + b), sum(a(1:3) +
    ! b(1:3))), or a bare call to a contained function that returns an
    ! allocatable array (sum(f())). The plain-identifier and bare-section
    ! arguments stay on the original lower_array_reduction_intrinsic path.

    recursive logical function reduction_arg_extent(arena, node_index, context, &
                                                     vk, extent) result(ok)
        ! True when node_index is a rank-1 array-valued expression of value
        ! kind vk built from identifiers, allocatables, array sections, and
        ! +, -, *, /, ** operators; extent is its element count.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: vk
        integer, intent(out) :: extent
        character(len=:), allocatable :: id_name, id_err, bin_op, bin_err
        type(array_section_info_t) :: info
        integer :: sym, bin_left, bin_right, bin_line, bin_col, left_extent, &
                   right_extent
        logical :: right_is_array

        ok = .false.
        extent = 0
        if (.not. node_exists(arena, node_index)) return

        select type (arg => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, arg, context, info, id_err)
            if (len_trim(id_err) > 0) return
            if (info%result_rank /= 1) return
            if (context%symbols(info%source_index)%value_kind /= vk) return
            extent = int(info%section_extents(info%kept_dims(1)))
            ok = extent > 0
            return
        end select

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            if (len_trim(id_err) > 0) return
            sym = find_symbol(context, id_name)
            if (sym <= 0) return
            if (.not. is_elementwise_array_operand(context, sym)) return
            if (context%symbols(sym)%value_kind /= vk) return
            if (context%symbols(sym)%is_allocatable) then
                extent = context%symbols(sym)%allocatable_static_size
            else
                if (context%symbols(sym)%has_runtime_dim_size(1)) return
                extent = context%symbols(sym)%array_size
            end if
            ok = extent > 0
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, bin_err)
            if (len_trim(bin_err) > 0) return
            select case (trim(bin_op))
            case ('+', '-', '*', '/', '**')
            case default
                return
            end select
            if (.not. reduction_arg_extent(arena, bin_left, context, vk, &
                                           left_extent)) return
            right_is_array = reduction_arg_extent(arena, bin_right, context, vk, &
                                                  right_extent)
            if (right_is_array .and. right_extent /= left_extent) return
            extent = left_extent
            ok = extent > 0
            return
        end if
    end function reduction_arg_extent

    recursive subroutine lower_reduction_arg_element(arena, node_index, &
                                                      linear_index, vk, context, &
                                                      value, error_msg)
        ! Evaluate element linear_index of a reduction argument expression
        ! (identifier, allocatable, array section, or +,-,*,/,** combination)
        ! in value kind vk. Scalar operands broadcast, matching whole-array
        ! arithmetic semantics.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer(c_int64_t), intent(in) :: linear_index
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_section_info_t) :: info
        character(len=:), allocatable :: id_name, bin_op
        integer :: sym, bin_left, bin_right, bin_line, bin_col, opcode
        type(lr_operand_desc_t) :: lhs, rhs

        if (.not. node_exists(arena, node_index)) then
            error_msg = 'reduction expression index does not reference an AST node'
            return
        end if

        select type (arg => arena%entries(node_index)%node)
        type is (array_slice_node)
            call describe_array_section(arena, arg, context, info, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_section_element_from_info(info, linear_index, &
                                                       context, value, error_msg)
            return
        end select

        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, id_name)
            if (sym <= 0) then
                error_msg = 'reduction expression identifier was not declared: '// &
                            trim(id_name)
                return
            end if
            if (is_elementwise_array_operand(context, sym)) then
                call load_array_linear_element(context, sym, linear_index, value, &
                                               error_msg)
            else
                call lower_reduction_scalar(arena, node_index, vk, context, &
                                            value, error_msg)
            end if
            return
        end if

        if (is_literal(arena, node_index)) then
            call lower_reduction_scalar(arena, node_index, vk, context, value, &
                                        error_msg)
            return
        end if

        if (is_binary_op(arena, node_index)) then
            call get_binary_op_info(arena, node_index, bin_op, bin_left, &
                                    bin_right, bin_line, bin_col, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_reduction_arg_element(arena, bin_left, linear_index, vk, &
                                             context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            select case (vk)
            case (VALUE_F32, VALUE_F64)
                call lower_reduction_arg_element(arena, bin_right, linear_index, &
                                                 vk, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                if (trim(bin_op) == '**') then
                    if (vk == VALUE_F32) then
                        if (.not. emit_liric_f32_call(context%session, 'powf', &
                                [lhs, rhs], value, error_msg)) return
                    else
                        if (.not. emit_liric_f64_call(context%session, 'pow', &
                                [lhs, rhs], value, error_msg)) return
                    end if
                    return
                end if
                call real_opcode(bin_op, bin_line, bin_col, opcode, error_msg)
                if (len_trim(error_msg) > 0) return
                if (vk == VALUE_F32) then
                    if (.not. emit_liric_f32_binary(context%session, opcode, lhs, &
                                                    rhs, value, error_msg)) return
                else
                    if (.not. emit_liric_f64_binary(context%session, opcode, lhs, &
                                                    rhs, value, error_msg)) return
                end if
            case default
                if (trim(bin_op) == '**') then
                    call lower_i32_pow(arena, bin_right, bin_line, bin_col, &
                                       context, lhs, value, error_msg)
                    return
                end if
                call lower_reduction_arg_element(arena, bin_right, linear_index, &
                                                 vk, context, rhs, error_msg)
                if (len_trim(error_msg) > 0) return
                call integer_opcode(bin_op, opcode, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, &
                                          value, error_msg)) return
            end select
            return
        end if

        error_msg = 'reduction argument expression is not supported'
    end subroutine lower_reduction_arg_element

    subroutine lower_reduction_scalar(arena, node_index, vk, context, value, &
                                      error_msg)
        ! A scalar operand (literal or scalar variable) broadcasting across
        ! every element of a reduction argument expression.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        select case (vk)
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        end select
    end subroutine lower_reduction_scalar

    subroutine lower_general_expr_reduction(arena, arg_index, extent, vk, &
                                            context, value, error_msg, &
                                            reduction_name)
        ! sum() (only) over a binary-op array expression argument, unrolled
        ! over the compile-time extent determined by reduction_arg_extent.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        integer, intent(in) :: extent
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: reduction_name
        type(lr_operand_desc_t) :: candidate, next_value
        integer :: i

        if (trim(reduction_name) /= 'sum') then
            error_msg = trim(reduction_name)//' of a general array expression '// &
                        'is not supported'
            return
        end if

        select case (vk)
        case (VALUE_F32)
            value = liric_f32_immediate(context%session, 0.0_c_float)
            do i = 0, extent - 1
                call lower_reduction_arg_element(arena, arg_index, &
                        int(i, c_int64_t), vk, context, candidate, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, &
                        value, candidate, next_value, error_msg)) return
                value = next_value
            end do
        case (VALUE_F64)
            value = liric_f64_immediate(context%session, 0.0_c_double)
            do i = 0, extent - 1
                call lower_reduction_arg_element(arena, arg_index, &
                        int(i, c_int64_t), vk, context, candidate, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, &
                        value, candidate, next_value, error_msg)) return
                value = next_value
            end do
        case default
            value = i32_immediate(context%session, 0_c_int64_t)
            do i = 0, extent - 1
                call lower_reduction_arg_element(arena, arg_index, &
                        int(i, c_int64_t), vk, context, candidate, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, value, &
                        candidate, next_value, error_msg)) return
                value = next_value
            end do
        end select
        call set_empty(error_msg)
    end subroutine lower_general_expr_reduction

    subroutine alloc_array_result_kind(arena, context, name, elem_kind, rank, ok)
        ! Element scalar kind and rank of a contained function's allocatable
        ! array result, looked up by function name.
        type(ast_arena_t), intent(in) :: arena
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer, intent(out) :: elem_kind
        integer, intent(out) :: rank
        logical, intent(out) :: ok
        type(function_def_node), pointer :: fn_node
        integer :: i, node_index

        ok = .false.
        elem_kind = VALUE_I32
        rank = 0
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
        call alloc_array_result_info(fn_node, context, ok, elem_kind, rank)
    end subroutine alloc_array_result_kind

    subroutine alloc_array_result_call_info(arena, value_index, context, &
                                            elem_kind, rank, ok)
        ! Element scalar kind and rank of a bare call node's allocatable array
        ! result, or ok=.false. when value_index is not such a call.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: elem_kind
        integer, intent(out) :: rank
        logical, intent(out) :: ok

        ok = .false.
        elem_kind = VALUE_I32
        rank = 0
        if (.not. node_exists(arena, value_index)) return
        select type (v => arena%entries(value_index)%node)
        type is (call_or_subscript_node)
            if (allocated(v%name)) &
                call alloc_array_result_kind(arena, context, v%name, elem_kind, &
                                             rank, ok)
        end select
    end subroutine alloc_array_result_call_info

    subroutine lower_alloc_call_reduction(arena, node, context, value, &
                                          error_msg, reduction_name)
        ! sum() (only) of a bare call to a contained function returning an
        ! allocatable array: materialise the result into a zeroed temporary
        ! descriptor, reduce over its compile-time extent, then free it.
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: reduction_name
        type(call_or_subscript_node) :: call_node
        type(lr_operand_desc_t), allocatable :: user_args(:), all_args(:)
        integer, allocatable :: copyback_indices(:)
        character(len=:), allocatable :: func_name
        type(lr_operand_desc_t) :: temp_desc, zero, data_ptr, addr, candidate, &
                                   next_value
        integer :: elem_kind, rank, static_size, i
        integer(c_int64_t) :: stride
        logical :: ok

        if (trim(reduction_name) /= 'sum') then
            error_msg = trim(reduction_name)//' of a function-call array '// &
                        'result is not supported'
            return
        end if

        select type (v => arena%entries(node%arg_indices(1))%node)
        type is (call_or_subscript_node)
            call_node = v
        class default
            error_msg = 'expected function call for array reduction argument'
            return
        end select
        func_name = call_node%name

        call alloc_array_result_kind(arena, context, func_name, elem_kind, rank, ok)
        if (.not. ok .or. rank /= 1) then
            error_msg = trim(reduction_name)//' argument function result is '// &
                        'not a rank-1 allocatable array'
            return
        end if

        static_size = alloc_array_result_static_size(arena, context, func_name)
        if (static_size <= 0) then
            error_msg = trim(reduction_name)//' requires a constant '// &
                        'allocatable extent for '//trim(func_name)
            return
        end if

        if (allocated(call_node%arg_indices)) then
            call prepare_reference_args(arena, call_node%arg_indices, context, &
                                        VALUE_I32, func_name, user_args, &
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

        if (.not. emit_ptr_load(context%session, temp_desc, data_ptr, error_msg)) &
            return
        stride = allocatable_elem_size(elem_kind)

        select case (elem_kind)
        case (VALUE_F32)
            value = liric_f32_immediate(context%session, 0.0_c_float)
            do i = 0, static_size - 1
                if (.not. emit_ptr_offset(context%session, data_ptr, &
                        int(i, c_int64_t) * stride, addr, error_msg)) return
                if (.not. emit_allocatable_element_load(context, elem_kind, addr, &
                        candidate, error_msg)) return
                if (.not. emit_liric_f32_binary(context%session, LR_OP_FADD, &
                        value, candidate, next_value, error_msg)) return
                value = next_value
            end do
        case (VALUE_F64)
            value = liric_f64_immediate(context%session, 0.0_c_double)
            do i = 0, static_size - 1
                if (.not. emit_ptr_offset(context%session, data_ptr, &
                        int(i, c_int64_t) * stride, addr, error_msg)) return
                if (.not. emit_allocatable_element_load(context, elem_kind, addr, &
                        candidate, error_msg)) return
                if (.not. emit_liric_f64_binary(context%session, LR_OP_FADD, &
                        value, candidate, next_value, error_msg)) return
                value = next_value
            end do
        case default
            value = i32_immediate(context%session, 0_c_int64_t)
            do i = 0, static_size - 1
                if (.not. emit_ptr_offset(context%session, data_ptr, &
                        int(i, c_int64_t) * stride, addr, error_msg)) return
                if (.not. emit_allocatable_element_load(context, elem_kind, addr, &
                        candidate, error_msg)) return
                if (.not. emit_i32_binary(context%session, LR_OP_ADD, value, &
                        candidate, next_value, error_msg)) return
                value = next_value
            end do
        end select

        if (.not. emit_free(context%session, data_ptr, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_alloc_call_reduction
