submodule (session_program_lowering_impl) session_program_lowering_reduction_expr
    implicit none
contains

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
        logical :: left_is_array, right_is_array

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
            sym = find_symbol_compat(context, id_name)
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
            left_is_array = reduction_arg_extent(arena, bin_left, context, vk, &
                                                 left_extent)
            right_is_array = reduction_arg_extent(arena, bin_right, context, vk, &
                                                  right_extent)
            if (.not. left_is_array .and. .not. reduction_arg_is_scalar(arena, &
                    bin_left, context)) return
            if (.not. right_is_array .and. .not. reduction_arg_is_scalar(arena, &
                    bin_right, context)) return
            if (.not. left_is_array .and. .not. right_is_array) return
            if (.not. left_is_array) then
                extent = right_extent
                ok = extent > 0
                return
            end if
            if (right_is_array .and. right_extent /= left_extent) return
            extent = left_extent
            ok = extent > 0
            return
        end if
    end function reduction_arg_extent

    recursive logical function reduction_arg_is_scalar(arena, node_index, &
                                                       context) result(is_scalar)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, error_msg, op
        integer :: sym, left, right, line, column

        is_scalar = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_literal(arena, node_index)) then
            is_scalar = .true.
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol_compat(context, name)
            if (sym > 0) is_scalar = .not. context%symbols(sym)%is_array
            return
        end if
        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, op, left, right, line, column, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        is_scalar = reduction_arg_is_scalar(arena, left, context) .and. &
                    reduction_arg_is_scalar(arena, right, context)
    end function reduction_arg_is_scalar

    recursive logical function reduction_expression_has_kind(arena, node_index, &
                                                             context, vk) result(has)
        !! True when an array-valued reduction expression contains an array of
        !! value kind vk. This predicate also covers runtime-shaped arrays,
        !! whose extent cannot be represented by reduction_arg_extent().
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: vk
        character(len=:), allocatable :: name, err, op
        integer :: sym, left, right, line, column

        has = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, err)
            if (len_trim(err) > 0) return
            sym = find_symbol_compat(context, name)
            if (sym > 0) then
                has = context%symbols(sym)%is_array .and. &
                      context%symbols(sym)%value_kind == vk
            end if
            return
        end if

        if (reduction_expression_is_abs_call(arena, node_index)) then
            select type (abs_node => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                has = reduction_expression_has_kind(arena, &
                    abs_node%arg_indices(1), context, vk)
            end select
            return
        end if

        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, op, left, right, line, column, err)
        if (len_trim(err) > 0) return
        select case (trim(op))
        case ('+', '-', '*', '/', '**')
            has = reduction_expression_has_kind(arena, left, context, vk) .or. &
                  reduction_expression_has_kind(arena, right, context, vk)
        end select
    end function reduction_expression_has_kind

    logical function reduction_expression_is_abs_call(arena, node_index) result(is_abs)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_abs = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%name)) return
            if (.not. allocated(node%arg_indices)) return
            is_abs = same_name(node%name, 'abs') .and. size(node%arg_indices) == 1
        class default
            return
        end select
    end function reduction_expression_is_abs_call

    recursive logical function reduction_expression_extent_operand(arena, &
            node_index, context, extent, error_msg) result(has_array)
        !! Return a runtime i32 element count for an array expression. Scalars
        !! deliberately return false so callers can choose the array side of a
        !! scalar-broadcast expression.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: extent
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name, err, op
        integer :: sym, left, right, line, column
        type(lr_operand_desc_t) :: left_extent, right_extent, product

        has_array = .false.
        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) return
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol_compat(context, name)
            if (sym <= 0 .or. .not. context%symbols(sym)%is_array) return
            if (context%symbols(sym)%is_allocatable) then
                call allocatable_descriptor_extent_i32(context, sym, 1, extent, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
                if (context%symbols(sym)%array_rank == 2) then
                    call allocatable_descriptor_extent_i32(context, sym, 2, &
                        right_extent, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. emit_i32_binary(context%session, LR_OP_MUL, extent, &
                            right_extent, product, error_msg)) return
                    extent = product
                end if
            else if (context%symbols(sym)%has_runtime_dim_size(1)) then
                extent = context%symbols(sym)%runtime_dim_size(1)
                if (context%symbols(sym)%array_rank == 2 .and. &
                    context%symbols(sym)%has_runtime_dim_size(2)) then
                    if (.not. emit_i32_binary(context%session, LR_OP_MUL, extent, &
                            context%symbols(sym)%runtime_dim_size(2), product, &
                            error_msg)) return
                    extent = product
                end if
            else
                extent = i32_immediate(context%session, int( &
                    context%symbols(sym)%array_size, c_int64_t))
            end if
            has_array = .true.
            return
        end if

        if (reduction_expression_is_abs_call(arena, node_index)) then
            select type (abs_node => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                has_array = reduction_expression_extent_operand(arena, &
                    abs_node%arg_indices(1), context, extent, error_msg)
            end select
            return
        end if

        if (.not. is_binary_op(arena, node_index)) return
        call get_binary_op_info(arena, node_index, op, left, right, line, column, err)
        if (len_trim(err) > 0) then
            error_msg = err
            return
        end if
        select case (trim(op))
        case ('+', '-', '*', '/', '**')
            has_array = reduction_expression_extent_operand(arena, left, context, &
                extent, error_msg)
            if (len_trim(error_msg) > 0) return
            if (has_array) return
            has_array = reduction_expression_extent_operand(arena, right, context, &
                extent, error_msg)
        end select
    end function reduction_expression_extent_operand

    recursive subroutine lower_runtime_reduction_arg_element(arena, node_index, &
            linear_index, vk, context, value, error_msg)
        !! Evaluate one element of a runtime-shaped reduction expression.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lr_operand_desc_t), intent(in) :: linear_index
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: op
        integer :: sym, left, right, line, column, opcode
        character(len=:), allocatable :: name, err
        type(lr_operand_desc_t) :: lhs, rhs
        type(lr_operand_desc_t) :: call_args(2)

        call set_empty(error_msg)
        if (.not. node_exists(arena, node_index)) then
            error_msg = 'runtime reduction expression index is invalid'
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol_compat(context, name)
            if (sym <= 0) then
                error_msg = 'runtime reduction expression name was not declared: '// &
                            trim(name)
                return
            end if
            if (context%symbols(sym)%is_array) then
                call load_array_element_at_operand(context, sym, linear_index, &
                                                   value, error_msg)
            else
                call lower_reduction_scalar(arena, node_index, vk, context, value, &
                                            error_msg)
            end if
            return
        end if
        if (reduction_expression_is_abs_call(arena, node_index)) then
            select type (abs_node => arena%entries(node_index)%node)
            type is (call_or_subscript_node)
                call lower_runtime_reduction_arg_element(arena, &
                    abs_node%arg_indices(1), linear_index, vk, context, lhs, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
            end select
            if (vk == VALUE_F32) then
                call_args(1) = lhs
                if (.not. emit_liric_f32_call(context%session, 'fabsf', &
                        call_args(1:1), &
                        value, error_msg)) return
            else if (vk == VALUE_F64) then
                call_args(1) = lhs
                if (.not. emit_liric_f64_call(context%session, 'fabs', &
                        call_args(1:1), &
                        value, error_msg)) return
            else
                error_msg = 'ABS reduction expression requires a real array'
            end if
            return
        end if
        if (.not. is_binary_op(arena, node_index)) then
            call lower_reduction_scalar(arena, node_index, vk, context, value, &
                                        error_msg)
            return
        end if
        call get_binary_op_info(arena, node_index, op, left, right, line, column, err)
        if (len_trim(err) > 0) then
            error_msg = err
            return
        end if
        call lower_runtime_reduction_arg_element(arena, left, linear_index, vk, &
            context, lhs, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_runtime_reduction_arg_element(arena, right, linear_index, vk, &
            context, rhs, error_msg)
        if (len_trim(error_msg) > 0) return
        if (trim(op) == '**') then
            call_args(1) = lhs
            call_args(2) = rhs
            if (vk == VALUE_F32) then
                if (.not. emit_liric_f32_call(context%session, 'powf', call_args, &
                        value, error_msg)) return
            else if (vk == VALUE_F64) then
                if (.not. emit_liric_f64_call(context%session, 'pow', call_args, &
                        value, error_msg)) return
            else
                call lower_i32_pow(arena, right, line, column, context, lhs, value, &
                    error_msg)
            end if
            return
        end if
        if (vk == VALUE_F32 .or. vk == VALUE_F64) then
            call real_opcode(op, line, column, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk == VALUE_F32) then
                if (.not. emit_liric_f32_binary(context%session, opcode, lhs, rhs, &
                        value, error_msg)) return
            else
                if (.not. emit_liric_f64_binary(context%session, opcode, lhs, rhs, &
                        value, error_msg)) return
            end if
        else
            call integer_opcode(op, opcode, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, opcode, lhs, rhs, value, &
                    error_msg)) return
        end if
    end subroutine lower_runtime_reduction_arg_element

    subroutine lower_runtime_general_expr_reduction(arena, arg_index, vk, context, &
            value, error_msg, reduction_name)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index, vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: reduction_name
        type(lr_operand_desc_t) :: extent, entry_index, header_index, backedge_index
        type(lr_operand_desc_t) :: entry_acc, header_acc, backedge_acc, next_index
        type(lr_operand_desc_t) :: candidate, next_acc, condition, one
        integer(c_int32_t) :: entry_block, header_block, body_block, latch_block, &
                               exit_block, index_vreg, acc_vreg
        integer :: op

        op = reduction_operation(reduction_name)
        if (op == 0 .or. op == DIM_REDUCE_NORM2) then
            error_msg = trim(reduction_name)//' of a runtime array expression '// &
                        'is not supported'
            return
        end if
        if (.not. reduction_expression_extent_operand(arena, arg_index, context, &
                extent, error_msg)) then
            if (len_trim(error_msg) == 0) error_msg = &
                trim(reduction_name)//' argument expression is not an array'
            return
        end if
        call reduction_identity(context, op, vk, entry_acc, error_msg)
        if (len_trim(error_msg) > 0) return
        one = i32_immediate(context%session, 1_c_int64_t)
        entry_index = i32_immediate(context%session, 0_c_int64_t)
        entry_block = context%current_block_id
        header_block = create_liric_block(context%session)
        body_block = create_liric_block(context%session)
        latch_block = create_liric_block(context%session)
        exit_block = create_liric_block(context%session)
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return
        if (.not. enter_liric_block(context, header_block, error_msg)) return
        index_vreg = reserve_i32_vreg(context%session)
        acc_vreg = reserve_i32_vreg(context%session)
        if (index_vreg <= 0_c_int32_t .or. acc_vreg <= 0_c_int32_t) then
            error_msg = 'LIRIC could not reserve runtime reduction vregs'
            return
        end if
        backedge_index = i32_vreg(context%session, index_vreg)
        if (vk == VALUE_F32) then
            backedge_acc = f32_vreg(context%session, acc_vreg)
        else if (vk == VALUE_F64) then
            backedge_acc = f64_vreg(context%session, acc_vreg)
        else
            backedge_acc = i32_vreg(context%session, acc_vreg)
        end if
        if (.not. emit_liric_i32_phi(context%session, entry_index, entry_block, &
                backedge_index, latch_block, header_index, error_msg)) return
        if (.not. emit_liric_phi(context%session, entry_acc, entry_block, &
                backedge_acc, latch_block, header_acc, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, header_index, &
                extent, condition, error_msg)) return
        if (.not. emit_liric_condbr(context%session, condition, body_block, &
                exit_block, error_msg)) return
        if (.not. enter_liric_block(context, body_block, error_msg)) return
        call lower_runtime_reduction_arg_element(arena, arg_index, header_index, vk, &
            context, candidate, error_msg)
        if (len_trim(error_msg) > 0) return
        next_acc = header_acc
        call reduction_combine(context, op, vk, next_acc, candidate, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, latch_block, error_msg)) return
        if (.not. enter_liric_block(context, latch_block, error_msg)) return
        if (.not. emit_i32_binary_into(context%session, LR_OP_ADD, header_index, one, &
                index_vreg, next_index, error_msg)) return
        if (vk == VALUE_F32 .or. vk == VALUE_F64) then
            if (.not. emit_real_copy_to(context%session, next_acc, acc_vreg, &
                    backedge_acc, error_msg)) return
        else
            if (.not. emit_i32_copy_to(context%session, next_acc, acc_vreg, &
                    backedge_acc, error_msg)) return
        end if
        if (.not. emit_liric_br(context%session, header_block, error_msg)) return
        if (.not. enter_liric_block(context, exit_block, error_msg)) return
        value = header_acc
        call set_empty(error_msg)
    end subroutine lower_runtime_general_expr_reduction

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
        type(lr_operand_desc_t) :: call_args(2)

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
            sym = find_symbol_compat(context, id_name)
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
                    call_args(1) = lhs
                    call_args(2) = rhs
                    if (vk == VALUE_F32) then
                        if (.not. emit_liric_f32_call(context%session, 'powf', &
                                call_args, value, error_msg)) return
                    else
                        if (.not. emit_liric_f64_call(context%session, 'pow', &
                                call_args, value, error_msg)) return
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
        !! Fold one reduction over an array-expression argument by iterating
        !! the expression's elements, starting from the reduction's identity
        !! and combining one element at a time. SUM, PRODUCT, MINVAL, MAXVAL,
        !! COUNT, ANY, and ALL all share this loop; the operation appears only
        !! in `reduction_identity` and `reduction_combine`.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: arg_index
        integer, intent(in) :: extent
        integer, intent(in) :: vk
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=*), intent(in) :: reduction_name
        type(lr_operand_desc_t) :: candidate, cond, one, zero
        integer :: i, op

        op = reduction_operation(reduction_name)
        if (op == 0 .or. op == DIM_REDUCE_NORM2) then
            error_msg = trim(reduction_name)//' of a general array expression '// &
                        'is not supported'
            return
        end if
        call reduction_identity(context, op, vk, value, error_msg)
        if (len_trim(error_msg) > 0) return
        do i = 0, extent - 1
            if (reduction_is_mask_valued(op)) then
                ! A mask-valued reduction consumes the logical value of the
                ! argument at this element, materialized as an i32 0 or 1.
                call lower_where_mask_element(arena, arg_index, i, context, &
                                              cond, error_msg)
                if (len_trim(error_msg) > 0) return
                one = i32_immediate(context%session, 1_c_int64_t)
                zero = i32_immediate(context%session, 0_c_int64_t)
                call select_value(context, cond, one, zero, candidate, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                call lower_reduction_arg_element(arena, arg_index, &
                        int(i, c_int64_t), vk, context, candidate, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            call reduction_combine(context, op, vk, value, candidate, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
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
                i64_immediate(context%session, &
                    int(ALLOC_DESCRIPTOR_BYTES, c_int64_t)), temp_desc, &
                error_msg)) return
        zero = i64_immediate(context%session, 0_c_int64_t)
        do i = 0, ALLOC_DESCRIPTOR_BYTES / 8 - 1
            if (.not. emit_i64_store_at(context%session, zero, temp_desc, &
                    int(i * 8, c_int64_t), error_msg)) return
        end do

        allocate (all_args(size(user_args) + 1))
        all_args(1) = temp_desc
        do i = 1, size(user_args)
            all_args(i + 1) = user_args(i)
        end do
        if (.not. emit_void_call(context%session, &
                                 call_emit_name(arena, trim(func_name), context), &
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

    ! ---------------------------------------------------------------------
    ! NORM2 support: a scaled sum of squares over already-loaded elements.
    ! ---------------------------------------------------------------------

    subroutine norm2_real_immediate(context, vk, val, value)
        !! Real immediate of the element kind vk.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        real(c_double), intent(in) :: val
        type(lr_operand_desc_t), intent(out) :: value

        if (vk == VALUE_F32) then
            value = liric_f32_immediate(context%session, real(val, c_float))
        else
            value = liric_f64_immediate(context%session, val)
        end if
    end subroutine norm2_real_immediate

    logical function norm2_binary(context, vk, op, lhs, rhs, value, error_msg)
        !! One real arithmetic instruction of the element kind vk.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        integer(c_int), intent(in) :: op
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (vk == VALUE_F32) then
            norm2_binary = emit_liric_f32_binary(context%session, op, lhs, rhs, &
                                                 value, error_msg)
        else
            norm2_binary = emit_liric_f64_binary(context%session, op, lhs, rhs, &
                                                 value, error_msg)
        end if
    end function norm2_binary

    logical function norm2_libm(context, vk, name32, name64, arg, value, error_msg)
        !! One single-argument libm call of the element kind vk.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        character(len=*), intent(in) :: name32
        character(len=*), intent(in) :: name64
        type(lr_operand_desc_t), intent(in) :: arg
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: call_args(1)

        call_args(1) = arg
        if (vk == VALUE_F32) then
            norm2_libm = emit_liric_f32_call(context%session, name32, call_args, &
                                             value, error_msg)
        else
            norm2_libm = emit_liric_f64_call(context%session, name64, call_args, &
                                             value, error_msg)
        end if
    end function norm2_libm

    logical function norm2_fcmp(context, vk, predicate, lhs, rhs, value, error_msg)
        !! One real comparison of the element kind vk.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        integer(c_int), intent(in) :: predicate
        type(lr_operand_desc_t), intent(in) :: lhs
        type(lr_operand_desc_t), intent(in) :: rhs
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        if (vk == VALUE_F32) then
            norm2_fcmp = emit_liric_f32_fcmp(context%session, predicate, lhs, &
                                             rhs, value, error_msg)
        else
            norm2_fcmp = emit_liric_f64_fcmp(context%session, predicate, lhs, &
                                             rhs, value, error_msg)
        end if
    end function norm2_fcmp

    subroutine norm2_scale_factor(context, vk, elems, scale, error_msg)
        !! Largest element magnitude, used as the scaling factor.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: elems(:)
        type(lr_operand_desc_t), intent(out) :: scale
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: mag, cond, picked
        integer :: i

        call set_empty(error_msg)
        if (.not. norm2_libm(context, vk, 'fabsf', 'fabs', elems(1), scale, &
                             error_msg)) return
        do i = 2, size(elems)
            if (.not. norm2_libm(context, vk, 'fabsf', 'fabs', elems(i), mag, &
                                 error_msg)) return
            if (.not. norm2_fcmp(context, vk, LR_FCMP_OGT, mag, scale, cond, &
                                 error_msg)) return
            call select_value(context, cond, mag, scale, picked, error_msg)
            if (len_trim(error_msg) > 0) return
            scale = picked
        end do
    end subroutine norm2_scale_factor

    subroutine emit_norm2_from_elements(context, vk, elems, value, error_msg)
        !! L2 norm of already-loaded real elements, computed as
        !! scale * sqrt(sum((x / scale)**2)) with scale the largest magnitude.
        !! Factoring the largest magnitude out keeps mixed large or small
        !! magnitudes from overflowing or underflowing the accumulator. A zero
        !! scale divides by one instead, so the zero vector yields exactly zero.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(in) :: elems(:)
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: scale, denom, acc, zero, one, cond
        type(lr_operand_desc_t) :: scaled, squared, next_acc, root
        integer :: i

        call set_empty(error_msg)
        if (vk /= VALUE_F32 .and. vk /= VALUE_F64) then
            error_msg = 'norm2 requires a real array argument'
            return
        end if
        if (size(elems) < 1) then
            error_msg = 'norm2 requires a non-empty array'
            return
        end if
        call norm2_scale_factor(context, vk, elems, scale, error_msg)
        if (len_trim(error_msg) > 0) return
        call norm2_real_immediate(context, vk, 0.0_c_double, zero)
        call norm2_real_immediate(context, vk, 1.0_c_double, one)
        if (.not. norm2_fcmp(context, vk, LR_FCMP_OEQ, scale, zero, cond, &
                             error_msg)) return
        call select_value(context, cond, one, scale, denom, error_msg)
        if (len_trim(error_msg) > 0) return
        acc = zero
        do i = 1, size(elems)
            if (.not. norm2_binary(context, vk, LR_OP_FDIV, elems(i), denom, &
                                   scaled, error_msg)) return
            if (.not. norm2_binary(context, vk, LR_OP_FMUL, scaled, scaled, &
                                   squared, error_msg)) return
            if (.not. norm2_binary(context, vk, LR_OP_FADD, acc, squared, &
                                   next_acc, error_msg)) return
            acc = next_acc
        end do
        if (.not. norm2_libm(context, vk, 'sqrtf', 'sqrt', acc, root, &
                             error_msg)) return
        if (.not. norm2_binary(context, vk, LR_OP_FMUL, scale, root, value, &
                               error_msg)) return
    end subroutine emit_norm2_from_elements

    subroutine lower_symbol_norm2(context, source_index, array_size, value, &
                                  error_msg)
        !! norm2 over the first array_size elements of a stored array symbol,
        !! walked in storage order (so a rank-2 source reduces over all of it).
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: source_index
        integer, intent(in) :: array_size
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: elems(:)
        integer :: i

        call set_empty(error_msg)
        if (array_size < 1) then
            error_msg = 'norm2 requires a non-empty array'
            return
        end if
        allocate (elems(array_size))
        do i = 1, array_size
            call load_array_linear_element(context, source_index, &
                                           int(i - 1, c_int64_t), elems(i), &
                                           error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call emit_norm2_from_elements(context, &
                                      context%symbols(source_index)%value_kind, &
                                      elems, value, error_msg)
    end subroutine lower_symbol_norm2

    subroutine lower_section_norm2(info, total, context, value, error_msg)
        !! norm2 over an array section a(lo:hi:st), walked in storage order.
        type(array_section_info_t), intent(in) :: info
        integer(c_int64_t), intent(in) :: total
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: elems(:)
        integer(c_int64_t) :: k

        call set_empty(error_msg)
        if (total < 1_c_int64_t) then
            error_msg = 'norm2 requires a non-empty section'
            return
        end if
        allocate (elems(total))
        do k = 0_c_int64_t, total - 1_c_int64_t
            call lower_array_section_element_from_info(info, k, context, &
                                                       elems(k + 1), error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call emit_norm2_from_elements(context, &
                                      context%symbols(info%source_index)%value_kind, &
                                      elems, value, error_msg)
    end subroutine lower_section_norm2

end submodule session_program_lowering_reduction_expr
