    ! WHERE construct and statement: masked elementwise array assignment.
    !
    ! A WHERE assigns to array elements only where a conformable logical mask is
    ! true; ELSEWHERE assigns the complement. ffc unrolls array work at lowering
    ! time (one set of LIRIC ops per element), so a WHERE is lowered the same
    ! way: for each linear element index, evaluate the mask element, evaluate the
    ! body and complement element values, and select between them. select_value
    ! emits the per-element branch/phi, so no explicit loop block is needed.

    subroutine lower_where_stmt(arena, node, context, error_msg)
        ! WHERE (mask) array_lhs = array_rhs  (single-statement form).
        type(ast_arena_t), intent(in) :: arena
        type(where_stmt_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (node%mask_expr_index <= 0) then
            error_msg = 'WHERE statement requires a mask expression'
            return
        end if
        if (node%assignment_index <= 0) then
            error_msg = 'WHERE statement requires an assignment'
            return
        end if
        call lower_where_masked_assignment(arena, node%assignment_index, &
                                           [node%mask_expr_index], &
                                           context, error_msg)
    end subroutine lower_where_stmt

    subroutine lower_where_construct(arena, node, context, error_msg)
        ! WHERE (mask) body [ELSEWHERE body] END WHERE.
        ! The body runs where the mask is true; a final unmasked ELSEWHERE runs
        ! the complement. The body may nest further WHERE constructs/statements,
        ! whose masks AND with the enclosing mask element-wise. Masked ELSEWHERE
        ! is out of scope and produces a clean diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(where_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i
        integer :: outer_masks(1)

        call set_empty(error_msg)
        if (node%mask_expr_index <= 0) then
            error_msg = 'WHERE construct requires a mask expression'
            return
        end if

        if (allocated(node%elsewhere_clauses)) then
            do i = 1, size(node%elsewhere_clauses)
                if (node%elsewhere_clauses(i)%mask_index > 0) then
                    call unsupported_feature_error('masked elsewhere', &
                        get_node_line(arena, node%mask_expr_index), &
                        get_node_column(arena, node%mask_expr_index), &
                        'ffc direct-session supports only a final unmasked '// &
                        'ELSEWHERE', error_msg)
                    return
                end if
            end do
        end if

        outer_masks(1) = node%mask_expr_index
        if (allocated(node%where_body_indices)) then
            do i = 1, size(node%where_body_indices)
                call lower_where_body_statement(arena, &
                    node%where_body_indices(i), outer_masks, context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if

        if (allocated(node%elsewhere_clauses)) then
            do i = 1, size(node%elsewhere_clauses)
                call lower_where_elsewhere_body(arena, &
                    node%elsewhere_clauses(i), node%mask_expr_index, context, &
                    error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        end if
    end subroutine lower_where_construct

    recursive subroutine lower_where_body_statement(arena, stmt_index, &
                                                    masks, context, error_msg)
        ! Lower one WHERE body statement under the conjunction of enclosing
        ! masks. An array assignment is applied where every mask is true; a
        ! nested WHERE construct/statement extends the mask list with its own
        ! mask and recurses over its body.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: stmt_index
        integer, intent(in) :: masks(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: nested_masks(:)
        integer :: i

        call set_empty(error_msg)
        if (.not. node_exists(arena, stmt_index)) then
            error_msg = 'WHERE body statement does not reference an AST node'
            return
        end if
        select type (stmt => arena%entries(stmt_index)%node)
        type is (where_node)
            if (stmt%mask_expr_index <= 0) then
                error_msg = 'nested WHERE construct requires a mask expression'
                return
            end if
            if (allocated(stmt%elsewhere_clauses)) then
                call unsupported_feature_error('nested elsewhere', &
                    get_node_line(arena, stmt_index), &
                    get_node_column(arena, stmt_index), &
                    'ffc direct-session does not support ELSEWHERE in a '// &
                    'nested WHERE', error_msg)
                return
            end if
            nested_masks = [masks, stmt%mask_expr_index]
            if (allocated(stmt%where_body_indices)) then
                do i = 1, size(stmt%where_body_indices)
                    call lower_where_body_statement(arena, &
                        stmt%where_body_indices(i), nested_masks, context, &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                end do
            end if
        type is (where_stmt_node)
            if (stmt%mask_expr_index <= 0 .or. stmt%assignment_index <= 0) then
                error_msg = 'nested WHERE statement is incomplete'
                return
            end if
            nested_masks = [masks, stmt%mask_expr_index]
            call lower_where_masked_assignment(arena, stmt%assignment_index, &
                                               nested_masks, context, error_msg)
        type is (assignment_node)
            call lower_where_masked_assignment(arena, stmt_index, masks, &
                                               context, error_msg)
        class default
            call unsupported_feature_error('where body statement', &
                get_node_line(arena, stmt_index), &
                get_node_column(arena, stmt_index), &
                'WHERE body supports array assignments and nested WHERE', &
                error_msg)
        end select
    end subroutine lower_where_body_statement

    subroutine lower_where_elsewhere_body(arena, clause, mask_index, context, &
                                          error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(elsewhere_clause_t), intent(in) :: clause
        integer, intent(in) :: mask_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: j

        call set_empty(error_msg)
        if (.not. allocated(clause%body_indices)) return
        do j = 1, size(clause%body_indices)
            call lower_where_complement_assignment(arena, &
                clause%body_indices(j), mask_index, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_where_elsewhere_body

    subroutine lower_where_masked_assignment(arena, assign_index, masks, &
                                             context, error_msg)
        ! Lower one WHERE body assignment under the conjunction of masks. For
        ! each element where every mask is true:
        !   lhs(i) := rhs(i)   otherwise lhs(i) keeps its current value.
        ! The kept branch reads the current element, so an earlier WHERE result
        ! survives into a complementary ELSEWHERE pass over the same target.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: assign_index
        integer, intent(in) :: masks(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: target_symbol, value_index, i, array_size
        type(lr_operand_desc_t) :: cond, rhs_val, cur_val, selected

        call set_empty(error_msg)
        call where_assignment_target(arena, assign_index, context, &
                                     target_symbol, value_index, error_msg)
        if (len_trim(error_msg) > 0) return

        array_size = context%symbols(target_symbol)%array_size
        if (array_size <= 0) then
            error_msg = 'WHERE assignment requires a positive array size'
            return
        end if

        do i = 0, array_size - 1
            call lower_where_mask_conjunction(arena, masks, i, context, cond, &
                                              error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_elementwise_value(arena, value_index, &
                                               target_symbol, i, context, &
                                               rhs_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, target_symbol, &
                                           int(i, c_int64_t), cur_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call select_value(context, cond, rhs_val, cur_val, selected, &
                              error_msg)
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, target_symbol, &
                                            int(i, c_int64_t), selected, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_where_masked_assignment

    subroutine lower_where_complement_assignment(arena, assign_index, &
                                                 mask_index, context, error_msg)
        ! ELSEWHERE body: lhs(i) := mask(i) ? lhs(i) : rhs(i). The kept branch
        ! preserves the WHERE-true result computed in the prior pass.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: assign_index
        integer, intent(in) :: mask_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: target_symbol, value_index, i, array_size
        type(lr_operand_desc_t) :: cond, rhs_val, cur_val, selected

        call set_empty(error_msg)
        call where_assignment_target(arena, assign_index, context, &
                                     target_symbol, value_index, error_msg)
        if (len_trim(error_msg) > 0) return

        array_size = context%symbols(target_symbol)%array_size
        if (array_size <= 0) then
            error_msg = 'WHERE assignment requires a positive array size'
            return
        end if

        do i = 0, array_size - 1
            call lower_where_mask_element(arena, mask_index, i, context, cond, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_elementwise_value(arena, value_index, &
                                               target_symbol, i, context, &
                                               rhs_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call load_array_linear_element(context, target_symbol, &
                                           int(i, c_int64_t), cur_val, error_msg)
            if (len_trim(error_msg) > 0) return
            call select_value(context, cond, cur_val, rhs_val, selected, &
                              error_msg)
            if (len_trim(error_msg) > 0) return
            call store_array_linear_element(context, target_symbol, &
                                            int(i, c_int64_t), selected, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_where_complement_assignment

    subroutine lower_where_mask_conjunction(arena, masks, linear_index, &
                                            context, cond, error_msg)
        ! AND the per-element values of every enclosing mask into one i1.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: masks(:)
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: cond
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: next, combined
        integer :: m

        call set_empty(error_msg)
        call lower_where_mask_element(arena, masks(1), linear_index, context, &
                                      cond, error_msg)
        if (len_trim(error_msg) > 0) return
        do m = 2, size(masks)
            call lower_where_mask_element(arena, masks(m), linear_index, &
                                          context, next, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_i32_binary(context%session, LR_OP_AND, cond, next, &
                                      combined, error_msg)) return
            cond = combined
        end do
    end subroutine lower_where_mask_conjunction

    subroutine where_assignment_target(arena, assign_index, context, &
                                       target_symbol, value_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: assign_index
        type(lowering_context_t), intent(inout) :: context
        integer, intent(out) :: target_symbol
        integer, intent(out) :: value_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name

        call set_empty(error_msg)
        target_symbol = 0
        value_index = 0
        if (.not. node_exists(arena, assign_index)) then
            error_msg = 'WHERE body statement does not reference an AST node'
            return
        end if
        select type (stmt => arena%entries(assign_index)%node)
        type is (assignment_node)
            value_index = stmt%value_index
            if (.not. is_identifier(arena, stmt%target_index)) then
                call unsupported_feature_error('where assignment', &
                    get_node_line(arena, assign_index), &
                    get_node_column(arena, assign_index), &
                    'WHERE body must assign to a whole array variable', error_msg)
                return
            end if
            call get_identifier_name(arena, stmt%target_index, name, error_msg)
            if (len_trim(error_msg) > 0) return
            target_symbol = find_symbol(context, name)
            if (target_symbol <= 0) then
                error_msg = 'WHERE assignment target was not declared: '//trim(name)
                return
            end if
            if (.not. context%symbols(target_symbol)%is_array) then
                call unsupported_feature_error('where assignment', &
                    get_node_line(arena, assign_index), &
                    get_node_column(arena, assign_index), &
                    'WHERE body must assign to an array', error_msg)
                return
            end if
        class default
            call unsupported_feature_error('where body statement', &
                get_node_line(arena, assign_index), &
                get_node_column(arena, assign_index), &
                'WHERE body supports only array assignments', error_msg)
        end select
    end subroutine where_assignment_target

    subroutine lower_where_mask_element(arena, mask_index, linear_index, &
                                        context, cond, error_msg)
        ! Evaluate a scalar-comparison mask (e.g. a > 0, b == 0) at one element,
        ! yielding an i1 condition. Operands are array elements or scalars.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: mask_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: cond
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: bin_op
        integer :: bin_left, bin_right, bin_line, bin_col
        integer :: operand_symbol, value_kind
        type(lr_operand_desc_t) :: lhs, rhs
        integer(c_int) :: ipred, fpred

        call set_empty(error_msg)
        if (.not. is_binary_op(arena, mask_index)) then
            call unsupported_feature_error('where mask', &
                get_node_line(arena, mask_index), &
                get_node_column(arena, mask_index), &
                'WHERE mask must be a scalar comparison over arrays', error_msg)
            return
        end if
        call get_binary_op_info(arena, mask_index, bin_op, bin_left, bin_right, &
                                bin_line, bin_col, error_msg)
        if (len_trim(error_msg) > 0) return

        call where_mask_operand_kind(arena, bin_left, bin_right, context, &
                                     value_kind, operand_symbol, error_msg)
        if (len_trim(error_msg) > 0) return

        call lower_where_operand_element(arena, bin_left, value_kind, &
                                         linear_index, context, lhs, error_msg)
        if (len_trim(error_msg) > 0) return
        call lower_where_operand_element(arena, bin_right, value_kind, &
                                         linear_index, context, rhs, error_msg)
        if (len_trim(error_msg) > 0) return

        if (value_kind == VALUE_F32 .or. value_kind == VALUE_F64) then
            call real_compare_fpred(bin_op, fpred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (value_kind == VALUE_F32) then
                if (.not. emit_liric_f32_fcmp(context%session, fpred, lhs, rhs, &
                                              cond, error_msg)) return
            else
                if (.not. emit_liric_f64_fcmp(context%session, fpred, lhs, rhs, &
                                              cond, error_msg)) return
            end if
        else
            call integer_compare_predicate(bin_op, ipred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, ipred, lhs, rhs, &
                                          cond, error_msg)) return
        end if
    end subroutine lower_where_mask_element

    subroutine where_mask_operand_kind(arena, left, right, context, value_kind, &
                                       array_symbol, error_msg)
        ! The mask compares an array against a scalar (or another array). The
        ! comparison kind follows the array operand's element type.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: left, right
        type(lowering_context_t), intent(in) :: context
        integer, intent(out) :: value_kind
        integer, intent(out) :: array_symbol
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym

        call set_empty(error_msg)
        value_kind = VALUE_I32
        array_symbol = 0
        sym = where_array_operand_symbol(arena, left, context)
        if (sym <= 0) sym = where_array_operand_symbol(arena, right, context)
        if (sym <= 0) then
            call unsupported_feature_error('where mask', &
                get_node_line(arena, left), get_node_column(arena, left), &
                'WHERE mask must reference at least one array operand', error_msg)
            return
        end if
        array_symbol = sym
        value_kind = context%symbols(sym)%value_kind
    end subroutine where_mask_operand_kind

    integer function where_array_operand_symbol(arena, node_index, context) &
            result(sym)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err

        sym = 0
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol(context, name)
        if (sym <= 0) then
            sym = 0
            return
        end if
        if (.not. context%symbols(sym)%is_array) sym = 0
    end function where_array_operand_symbol

    subroutine lower_where_operand_element(arena, node_index, value_kind, &
                                           linear_index, context, value, error_msg)
        ! Materialize a mask comparison operand at one element. Array operands
        ! load the element; scalars are evaluated once per element (constant
        ! folded by the backend) in the comparison's kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: value_kind
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym

        call set_empty(error_msg)
        sym = where_array_operand_symbol(arena, node_index, context)
        if (sym > 0) then
            call load_array_linear_element(context, sym, &
                                           int(linear_index, c_int64_t), value, &
                                           error_msg)
            return
        end if
        ! Scalar operand: lower in the comparison's kind.
        select case (value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        end select
    end subroutine lower_where_operand_element

    subroutine real_compare_fpred(source_op, predicate, error_msg)
        character(len=*), intent(in) :: source_op
        integer(c_int), intent(out) :: predicate
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(source_op))
        case ('>')
            predicate = LR_FCMP_OGT
        case ('>=')
            predicate = LR_FCMP_OGE
        case ('<')
            predicate = LR_FCMP_OLT
        case ('<=')
            predicate = LR_FCMP_OLE
        case default
            error_msg = 'WHERE real mask supports only >, >=, <, <= comparisons'
            predicate = LR_FCMP_OGT
        end select
    end subroutine real_compare_fpred

    subroutine lower_elementwise_scalar_operand(arena, node_index, &
                                                target_symbol_index, context, &
                                                value, error_msg)
        ! Broadcast a scalar (literal or scalar variable) used in whole-array
        ! arithmetic: lower it once in the target element's kind.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: target_symbol_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (context%symbols(target_symbol_index)%value_kind)
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, value, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, value, error_msg)
        case (VALUE_LOGICAL)
            call lower_logical_expression(arena, node_index, context, value, &
                                          error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        end select
    end subroutine lower_elementwise_scalar_operand

    logical function is_elemental_real_call(arena, node_index) result(is_call)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=:), allocatable :: libm32, libm64

        is_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(n%name)) return
            if (.not. allocated(n%arg_indices)) return
            if (size(n%arg_indices) /= 1) return
            call elemental_real_libm_names(n%name, libm32, libm64)
            is_call = len(libm64) > 0
        end select
    end function is_elemental_real_call

    subroutine lower_elementwise_elemental_call(arena, node_index, &
                                                target_symbol_index, &
                                                linear_index, context, value, &
                                                error_msg)
        ! Apply an elemental real intrinsic (sqrt, exp, ...) to one array element
        ! by loading the argument element and calling the matching libm routine.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: arg(1)
        character(len=:), allocatable :: libm32, libm64

        call set_empty(error_msg)
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            call elemental_real_libm_names(n%name, libm32, libm64)
            call lower_array_elementwise_value(arena, n%arg_indices(1), &
                                               target_symbol_index, linear_index, &
                                               context, arg(1), error_msg)
            if (len_trim(error_msg) > 0) return
            if (context%symbols(target_symbol_index)%value_kind == VALUE_F32) then
                if (.not. emit_liric_f32_call(context%session, libm32, arg, &
                                              value, error_msg)) return
            else
                if (.not. emit_liric_f64_call(context%session, libm64, arg, &
                                              value, error_msg)) return
            end if
        class default
            error_msg = 'elemental call node lost its type'
        end select
    end subroutine lower_elementwise_elemental_call

    ! True when node is a one-argument call to a contained real function, used
    ! to apply a user-defined ELEMENTAL function element-wise (arr = f(arr)).
    logical function is_elemental_user_call(arena, node_index, context) &
        result(is_call)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context

        is_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (n%is_array_access) return
            if (.not. allocated(n%name)) return
            if (.not. allocated(n%arg_indices)) return
            if (size(n%arg_indices) /= 1) return
            is_call = is_contained_f32_function(context, n%name) .or. &
                      is_contained_f64_function(context, n%name) .or. &
                      is_contained_i32_function(context, n%name)
        end select
    end function is_elemental_user_call

    ! Apply a user-defined elemental real function to one array element: load the
    ! argument's element, materialise it as a by-reference scalar (the contained
    ! ABI), and call the function. The result kind follows the target array.
    subroutine lower_elementwise_user_call(arena, node_index, &
                                           target_symbol_index, linear_index, &
                                           context, value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: elem, arg(1)
        integer :: vk

        call set_empty(error_msg)
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            call lower_array_elementwise_value(arena, n%arg_indices(1), &
                                               target_symbol_index, linear_index, &
                                               context, elem, error_msg)
            if (len_trim(error_msg) > 0) return
            vk = context%symbols(target_symbol_index)%value_kind
            call make_reference_argument(context, vk, elem, arg(1), error_msg)
            if (len_trim(error_msg) > 0) return
            select case (vk)
            case (VALUE_F32)
                if (.not. emit_liric_f32_call(context%session, &
                        call_emit_name(arena, n%name), arg, value, error_msg)) return
            case (VALUE_I32)
                if (.not. emit_i32_call(context%session, &
                        call_emit_name(arena, n%name), arg, value, error_msg)) return
            case default
                if (.not. emit_liric_f64_call(context%session, &
                        call_emit_name(arena, n%name), arg, value, error_msg)) return
            end select
        class default
            error_msg = 'elemental user call node lost its type'
        end select
    end subroutine lower_elementwise_user_call

    logical function is_elemental_minmax_call(arena, node_index) result(is_call)
        ! True for a two-array-operand MAX/MIN call usable as a whole-array
        ! elementwise RHS operand (a = max(b, c)).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index

        is_call = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(n%name)) return
            if (.not. allocated(n%arg_indices)) return
            if (size(n%arg_indices) /= 2) return
            is_call = same_name(n%name, 'max') .or. same_name(n%name, 'min')
        end select
    end function is_elemental_minmax_call

    subroutine lower_elementwise_minmax_call(arena, node_index, &
                                             target_symbol_index, linear_index, &
                                             context, value, error_msg)
        ! Apply whole-array MAX/MIN elementwise: max(a, b) = (a > b) ? a : b.
        ! The comparison is consumed only as a select_value branch condition
        ! (as WHERE already does), never stored directly: this backend
        ! mis-compiles a raw comparison result written straight to memory.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: target_symbol_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: lhs, rhs, cond
        logical :: want_max
        integer :: vk

        call set_empty(error_msg)
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            want_max = same_name(n%name, 'max')
            vk = context%symbols(target_symbol_index)%value_kind
            if (vk /= VALUE_F32 .and. vk /= VALUE_F64 .and. vk /= VALUE_I32) then
                call unsupported_feature_error('whole-array max/min', n%line, &
                    n%column, 'direct LIRIC session supports max/min for '// &
                    'default-kind integer and real arrays only', error_msg)
                return
            end if
            call lower_array_elementwise_value(arena, n%arg_indices(1), &
                                               target_symbol_index, linear_index, &
                                               context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_array_elementwise_value(arena, n%arg_indices(2), &
                                               target_symbol_index, linear_index, &
                                               context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk == VALUE_F32) then
                if (want_max) then
                    if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OGT, &
                            lhs, rhs, cond, error_msg)) return
                else
                    if (.not. emit_liric_f32_fcmp(context%session, LR_FCMP_OLT, &
                            lhs, rhs, cond, error_msg)) return
                end if
            else if (vk == VALUE_F64) then
                if (want_max) then
                    if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OGT, &
                            lhs, rhs, cond, error_msg)) return
                else
                    if (.not. emit_liric_f64_fcmp(context%session, LR_FCMP_OLT, &
                            lhs, rhs, cond, error_msg)) return
                end if
            else
                if (want_max) then
                    if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SGT, &
                            lhs, rhs, cond, error_msg)) return
                else
                    if (.not. emit_liric_i32_icmp(context%session, LR_CMP_SLT, &
                            lhs, rhs, cond, error_msg)) return
                end if
            end if
            call select_value(context, cond, lhs, rhs, value, error_msg)
        class default
            error_msg = 'max/min call node lost its type'
        end select
    end subroutine lower_elementwise_minmax_call

    subroutine elemental_real_libm_names(name, libm32, libm64)
        character(len=*), intent(in) :: name
        character(len=:), allocatable, intent(out) :: libm32
        character(len=:), allocatable, intent(out) :: libm64

        libm32 = ''
        libm64 = ''
        if (same_name(name, 'sqrt')) then
            libm32 = 'sqrtf'; libm64 = 'sqrt'
        else if (same_name(name, 'exp')) then
            libm32 = 'expf'; libm64 = 'exp'
        else if (same_name(name, 'log')) then
            libm32 = 'logf'; libm64 = 'log'
        else if (same_name(name, 'sin')) then
            libm32 = 'sinf'; libm64 = 'sin'
        else if (same_name(name, 'cos')) then
            libm32 = 'cosf'; libm64 = 'cos'
        else if (same_name(name, 'tan')) then
            libm32 = 'tanf'; libm64 = 'tan'
        else if (same_name(name, 'asin')) then
            libm32 = 'asinf'; libm64 = 'asin'
        else if (same_name(name, 'acos')) then
            libm32 = 'acosf'; libm64 = 'acos'
        else if (same_name(name, 'atan')) then
            libm32 = 'atanf'; libm64 = 'atan'
        else if (same_name(name, 'sinh')) then
            libm32 = 'sinhf'; libm64 = 'sinh'
        else if (same_name(name, 'cosh')) then
            libm32 = 'coshf'; libm64 = 'cosh'
        else if (same_name(name, 'tanh')) then
            libm32 = 'tanhf'; libm64 = 'tanh'
        else if (same_name(name, 'asinh')) then
            libm32 = 'asinhf'; libm64 = 'asinh'
        else if (same_name(name, 'acosh')) then
            libm32 = 'acoshf'; libm64 = 'acosh'
        else if (same_name(name, 'atanh')) then
            libm32 = 'atanhf'; libm64 = 'atanh'
        else if (same_name(name, 'log10')) then
            libm32 = 'log10f'; libm64 = 'log10'
        else if (same_name(name, 'erf')) then
            libm32 = 'erff'; libm64 = 'erf'
        else if (same_name(name, 'erfc')) then
            libm32 = 'erfcf'; libm64 = 'erfc'
        else if (same_name(name, 'abs')) then
            libm32 = 'fabsf'; libm64 = 'fabs'
        end if
    end subroutine elemental_real_libm_names
