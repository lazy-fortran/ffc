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
        ! WHERE body assignment: apply the right-hand side where every
        ! enclosing mask is true and keep the current element elsewhere.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: assign_index
        integer, intent(in) :: masks(:)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        call lower_where_selected_assignment(arena, assign_index, masks, &
                                             .false., context, error_msg)
    end subroutine lower_where_masked_assignment

    subroutine lower_where_complement_assignment(arena, assign_index, &
                                                 mask_index, context, error_msg)
        ! ELSEWHERE body: the complement of the WHERE mask. The kept branch
        ! preserves the WHERE-true result computed in the prior pass.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: assign_index
        integer, intent(in) :: mask_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        call lower_where_selected_assignment(arena, assign_index, &
                                             [mask_index], .true., context, &
                                             error_msg)
    end subroutine lower_where_complement_assignment

    subroutine lower_where_selected_assignment(arena, assign_index, masks, &
                                               complement, context, error_msg)
        ! The one masked array assignment WHERE and ELSEWHERE share:
        !   lhs(i) := mask(i) ? rhs(i) : lhs(i)          (complement = .false.)
        !   lhs(i) := mask(i) ? lhs(i) : rhs(i)          (complement = .true.)
        !
        ! Every mask value, right-hand-side element, and kept current element
        ! is produced before the first store, so the assignment behaves as if
        ! the mask and the right-hand side were fully evaluated before any
        ! element of the target changed (Fortran 2018 clause 10.2.3.1). A mask
        ! or a right-hand side that overlaps the target -- `where (m == 1)
        ! a = a(n:1:-1)` -- therefore reads pre-assignment values regardless of
        ! traversal order.
        !
        ! The iteration comes from the target's array-expression plan, so the
        ! shape, the element type, and the column-major linear index are the
        ! same ones every other array expression uses.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: assign_index
        integer, intent(in) :: masks(:)
        logical, intent(in) :: complement
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(array_expr_plan_t) :: plan
        type(lr_operand_desc_t), allocatable :: selected(:)
        type(lr_operand_desc_t) :: cond, rhs_val, cur_val
        integer :: target_symbol, value_index, i, n

        call set_empty(error_msg)
        call where_assignment_target(arena, assign_index, context, &
                                     target_symbol, value_index, error_msg)
        if (len_trim(error_msg) > 0) return

        call build_array_expression(context, target_symbol, value_index, plan, &
                                    error_msg)
        if (len_trim(error_msg) > 0) return
        n = plan%element_count()
        allocate (selected(n))

        do i = 0, n - 1
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
            if (complement) then
                call select_value(context, cond, cur_val, rhs_val, &
                                  selected(i + 1), error_msg)
            else
                call select_value(context, cond, rhs_val, cur_val, &
                                  selected(i + 1), error_msg)
            end if
            if (len_trim(error_msg) > 0) return
        end do

        do i = 0, n - 1
            call store_array_linear_element(context, target_symbol, &
                                            int(i, c_int64_t), selected(i + 1), &
                                            error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine lower_where_selected_assignment

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
            target_symbol = find_symbol_compat(context, name)
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
        type(lr_operand_desc_t) :: lhs_data, lhs_length
        type(lr_operand_desc_t) :: rhs_data, rhs_length, cmp
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

        if (value_kind == VALUE_CHARACTER) then
            call lower_where_character_operand_element(arena, bin_left, &
                linear_index, context, lhs_data, lhs_length, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_where_character_operand_element(arena, bin_right, &
                linear_index, context, rhs_data, rhs_length, error_msg)
            if (len_trim(error_msg) > 0) return
            call char_compare(context, lhs_data, lhs_length, rhs_data, &
                              rhs_length, cmp, error_msg)
            if (len_trim(error_msg) > 0) return
            call integer_compare_predicate(bin_op, ipred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, ipred, cmp, &
                                          i32_immediate(context%session, 0_c_int64_t), &
                                          cond, error_msg)) return
        else
            call lower_where_operand_element(arena, bin_left, value_kind, &
                                             linear_index, context, lhs, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_where_operand_element(arena, bin_right, value_kind, &
                                             linear_index, context, rhs, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

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
        else if (value_kind /= VALUE_CHARACTER) then
            call integer_compare_predicate(bin_op, ipred, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, ipred, lhs, rhs, &
                                          cond, error_msg)) return
        end if
    end subroutine lower_where_mask_element

    subroutine lower_where_character_operand_element(arena, node_index, &
                                                     linear_index, context, &
                                                     data_ptr, length, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: data_ptr
        type(lr_operand_desc_t), intent(out) :: length
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: symbol_index
        integer, allocatable :: flat(:)
        character(len=:), allocatable :: flat_error

        call set_empty(error_msg)
        symbol_index = where_array_operand_symbol(arena, node_index, context)
        if (symbol_index > 0 .and. context%symbols(symbol_index)%value_kind == &
            VALUE_CHARACTER) then
            call character_array_symbol_element_operands(context, symbol_index, &
                linear_index, data_ptr, length, error_msg)
            return
        end if

        select type (constructor => arena%entries(node_index)%node)
        type is (array_literal_node)
            if (.not. allocated(constructor%element_indices)) then
                error_msg = 'character array constructor has no elements'
                return
            end if
            allocate (flat(0))
            call flatten_constructor_elements(arena, constructor%element_indices, &
                                              flat, flat_error)
            if (len_trim(flat_error) > 0) then
                error_msg = flat_error
                return
            end if
            if (linear_index < 0 .or. linear_index >= size(flat)) then
                error_msg = 'character array constructor is shorter than its mask'
                return
            end if
            call char_expr_operands(arena, flat(linear_index + 1), context, &
                                    data_ptr, length, error_msg)
            return
        end select

        call char_expr_operands(arena, node_index, context, data_ptr, length, &
                                error_msg)
    end subroutine lower_where_character_operand_element

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
        sym = cached_array_expression_symbol(context, node_index)
        if (sym > 0) return
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, err)
        if (len_trim(err) > 0) return
        sym = find_symbol_compat(context, name)
        if (sym <= 0) then
            sym = 0
            return
        end if
        if (.not. is_elementwise_array_operand(context, sym)) sym = 0
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
            if (n%base_expr_index > 0 .and. &
                is_type_bound_method_call(arena, n, context)) then
                is_call = is_elemental_type_bound_call(arena, node_index, context)
                return
            end if
            if (n%is_array_access) return
            if (.not. allocated(n%name)) return
            if (.not. allocated(n%arg_indices)) return
            if (size(n%arg_indices) < 1) return
            if (index(n%name, '%') > 0) then
                is_call = is_elemental_type_bound_call(arena, node_index, context)
                return
            end if
            is_call = is_contained_f32_function(context, n%name) .or. &
                      is_contained_f64_function(context, n%name) .or. &
                      is_contained_i32_function(context, n%name)
            if (.not. is_call) return
            ! One dummy keeps the historical path (a contained scalar function
            ! applied element-wise). More than one dummy is only elementwise
            ! when the callee really is ELEMENTAL (#405).
            if (size(n%arg_indices) > 1) &
                is_call = elemental_multi_arg_callee(arena, n%name, &
                                                     size(n%arg_indices))
        end select
    end function is_elemental_user_call

    logical function is_elemental_type_bound_call(arena, node_index, context) &
        result(is_call)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: impl_name, signature_name
        integer :: receiver_symbol, binding_slot
        logical :: has_receiver

        call type_bound_elemental_info(arena, node_index, context, is_call, &
            impl_name, signature_name, receiver_symbol, binding_slot, &
            has_receiver)
    end function is_elemental_type_bound_call

    subroutine type_bound_elemental_info(arena, node_index, context, valid, &
                                         impl_name, signature_name, &
                                         receiver_symbol, binding_slot, &
                                         has_receiver)
        !! Return the concrete/interface signature and dispatch metadata for a
        !! type-bound elemental function.  The AST keeps an abstract binding's
        !! interface name separate from its eventual child override, so use the
        !! interface for the elemental contract while retaining the binding slot
        !! for dynamic dispatch.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        logical, intent(out) :: valid
        character(len=:), allocatable, intent(out) :: impl_name, signature_name
        integer, intent(out) :: receiver_symbol, binding_slot
        logical, intent(out) :: has_receiver
        character(len=:), allocatable :: pass_name, method_name, var_name, err
        logical :: found, found_signature, is_elemental, all_intent_in
        character(len=:), allocatable :: bad_dummy
        integer :: base_var_index, type_index, symbol_index, expected_args
        integer :: i, j

        valid = .false.
        receiver_symbol = 0
        binding_slot = 0
        has_receiver = .true.
        call set_empty(impl_name)
        call set_empty(signature_name)
        if (.not. node_exists(arena, node_index)) return
        select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (.not. allocated(node%arg_indices)) return
            call method_call_info(arena, node, context, base_var_index, &
                impl_name, pass_name, found, err, receiver_symbol, binding_slot)
            if (len_trim(err) > 0 .or. .not. found) return
            select type (base => arena%entries(node%base_expr_index)%node)
            type is (component_access_node)
                if (.not. allocated(base%component_name)) return
                method_name = trim(base%component_name)
            class default
                return
            end select
            if (.not. is_identifier(arena, base_var_index)) return
            call get_identifier_name(arena, base_var_index, var_name, err)
            if (len_trim(err) > 0) return
            symbol_index = find_symbol_compat(context, var_name)
            if (symbol_index <= 0) return
            type_index = context%symbols(symbol_index)%derived_type_index
            if (type_index <= 0) return
            has_receiver = .true.
            do i = 1, arena%size
                if (.not. node_exists(arena, i)) cycle
                select type (derived => arena%entries(i)%node)
                type is (derived_type_node)
                    if (.not. allocated(derived%name)) cycle
                    if (.not. same_name(derived%name, &
                            context%derived_types(type_index)%name)) cycle
                    if (.not. allocated(derived%binding_indices)) cycle
                    do j = 1, size(derived%binding_indices)
                        if (.not. node_exists(arena, derived%binding_indices(j))) cycle
                        select type (binding => &
                            arena%entries(derived%binding_indices(j))%node)
                        type is (type_binding_node)
                            if (.not. allocated(binding%binding_name)) cycle
                            if (.not. same_name(binding%binding_name, method_name)) cycle
                            has_receiver = binding%pass_arg
                            if (allocated(binding%interface_name)) then
                                if (len_trim(binding%interface_name) > 0) &
                                    signature_name = trim(binding%interface_name)
                            end if
                            exit
                        end select
                    end do
                    exit
                end select
            end do
            if (len_trim(signature_name) == 0) signature_name = trim(impl_name)
            call contained_elemental_signature(arena, signature_name, &
                found_signature, expected_args, is_elemental, all_intent_in, &
                bad_dummy)
            if (.not. found_signature .or. .not. is_elemental) return
            if (has_receiver) expected_args = expected_args - 1
            valid = expected_args == size(node%arg_indices)
        end select
    end subroutine type_bound_elemental_info

    ! True when callee_name is a contained ELEMENTAL function taking exactly
    ! nargs dummies. Arity mismatches decline so the ordinary call path (and
    ! its diagnostics) stays in charge.
    logical function elemental_multi_arg_callee(arena, callee_name, nargs) &
        result(ok)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: nargs
        character(len=:), allocatable :: bad_dummy
        integer :: param_count
        logical :: found, is_elemental, all_intent_in

        ok = .false.
        call contained_elemental_signature(arena, callee_name, found, &
                                           param_count, is_elemental, &
                                           all_intent_in, bad_dummy)
        if (.not. found) return
        if (.not. is_elemental) return
        ok = param_count == nargs
    end function elemental_multi_arg_callee

    ! Reject an elemental call whose dummies violate the elemental
    ! restrictions: every dummy of an elemental function must be INTENT(IN)
    ! (F2018 15.8.1).
    subroutine check_elemental_dummy_attributes(arena, callee_name, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: bad_dummy
        integer :: param_count
        logical :: found, is_elemental, all_intent_in

        call set_empty(error_msg)
        call contained_elemental_signature(arena, callee_name, found, &
                                           param_count, is_elemental, &
                                           all_intent_in, bad_dummy)
        if (.not. found) return
        if (.not. is_elemental) return
        if (all_intent_in) return
        error_msg = 'elemental procedure "'//trim(callee_name)// &
                    '" requires INTENT(IN) for every dummy argument, but "'// &
                    trim(bad_dummy)//'" is not INTENT(IN)'
    end subroutine check_elemental_dummy_attributes

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
        type(lr_operand_desc_t) :: elem
        type(lr_operand_desc_t), allocatable :: arg(:)
        integer :: vk, i
        logical :: derived_actual

        call set_empty(error_msg)
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            if (n%base_expr_index > 0 .and. &
                is_type_bound_method_call(arena, n, context)) then
                call lower_elementwise_type_bound_user_call(arena, node_index, &
                    target_symbol_index, linear_index, context, value, error_msg)
                return
            end if
            vk = context%symbols(target_symbol_index)%value_kind
            call check_elemental_dummy_attributes(arena, n%name, error_msg)
            if (len_trim(error_msg) > 0) return
            allocate (arg(size(n%arg_indices)))
            do i = 1, size(n%arg_indices)
                ! A derived or polymorphic actual is passed by reference at its
                ! own element stride, not loaded as a target-kind scalar (#369).
                call elementwise_derived_actual_address(arena, n%arg_indices(i), &
                                                        n%name, linear_index, &
                                                        context, arg(i), &
                                                        derived_actual, error_msg)
                if (len_trim(error_msg) > 0) return
                if (derived_actual) cycle
                ! Array actuals contribute their own element; scalar actuals
                ! broadcast (both handled by the shared element iterator).
                call lower_array_elementwise_value(arena, n%arg_indices(i), &
                                                   target_symbol_index, &
                                                   linear_index, context, elem, &
                                                   error_msg)
                if (len_trim(error_msg) > 0) return
                call make_reference_argument(context, vk, elem, arg(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            select case (vk)
            case (VALUE_F32)
                if (.not. emit_liric_f32_call(context%session, &
                        call_emit_name(arena, n%name, context), arg, value, error_msg)) return
            case (VALUE_I32)
                if (.not. emit_i32_call(context%session, &
                        call_emit_name(arena, n%name, context), arg, value, error_msg)) return
            case default
                if (.not. emit_liric_f64_call(context%session, &
                        call_emit_name(arena, n%name, context), arg, value, error_msg)) return
            end select
        class default
            error_msg = 'elemental user call node lost its type'
        end select
    end subroutine lower_elementwise_user_call

    subroutine lower_elementwise_type_bound_user_call(arena, node_index, &
                                                      target_symbol_index, &
                                                      linear_index, context, value, &
                                                      error_msg)
        !! Lower a NOPASS type-bound elemental call over one array element. The
        !! receiver still selects the dynamic vtable entry even though it is not
        !! passed to the implementation.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index, target_symbol_index, linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: args(:)
        type(lr_operand_desc_t) :: elem, callee_ptr
        character(len=:), allocatable :: impl_name, signature_name
        integer :: receiver_symbol, binding_slot, vk, i
        logical :: valid, has_receiver

        call set_empty(error_msg)
        call type_bound_elemental_info(arena, node_index, context, valid, &
            impl_name, signature_name, receiver_symbol, binding_slot, has_receiver)
        if (.not. valid) then
            error_msg = 'type-bound array expression is not an elemental function'
            return
        end if
        if (has_receiver) then
            error_msg = 'elemental type-bound calls with a passed object are not '// &
                        'supported in array SOURCE expressions'
            return
        end if
        vk = context%symbols(target_symbol_index)%value_kind
        call check_elemental_dummy_attributes(arena, signature_name, error_msg)
        if (len_trim(error_msg) > 0) return
        select type (n => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
            allocate (args(size(n%arg_indices)))
            do i = 1, size(n%arg_indices)
                call lower_array_elementwise_value(arena, n%arg_indices(i), &
                    target_symbol_index, linear_index, context, elem, error_msg)
                if (len_trim(error_msg) > 0) return
                call make_reference_argument(context, vk, elem, args(i), error_msg)
                if (len_trim(error_msg) > 0) return
            end do
        class default
            error_msg = 'type-bound elemental call node lost its type'
            return
        end select

        if (receiver_symbol > 0 .and. binding_slot > 0 .and. &
            class_receiver_dispatches_dynamically(context, receiver_symbol)) then
            call class_dispatch_callee(context, receiver_symbol, binding_slot, &
                                       callee_ptr, error_msg)
            if (len_trim(error_msg) > 0) return
            if (vk /= VALUE_I32) then
                error_msg = 'dynamic type-bound elemental calls currently require '// &
                            'an INTEGER result'
                return
            end if
            if (.not. emit_i32_indirect_call(context%session, callee_ptr, args, &
                                            value, error_msg)) return
        else
            select case (vk)
            case (VALUE_I32)
                if (.not. emit_i32_call(context%session, &
                        call_emit_name(arena, impl_name, context), args, value, &
                        error_msg)) return
            case (VALUE_F32)
                if (.not. emit_liric_f32_call(context%session, &
                        call_emit_name(arena, impl_name, context), args, value, &
                        error_msg)) return
            case default
                if (.not. emit_liric_f64_call(context%session, &
                        call_emit_name(arena, impl_name, context), args, value, &
                        error_msg)) return
            end select
        end if
    end subroutine lower_elementwise_type_bound_user_call

    ! Address of element linear_index of a derived-type array actual, for an
    ! elemental call whose dummy is a scalar type(...) or class(...) argument
    ! (#369). The stride is the derived type's own slot count, so the callee
    ! receives the same base address it would get from a scalar actual.
    ! handled stays .false. for any other actual, leaving the caller's
    ! intrinsic-kind path in charge.
    subroutine elementwise_derived_actual_address(arena, node_index, &
                                                  callee_name, linear_index, &
                                                  context, address, handled, &
                                                  error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: callee_name
        integer, intent(in) :: linear_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: address
        logical, intent(out) :: handled
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: id_name
        integer :: symbol_index, slots

        handled = .false.
        call set_empty(error_msg)
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, id_name, error_msg)
        if (len_trim(error_msg) > 0) return
        symbol_index = find_symbol_compat(context, id_name)
        if (symbol_index <= 0) return
        if (.not. context%symbols(symbol_index)%is_derived) return
        if (.not. context%symbols(symbol_index)%is_array) return
        if (context%symbols(symbol_index)%is_allocatable) return
        call check_elemental_dummy_conformance(arena, callee_name, context, &
            context%symbols(symbol_index)%derived_type_index, error_msg)
        if (len_trim(error_msg) > 0) return
        slots = total_component_slots(context, &
                                      context%symbols(symbol_index)%derived_type_index)
        if (slots <= 0) return
        if (.not. emit_i32_array_element_addr(context%session, &
                context%symbols(symbol_index)%array_size, &
                context%symbols(symbol_index)%element_address, &
                i32_immediate(context%session, &
                              int(linear_index, c_int64_t)*int(slots, c_int64_t)), &
                address, error_msg)) return
        call wrap_elemental_class_element(arena, callee_name, context, &
            context%symbols(symbol_index)%derived_type_index, address, error_msg)
        if (len_trim(error_msg) > 0) return
        handled = .true.
    end subroutine elementwise_derived_actual_address

    subroutine wrap_elemental_class_element(arena, callee_name, context, &
                                            actual_type_index, address, error_msg)
        ! An elemental procedure with a class(t) scalar dummy takes the same
        ! borrowed class descriptor a non-elemental class dummy takes, one per
        ! element. The array is monomorphic, so every element's dynamic type is
        ! the array's declared type. A type(t) dummy keeps the bare address.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: actual_type_index
        type(lr_operand_desc_t), intent(inout) :: address
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: descriptor
        integer :: declared_index

        call set_empty(error_msg)
        call class_dummy_declared_type_index(context, arena, callee_name, 1, &
                                             declared_index)
        if (declared_index <= 0) return
        call emit_class_scalar_descriptor(context, address, declared_index, &
            i64_immediate(context%session, &
                int(actual_type_index, c_int64_t)), descriptor, error_msg)
        if (len_trim(error_msg) > 0) return
        address = descriptor
    end subroutine wrap_elemental_class_element

    ! Reject a derived actual whose declared type is neither the elemental
    ! dummy's declared type nor an extension of it (#369). Silence when the
    ! dummy's declared type cannot be determined, so unrelated calls keep
    ! their existing behaviour.
    subroutine check_elemental_dummy_conformance(arena, callee_name, context, &
                                                 actual_type_index, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        type(lowering_context_t), intent(in) :: context
        integer, intent(in) :: actual_type_index
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: dummy_type
        integer :: type_index, depth

        call set_empty(error_msg)
        if (actual_type_index <= 0) return
        call elemental_dummy_declared_type(arena, callee_name, dummy_type)
        if (len_trim(dummy_type) == 0) return
        type_index = actual_type_index
        depth = 0
        do while (type_index > 0 .and. depth <= context%derived_type_count)
            if (same_name(trim(context%derived_types(type_index)%name), &
                          dummy_type)) return
            if (len_trim(context%derived_types(type_index)%parent_name) == 0) exit
            type_index = find_derived_type(context, &
                trim(context%derived_types(type_index)%parent_name))
            depth = depth + 1
        end do
        error_msg = 'incompatible actual argument for elemental procedure "'// &
                    trim(callee_name)//'": declared type '// &
                    trim(context%derived_types(actual_type_index)%name)// &
                    ' is not '//trim(dummy_type)//' or an extension of it'
    end subroutine check_elemental_dummy_conformance

    ! Declared derived-type name of the first dummy of the named procedure,
    ! for a type(name) or class(name) declaration. Empty when the procedure
    ! or its dummy declaration is not visible in the arena.
    subroutine elemental_dummy_declared_type(arena, callee_name, type_name)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: callee_name
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable :: dummy_name, err
        integer :: i

        type_name = ''
        do i = 1, arena%size
            if (.not. allocated(arena%entries(i)%node)) cycle
            select type (fn => arena%entries(i)%node)
            type is (function_def_node)
                if (.not. allocated(fn%name)) cycle
                if (.not. same_name(fn%name, callee_name)) cycle
                if (.not. allocated(fn%param_indices)) cycle
                if (size(fn%param_indices) /= 1) cycle
                if (.not. allocated(fn%body_indices)) cycle
                call parameter_name(arena, fn%param_indices(1), dummy_name, err)
                if (len_trim(err) > 0) cycle
                call declared_derived_type_of(arena, fn%body_indices, &
                                              dummy_name, type_name)
                if (len_trim(type_name) > 0) return
            end select
        end do
    end subroutine elemental_dummy_declared_type

    subroutine declared_derived_type_of(arena, body_indices, dummy_name, &
                                        type_name)
        ! Derived-type name declared for dummy_name by a type(name)/class(name)
        ! declaration among body_indices; empty when there is none.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: body_indices(:)
        character(len=*), intent(in) :: dummy_name
        character(len=:), allocatable, intent(out) :: type_name
        character(len=:), allocatable :: text, base
        integer :: i, open_paren, close_paren

        type_name = ''
        do i = 1, size(body_indices)
            if (.not. node_exists(arena, body_indices(i))) cycle
            select type (decl => arena%entries(body_indices(i))%node)
            type is (declaration_node)
                if (.not. allocated(decl%var_name)) cycle
                if (.not. same_name(decl%var_name, dummy_name)) cycle
                if (.not. allocated(decl%type_name)) cycle
                text = lowercase_text(trim(adjustl(decl%type_name)))
                open_paren = index(text, '(')
                close_paren = index(text, ')', back=.true.)
                if (open_paren <= 1) return
                if (close_paren <= open_paren + 1) return
                base = trim(adjustl(text(1:open_paren - 1)))
                if (base /= 'class' .and. base /= 'type') return
                type_name = trim(adjustl(text(open_paren + 1:close_paren - 1)))
                if (type_name == '*') type_name = ''
                return
            end select
        end do
    end subroutine declared_derived_type_of

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
        else if (same_name(name, 'aint')) then
            libm32 = 'truncf'; libm64 = 'trunc'
        else if (same_name(name, 'anint')) then
            ! ANINT rounds half away from zero, matching C round().
            libm32 = 'roundf'; libm64 = 'round'
        end if
    end subroutine elemental_real_libm_names
