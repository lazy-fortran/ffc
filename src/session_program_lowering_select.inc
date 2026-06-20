    subroutine lower_select_case(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: selector
        type(symbol_t), allocatable :: baseline(:)
        type(branch_result_t), allocatable :: results(:)
        integer(c_int32_t) :: default_block, merge_block
        integer :: case_idx, baseline_count, nresults
        logical :: captured

        if (node%selector_index <= 0) then
            error_msg = 'select case requires a selector expression'
            return
        end if
        if (.not. allocated(node%case_indices)) then
            call unsupported_feature_error('select case statement', node%line, &
                                           node%column, &
                                           'select case requires at least one '// &
                                           'case arm', error_msg)
            return
        end if
        if (node%default_index <= 0) then
            call unsupported_feature_error('select case statement', node%line, &
                                           node%column, &
                                           'select case requires a case default arm', &
                                           error_msg)
            return
        end if

        if (is_character_selector(arena, node%selector_index, context)) then
            call lower_select_case_character(arena, node, context, error_msg)
            return
        end if

        call lower_i32_expression(arena, node%selector_index, context, &
                                   selector, error_msg)
        if (len_trim(error_msg) > 0) return

        default_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        call snapshot_symbols(context, baseline, baseline_count)
        allocate (results(size(node%case_indices) + 1))
        nresults = 0

        do case_idx = 1, size(node%case_indices)
            call lower_select_case_arm(arena, node%case_indices(case_idx), &
                                       case_idx == size(node%case_indices), &
                                       selector, default_block, merge_block, &
                                       baseline, baseline_count, context, &
                                       results(nresults + 1), captured, error_msg)
            if (len_trim(error_msg) > 0) return
            if (captured) nresults = nresults + 1
        end do

        call lower_select_case_default(arena, node%default_index, default_block, &
                                       merge_block, baseline, baseline_count, &
                                       context, results(nresults + 1), captured, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        if (captured) nresults = nresults + 1

        call finish_select_merge(context, merge_block, results, nresults, error_msg)
    end subroutine lower_select_case

    subroutine snapshot_symbols(context, baseline, baseline_count)
        type(lowering_context_t), intent(in) :: context
        type(symbol_t), allocatable, intent(out) :: baseline(:)
        integer, intent(out) :: baseline_count
        integer :: i

        allocate (baseline(context%symbol_count))
        do i = 1, context%symbol_count
            baseline(i) = context%symbols(i)
        end do
        baseline_count = context%symbol_count
    end subroutine snapshot_symbols

    subroutine restore_symbols(context, baseline, baseline_count)
        type(lowering_context_t), intent(inout) :: context
        type(symbol_t), intent(in) :: baseline(:)
        integer, intent(in) :: baseline_count
        integer :: i

        do i = 1, baseline_count
            context%symbols(i) = baseline(i)
        end do
        context%symbol_count = baseline_count
    end subroutine restore_symbols

    subroutine capture_select_result(context, terminated, result)
        type(lowering_context_t), intent(in) :: context
        logical, intent(in) :: terminated
        type(branch_result_t), intent(out) :: result
        integer :: i

        allocate (result%symbols(context%symbol_count))
        do i = 1, context%symbol_count
            result%symbols(i) = context%symbols(i)
        end do
        result%symbol_count = context%symbol_count
        result%predecessor_block_id = context%current_block_id
        result%terminated = terminated
    end subroutine capture_select_result

    subroutine finish_select_merge(context, merge_block, results, nresults, &
                                   error_msg)
        ! Set the merge block live and phi the carried symbols across every
        ! non-terminated arm/default predecessor. If all arms terminated, the
        ! merge block is unreachable and the construct terminates.
        type(lowering_context_t), intent(inout) :: context
        integer(c_int32_t), intent(in) :: merge_block
        type(branch_result_t), intent(in) :: results(:)
        integer, intent(in) :: nresults
        character(len=:), allocatable, intent(out) :: error_msg

        if (nresults == 0) then
            context%current_block_terminated = .true.
            call set_empty(error_msg)
            return
        end if

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.

        call merge_select_results(context, results, nresults, error_msg)
    end subroutine finish_select_merge

    subroutine merge_select_results(context, results, nresults, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(in) :: results(:)
        integer, intent(in) :: nresults
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, k
        logical :: all_same

        context%symbol_count = results(1)%symbol_count
        do i = 1, results(1)%symbol_count
            do k = 2, nresults
                if (results(k)%symbol_count /= results(1)%symbol_count .or. &
                    trim(results(k)%symbols(i)%name) /= &
                    trim(results(1)%symbols(i)%name) .or. &
                    results(k)%symbols(i)%value_kind /= &
                    results(1)%symbols(i)%value_kind) then
                    error_msg = 'select case arms have divergent symbols'
                    return
                end if
            end do

            context%symbols(i) = results(1)%symbols(i)
            if (nresults == 1) cycle

            all_same = .true.
            do k = 2, nresults
                if (.not. same_operand(results(k)%symbols(i)%value, &
                                       results(1)%symbols(i)%value)) &
                    all_same = .false.
            end do
            if (all_same) cycle

            call phi_select_symbol(context, results, nresults, i, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    end subroutine merge_select_results

    subroutine phi_select_symbol(context, results, nresults, index, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(in) :: results(:)
        integer, intent(in) :: nresults
        integer, intent(in) :: index
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t), allocatable :: values(:)
        integer(c_int32_t), allocatable :: blocks(:)
        integer :: k

        select case (results(1)%symbols(index)%value_kind)
        case (VALUE_DERIVED)
            ! Derived-type symbols live in a stable alloca; no phi needed.
            context%symbols(index) = results(1)%symbols(index)
            call set_empty(error_msg)
            return
        case (VALUE_I32, VALUE_LOGICAL, VALUE_F64, VALUE_CHARACTER)
            allocate (values(nresults), blocks(nresults))
            do k = 1, nresults
                values(k) = results(k)%symbols(index)%value
                blocks(k) = results(k)%predecessor_block_id
            end do
            if (.not. emit_liric_phi_n(context%session, values, blocks, &
                    context%symbols(index)%value, error_msg)) return
            if (results(1)%symbols(index)%value_kind == VALUE_CHARACTER) then
                context%symbols(index)%is_deferred_character = &
                    results(1)%symbols(index)%is_deferred_character
                context%symbols(index)%deferred_data = &
                    results(1)%symbols(index)%deferred_data
                context%symbols(index)%deferred_length = &
                    results(1)%symbols(index)%deferred_length
                context%symbols(index)%character_length = &
                    results(1)%symbols(index)%character_length
                context%symbols(index)%has_character_value = &
                    results(1)%symbols(index)%has_character_value
            end if
        case default
            error_msg = 'select case cannot merge unknown symbol kind'
            return
        end select
        call set_empty(error_msg)
    end subroutine phi_select_symbol

    logical function is_character_selector(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index
        character(len=:), allocatable :: id_name, id_err

        is_character_selector = .false.
        if (.not. node_exists(arena, node_index)) return
        if (is_literal(arena, node_index)) then
            is_character_selector = is_character_literal(arena, node_index)
            return
        end if
        if (is_identifier(arena, node_index)) then
            call get_identifier_name(arena, node_index, id_name, id_err)
            symbol_index = find_symbol(context, id_name)
            if (symbol_index > 0) is_character_selector = &
                context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
        end if
    end function is_character_selector

    subroutine lower_select_case_character(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: sel_data, sel_len
        type(symbol_t), allocatable :: baseline(:)
        type(branch_result_t), allocatable :: results(:)
        integer(c_int32_t) :: default_block, merge_block
        integer :: case_idx, baseline_count, nresults
        logical :: captured

        call char_expr_operands(arena, node%selector_index, context, sel_data, &
                                sel_len, error_msg)
        if (len_trim(error_msg) > 0) return

        default_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        call snapshot_symbols(context, baseline, baseline_count)
        allocate (results(size(node%case_indices) + 1))
        nresults = 0

        do case_idx = 1, size(node%case_indices)
            call lower_select_case_character_arm(arena, &
                node%case_indices(case_idx), &
                case_idx == size(node%case_indices), sel_data, sel_len, &
                default_block, merge_block, baseline, baseline_count, context, &
                results(nresults + 1), captured, error_msg)
            if (len_trim(error_msg) > 0) return
            if (captured) nresults = nresults + 1
        end do

        call lower_select_case_default(arena, node%default_index, default_block, &
                                       merge_block, baseline, baseline_count, &
                                       context, results(nresults + 1), captured, &
                                       error_msg)
        if (len_trim(error_msg) > 0) return
        if (captured) nresults = nresults + 1

        call finish_select_merge(context, merge_block, results, nresults, error_msg)
    end subroutine lower_select_case_character

    subroutine lower_select_case_character_arm(arena, case_node_index, &
            is_last_arm, sel_data, sel_len, default_block, merge_block, &
            baseline, baseline_count, context, result, captured, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: case_node_index
        logical, intent(in) :: is_last_arm
        type(lr_operand_desc_t), intent(in) :: sel_data, sel_len
        integer(c_int32_t), intent(in) :: default_block
        integer(c_int32_t), intent(in) :: merge_block
        type(symbol_t), intent(in) :: baseline(:)
        integer, intent(in) :: baseline_count
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(out) :: result
        logical, intent(out) :: captured
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: arm_block, next_check_block, false_target
        type(lr_operand_desc_t) :: dummy_value
        type(lr_operand_desc_t) :: lbl_data, lbl_len, eq, cond
        logical :: terminated
        integer :: i
        integer, allocatable :: value_indices(:), body(:)

        captured = .false.
        call get_case_block_info(arena, case_node_index, value_indices, body)
        if (.not. allocated(value_indices)) then
            error_msg = 'select case arm node has unexpected kind'
            return
        end if
        arm_block = create_liric_block(context%session)
        if (is_last_arm) then
            next_check_block = default_block
        else
            next_check_block = create_liric_block(context%session)
        end if
        ! Match the selector against each character label in turn.
        do i = 1, size(value_indices)
            call char_expr_operands(arena, value_indices(i), context, &
                                    lbl_data, lbl_len, error_msg)
            if (len_trim(error_msg) > 0) return
            call char_equals(context, sel_data, sel_len, lbl_data, lbl_len, &
                             eq, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, eq, &
                    i32_immediate(context%session, 0_c_int64_t), cond, &
                    error_msg)) return
            if (i == size(value_indices)) then
                false_target = next_check_block
            else
                false_target = create_liric_block(context%session)
            end if
            if (.not. emit_liric_condbr(context%session, cond, arm_block, &
                    false_target, error_msg)) return
            if (i < size(value_indices)) then
                if (.not. set_liric_block(context%session, false_target, &
                        error_msg)) return
                context%current_block_id = false_target
                context%current_block_terminated = .false.
            end if
        end do

        if (.not. set_liric_block(context%session, arm_block, error_msg)) &
            return
        context%current_block_id = arm_block
        context%current_block_terminated = .false.
        call restore_symbols(context, baseline, baseline_count)
        terminated = .false.
        if (allocated(body)) then
            call lower_statement_list(arena, body, context, &
                                      dummy_value, terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call capture_select_result(context, terminated, result)
        if (.not. terminated) then
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                return
            captured = .true.
        end if
        if (.not. is_last_arm) then
            if (.not. set_liric_block(context%session, next_check_block, &
                    error_msg)) return
            context%current_block_id = next_check_block
            context%current_block_terminated = .false.
        end if
        call set_empty(error_msg)
    end subroutine lower_select_case_character_arm

    subroutine lower_select_case_arm(arena, case_node_index, is_last_arm, &
                                     selector, default_block, merge_block, &
                                     baseline, baseline_count, context, result, &
                                     captured, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: case_node_index
        logical, intent(in) :: is_last_arm
        type(lr_operand_desc_t), intent(in) :: selector
        integer(c_int32_t), intent(in) :: default_block
        integer(c_int32_t), intent(in) :: merge_block
        type(symbol_t), intent(in) :: baseline(:)
        integer, intent(in) :: baseline_count
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(out) :: result
        logical, intent(out) :: captured
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dummy_value
        integer(c_int32_t) :: arm_block
        integer(c_int32_t) :: next_check_block
        logical :: terminated
        integer, allocatable :: value_indices(:), body(:)

        captured = .false.
        call get_case_block_info(arena, case_node_index, value_indices, body)
        if (.not. allocated(value_indices)) then
            error_msg = 'select case arm node has unexpected kind'
            return
        end if
        if (size(value_indices) < 1) then
            call unsupported_feature_error('select case statement', &
                                           0, 0, &
                                           'case arm requires at least one label', &
                                           error_msg)
            return
        end if

        arm_block = create_liric_block(context%session)
        if (is_last_arm) then
            next_check_block = default_block
        else
            next_check_block = create_liric_block(context%session)
        end if

        call emit_case_label_dispatch(arena, value_indices, selector, &
                                      arm_block, next_check_block, &
                                      context, error_msg)
        if (len_trim(error_msg) > 0) return

        if (.not. set_liric_block(context%session, arm_block, &
                                  error_msg)) return
        context%current_block_id = arm_block
        context%current_block_terminated = .false.
        call restore_symbols(context, baseline, baseline_count)
        terminated = .false.
        if (allocated(body)) then
            call lower_statement_list(arena, body, context, &
                                      dummy_value, terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call capture_select_result(context, terminated, result)
        if (.not. terminated) then
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                return
            captured = .true.
        end if

        if (.not. is_last_arm) then
            if (.not. set_liric_block(context%session, next_check_block, &
                                      error_msg)) return
            context%current_block_id = next_check_block
            context%current_block_terminated = .false.
        end if
        call set_empty(error_msg)
    end subroutine lower_select_case_arm

    subroutine lower_select_case_default(arena, default_idx, default_block, &
                                         merge_block, baseline, baseline_count, &
                                         context, result, captured, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: default_idx
        integer(c_int32_t), intent(in) :: default_block
        integer(c_int32_t), intent(in) :: merge_block
        type(symbol_t), intent(in) :: baseline(:)
        integer, intent(in) :: baseline_count
        type(lowering_context_t), intent(inout) :: context
        type(branch_result_t), intent(out) :: result
        logical, intent(out) :: captured
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dummy_value
        logical :: terminated
        integer, allocatable :: body(:)

        captured = .false.
        call get_case_default_body(arena, default_idx, body)
        if (.not. set_liric_block(context%session, default_block, &
                                  error_msg)) return
        context%current_block_id = default_block
        context%current_block_terminated = .false.
        call restore_symbols(context, baseline, baseline_count)

        terminated = .false.
        if (allocated(body)) then
            call lower_statement_list(arena, body, context, &
                                      dummy_value, terminated, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call capture_select_result(context, terminated, result)
        if (.not. terminated) then
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                return
            captured = .true.
        end if
        call set_empty(error_msg)
    end subroutine lower_select_case_default

    subroutine emit_case_label_dispatch(arena, value_indices, selector, &
                                        arm_block, no_match_block, context, &
                                        error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_indices(:)
        type(lr_operand_desc_t), intent(in) :: selector
        integer(c_int32_t), intent(in) :: arm_block
        integer(c_int32_t), intent(in) :: no_match_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: false_target
        integer :: i

        do i = 1, size(value_indices)
            if (i == size(value_indices)) then
                false_target = no_match_block
            else
                false_target = create_liric_block(context%session)
            end if
            call emit_one_case_label(arena, value_indices(i), selector, &
                                     arm_block, false_target, context, error_msg)
            if (len_trim(error_msg) > 0) return
            if (i < size(value_indices)) then
                if (.not. set_liric_block(context%session, false_target, &
                                          error_msg)) return
                context%current_block_id = false_target
                context%current_block_terminated = .false.
            end if
        end do
        call set_empty(error_msg)
    end subroutine emit_case_label_dispatch

    subroutine emit_one_case_label(arena, value_index, selector, arm_block, &
                                   no_match_block, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: value_index
        type(lr_operand_desc_t), intent(in) :: selector
        integer(c_int32_t), intent(in) :: arm_block
        integer(c_int32_t), intent(in) :: no_match_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: bound_value
        type(lr_operand_desc_t) :: cond
        integer(c_int32_t) :: hi_check_block
        integer :: start_value, end_value

        call get_case_range_info(arena, value_index, start_value, end_value)
        if (start_value > 0 .or. end_value > 0) then
            ! lo: -> selector >= lo ; :hi -> selector <= hi ;
            ! lo:hi -> selector >= lo AND selector <= hi (chained checks)
            if (start_value > 0 .and. end_value > 0) then
                hi_check_block = create_liric_block(context%session)
                call emit_case_bound(arena, start_value, selector, &
                                     LR_CMP_SGE, hi_check_block, &
                                     no_match_block, context, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. set_liric_block(context%session, hi_check_block, &
                                          error_msg)) return
                context%current_block_id = hi_check_block
                context%current_block_terminated = .false.
                call emit_case_bound(arena, end_value, selector, &
                                     LR_CMP_SLE, arm_block, no_match_block, &
                                     context, error_msg)
            else if (start_value > 0) then
                call emit_case_bound(arena, start_value, selector, &
                                     LR_CMP_SGE, arm_block, no_match_block, &
                                     context, error_msg)
            else if (end_value > 0) then
                call emit_case_bound(arena, end_value, selector, &
                                     LR_CMP_SLE, arm_block, no_match_block, &
                                     context, error_msg)
            end if
            call set_empty(error_msg)
            return
        end if

        ! Literal label: selector == value
        call lower_i32_expression(arena, value_index, context, bound_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, selector, &
                                      bound_value, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, arm_block, &
                                    no_match_block, error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_one_case_label

    subroutine emit_case_bound(arena, bound_index, selector, predicate, &
                               match_block, no_match_block, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: bound_index
        type(lr_operand_desc_t), intent(in) :: selector
        integer(c_int), intent(in) :: predicate
        integer(c_int32_t), intent(in) :: match_block
        integer(c_int32_t), intent(in) :: no_match_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: bound_value
        type(lr_operand_desc_t) :: cond

        call lower_i32_expression(arena, bound_index, context, bound_value, &
                                  error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_i32_icmp(context%session, predicate, selector, &
                                      bound_value, cond, error_msg)) return
        if (.not. emit_liric_condbr(context%session, cond, match_block, &
                                    no_match_block, error_msg)) return
        call set_empty(error_msg)
    end subroutine emit_case_bound

    subroutine lower_select_type(arena, node_index, context, error_msg)
        ! Lower select type on a class(*) selector: load the descriptor's
        ! runtime type id and branch to the matching `type is` arm; within the
        ! arm the selector is rebound to its concrete typed value (#142, #143).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: sel_name, assoc_name
        integer :: sel_index, g
        type(lr_operand_desc_t) :: sel_type_id, data_ptr, cond, guard_id
        integer(c_int32_t) :: arm_block, next_block, merge_block
        type(symbol_t), allocatable :: baseline(:)
        integer :: baseline_count
        logical :: terminated
        integer :: selector_idx
        integer, allocatable :: guard_indices(:)
        integer :: default_idx
        type(branch_result_t), allocatable :: results(:)
        integer :: nresults
        logical :: captured

        call set_empty(error_msg)
        call get_select_type_info(arena, node_index, selector_idx, &
                                  guard_indices, default_idx)
        ! Two forms: select type (var) and select type (assoc => var). The
        ! latter is parsed as an assignment node {target = assoc, value = var}.
        call select_type_selector(arena, selector_idx, sel_name, &
                                  assoc_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sel_index = find_symbol(context, sel_name)
        if (sel_index <= 0) then
            error_msg = 'select type selector was not declared: '//trim(sel_name)
            return
        end if
        if (context%symbols(sel_index)%value_kind /= VALUE_CLASS_STAR) then
            error_msg = 'select type selector is not class(*): '//trim(sel_name)
            return
        end if

        ! type id (low word of the i64 type slot) and the boxed data pointer.
        if (.not. emit_i32_load(context%session, &
            context%symbols(sel_index)%deferred_length, sel_type_id, error_msg)) &
            return
        if (.not. emit_ptr_load(context%session, &
            context%symbols(sel_index)%deferred_data, data_ptr, error_msg)) return

        merge_block = create_liric_block(context%session)
        ! Each arm is a separate predecessor of the merge block; carry every
        ! baseline symbol's per-arm value across so an SSA scalar reassigned in
        ! one arm (e.g. i = 1) reaches the trailing statement through a phi,
        ! exactly as select case does. The associate name is local to each arm
        ! and is restored to baseline before capture so it never diverges.
        call snapshot_symbols(context, baseline, baseline_count)
        nresults = 0
        if (allocated(guard_indices)) then
            allocate (results(size(guard_indices) + 1))
        else
            allocate (results(1))
        end if

        if (allocated(guard_indices)) then
            do g = 1, size(guard_indices)
                call select_type_guard_id(arena, guard_indices(g), context, &
                                          guard_id, error_msg)
                if (len_trim(error_msg) > 0) return
                arm_block = create_liric_block(context%session)
                next_block = create_liric_block(context%session)
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, &
                        sel_type_id, guard_id, cond, error_msg)) return
                if (.not. emit_liric_condbr(context%session, cond, arm_block, &
                                            next_block, error_msg)) return
                if (.not. set_liric_block(context%session, arm_block, &
                                          error_msg)) return
                context%current_block_id = arm_block
                context%current_block_terminated = .false.
                call restore_symbols(context, baseline, baseline_count)
                call lower_select_type_arm(arena, guard_indices(g), &
                    assoc_name, data_ptr, context, terminated, error_msg)
                if (len_trim(error_msg) > 0) return
                call select_type_capture_arm(context, baseline, baseline_count, &
                    terminated, results(nresults + 1), captured)
                if (captured) nresults = nresults + 1
                if (.not. terminated) then
                    if (.not. emit_liric_br(context%session, merge_block, &
                                            error_msg)) return
                end if
                if (.not. set_liric_block(context%session, next_block, &
                                          error_msg)) return
                context%current_block_id = next_block
                context%current_block_terminated = .false.
            end do
        end if

        call restore_symbols(context, baseline, baseline_count)
        if (default_idx > 0) then
            call lower_select_type_default(arena, default_idx, context, &
                                           terminated, error_msg)
            if (len_trim(error_msg) > 0) return
            call select_type_capture_arm(context, baseline, baseline_count, &
                terminated, results(nresults + 1), captured)
            if (captured) nresults = nresults + 1
            if (.not. terminated) then
                if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                    return
            end if
        else
            ! No default: the unguarded fall-through path branches straight to
            ! the merge carrying the (already-restored) baseline symbols.
            call capture_select_result(context, .false., results(nresults + 1))
            nresults = nresults + 1
            if (.not. emit_liric_br(context%session, merge_block, error_msg)) return
        end if

        call restore_symbols(context, baseline, baseline_count)
        call finish_select_merge(context, merge_block, results, nresults, &
                                 error_msg)
    end subroutine lower_select_type

    subroutine select_type_capture_arm(context, baseline, baseline_count, &
                                       terminated, result, captured)
        ! Capture a select type arm's symbol state for the merge phi, but only
        ! the baseline symbols: the associate name lives beyond baseline (or
        ! rebinds a baseline slot inside the arm) and must not enter the merge.
        ! A baseline slot rebound as the associate name is reverted first.
        type(lowering_context_t), intent(inout) :: context
        type(symbol_t), intent(in) :: baseline(:)
        integer, intent(in) :: baseline_count
        logical, intent(in) :: terminated
        type(branch_result_t), intent(out) :: result
        logical, intent(out) :: captured
        integer :: i

        captured = .not. terminated
        if (.not. captured) return
        ! Drop any associate-name symbol appended past the baseline and revert a
        ! baseline slot whose kind changed (the associate bound onto it), keeping
        ! the merge's symbol set identical across arms.
        do i = 1, baseline_count
            if (context%symbols(i)%value_kind /= baseline(i)%value_kind) &
                context%symbols(i) = baseline(i)
        end do
        context%symbol_count = baseline_count
        call capture_select_result(context, terminated, result)
    end subroutine select_type_capture_arm

    subroutine select_type_selector(arena, selector_index, sel_name, &
                                    assoc_name, error_msg)
        ! Resolve the class(*) variable name and the associate name.
        ! select type (v)        -> sel_name = v, assoc_name = v
        ! select type (a => v)   -> sel_name = v, assoc_name = a (parsed as an
        !                           assignment node {target=a, value=v})
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: selector_index
        character(len=:), allocatable, intent(out) :: sel_name
        character(len=:), allocatable, intent(out) :: assoc_name
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (is_identifier(arena, selector_index)) then
            call get_identifier_name(arena, selector_index, sel_name, error_msg)
            assoc_name = sel_name
            return
        end if
        if (.not. node_exists(arena, selector_index)) then
            error_msg = 'select type selector does not reference an AST node'
            return
        end if
        select type (sel => arena%entries(selector_index)%node)
        type is (pointer_assignment_node)
            ! name => variable: pointer_index is the associate name, target_index
            ! the selector variable.
            call get_identifier_name(arena, sel%pointer_index, assoc_name, error_msg)
            if (len_trim(error_msg) > 0) return
            call get_identifier_name(arena, sel%target_index, sel_name, error_msg)
        type is (assignment_node)
            call get_identifier_name(arena, sel%target_index, assoc_name, error_msg)
            if (len_trim(error_msg) > 0) return
            call get_identifier_name(arena, sel%value_index, sel_name, error_msg)
        class default
            error_msg = 'select type selector must be a class(*) variable or '// &
                        'an associate (name => variable)'
        end select
    end subroutine select_type_selector


    subroutine select_type_guard_id(arena, guard_index, context, guard_id, &
                                    error_msg)
        ! The runtime type id constant for a `type is (X)` guard.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: guard_id
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: value_kind
        integer(c_int64_t) :: id

        call select_type_guard_kind(arena, guard_index, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        select case (value_kind)
        case (VALUE_I32)
            id = int(TYPE_ID_INTEGER, c_int64_t)
        case (VALUE_F64)
            id = int(TYPE_ID_REAL, c_int64_t)
        case (VALUE_LOGICAL)
            id = int(TYPE_ID_LOGICAL, c_int64_t)
        case default
            error_msg = 'select type supports type is (integer|real|logical)'
            return
        end select
        guard_id = i32_immediate(context%session, id)
    end subroutine select_type_guard_id

    subroutine select_type_guard_kind(arena, guard_index, value_kind, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        integer, intent(out) :: value_kind
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: guard_type, type_name, terr
        integer :: tname_index
        integer, allocatable :: body_tmp(:)

        call set_empty(error_msg)
        value_kind = VALUE_I32
        guard_type = ''
        type_name = ''
        terr = ''
        call get_type_guard_info(arena, guard_index, guard_type, &
                                 tname_index, body_tmp)
        if (trim(guard_type) /= 'type_is') then
            error_msg = 'select type supports type is guards only'
            return
        end if
        if (is_identifier(arena, tname_index)) then
            call get_identifier_name(arena, tname_index, type_name, terr)
        else if (is_literal(arena, tname_index)) then
            block
                character(len=:), allocatable :: lit_type
                call get_literal_info(arena, tname_index, type_name, lit_type, terr)
            end block
        else
            error_msg = 'select type guard type name is not resolvable'
            return
        end if
        select case (trim(lowercase_text(type_name)))
        case ('integer')
            value_kind = VALUE_I32
        case ('real', 'double precision')
            value_kind = VALUE_F64
        case ('logical')
            value_kind = VALUE_LOGICAL
        case default
            error_msg = 'select type supports type is (integer|real|logical)'
        end select
    end subroutine select_type_guard_kind

    subroutine lower_select_type_arm(arena, guard_index, assoc_name, data_ptr, &
                                     context, terminated, error_msg)
        ! Bind the associate name to the selector's concrete typed value (loaded
        ! from the boxed data pointer), then lower the arm body.
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        character(len=*), intent(in) :: assoc_name
        type(lr_operand_desc_t), intent(in) :: data_ptr
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: terminated
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: value_kind, idx
        type(lr_operand_desc_t) :: typed_value, value
        integer, allocatable :: body(:)

        terminated = .false.
        call select_type_guard_kind(arena, guard_index, value_kind, error_msg)
        if (len_trim(error_msg) > 0) return
        if (value_kind == VALUE_F64) then
            if (.not. emit_liric_f64_load(context%session, data_ptr, &
                                          typed_value, error_msg)) return
        else
            if (.not. emit_i32_load(context%session, data_ptr, typed_value, &
                                    error_msg)) return
        end if
        idx = find_symbol(context, assoc_name)
        if (idx <= 0) then
            call grow_symbols(context)
            idx = context%symbol_count + 1
            context%symbol_count = idx
        end if
        context%symbols(idx)%name = trim(assoc_name)
        context%symbols(idx)%value_kind = value_kind
        context%symbols(idx)%value = typed_value
        context%symbols(idx)%has_address = .false.
        context%symbols(idx)%is_reference = .false.
        context%symbols(idx)%is_derived = .false.
        context%symbols(idx)%is_deferred_character = .false.

        call select_type_guard_body(arena, guard_index, body)
        call lower_statement_list(arena, body, context, value, terminated, &
                                  error_msg)
    end subroutine lower_select_type_arm

    subroutine select_type_guard_body(arena, guard_index, body)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: guard_index
        integer, allocatable, intent(out) :: body(:)
        character(len=:), allocatable :: guard_type
        integer :: tname_index

        guard_type = ''
        tname_index = 0
        call get_type_guard_info(arena, guard_index, guard_type, &
                                 tname_index, body)
    end subroutine select_type_guard_body

    subroutine lower_select_type_default(arena, default_index, context, &
                                         terminated, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: default_index
        type(lowering_context_t), intent(inout) :: context
        logical, intent(out) :: terminated
        character(len=:), allocatable, intent(out) :: error_msg
        integer, allocatable :: body(:)
        type(lr_operand_desc_t) :: value

        terminated = .false.
        call select_type_guard_body(arena, default_index, body)
        call lower_statement_list(arena, body, context, value, terminated, &
                                  error_msg)
    end subroutine lower_select_type_default

    subroutine lower_select_rank(arena, node, context, error_msg)
        ! A concrete-rank selector has a statically known rank, so exactly one
        ! arm is ever live. Resolve that rank from the selector's declaration,
        ! then lower only the matching `rank (n)` arm (or `rank default`)
        ! inline. Genuine assumed-rank `(..)` dummies would need a runtime rank
        ! descriptor and are rejected with a clean diagnostic.
        type(ast_arena_t), intent(in) :: arena
        type(select_rank_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: sel_name
        integer :: sel_index, selector_rank, chosen_body_index
        integer, allocatable :: body(:)
        type(lr_operand_desc_t) :: value
        logical :: terminated

        call set_empty(error_msg)
        if (node%selector_index <= 0 .or. &
            .not. is_identifier(arena, node%selector_index)) then
            call unsupported_feature_error('select rank statement', node%line, &
                node%column, 'select rank selector must be a named variable', &
                error_msg)
            return
        end if
        call get_identifier_name(arena, node%selector_index, sel_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sel_index = find_symbol(context, sel_name)
        if (sel_index <= 0) then
            call unsupported_feature_error('select rank statement', node%line, &
                node%column, 'select rank selector was not declared: '// &
                trim(sel_name), error_msg)
            return
        end if
        ! A scalar selector has rank 0; a declared array carries array_rank.
        ! Assumed-rank `(..)` dummies are not tracked with a static rank and
        ! land in the default branch below with a clean diagnostic.
        if (context%symbols(sel_index)%is_array) then
            selector_rank = context%symbols(sel_index)%array_rank
        else
            selector_rank = 0
        end if

        call select_rank_choose_body(arena, node, selector_rank, &
                                     chosen_body_index, error_msg)
        if (len_trim(error_msg) > 0) return
        if (chosen_body_index <= 0) then
            ! No matching arm and no default: the construct runs no body.
            return
        end if
        call select_rank_block_body(arena, chosen_body_index, body)
        if (.not. allocated(body)) return
        call lower_statement_list(arena, body, context, value, terminated, &
                                  error_msg)
    end subroutine lower_select_rank

    subroutine select_rank_choose_body(arena, node, selector_rank, &
                                       chosen_body_index, error_msg)
        ! Return the arena index of the rank_block whose rank matches the
        ! selector, else the `rank default` block index, else 0.
        type(ast_arena_t), intent(in) :: arena
        type(select_rank_node), intent(in) :: node
        integer, intent(in) :: selector_rank
        integer, intent(out) :: chosen_body_index
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, rank_value

        call set_empty(error_msg)
        chosen_body_index = 0
        if (allocated(node%rank_indices)) then
            do i = 1, size(node%rank_indices)
                call select_rank_block_value(arena, node%rank_indices(i), &
                                             rank_value, error_msg)
                if (len_trim(error_msg) > 0) return
                if (rank_value == selector_rank) then
                    chosen_body_index = node%rank_indices(i)
                    return
                end if
            end do
        end if
        if (node%default_index > 0) chosen_body_index = node%default_index
    end subroutine select_rank_choose_body

    subroutine select_rank_block_value(arena, block_index, rank_value, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: block_index
        integer, intent(out) :: rank_value
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        rank_value = -1
        if (.not. node_exists(arena, block_index)) then
            error_msg = 'select rank arm does not reference an AST node'
            return
        end if
        select type (blk => arena%entries(block_index)%node)
        type is (rank_block_node)
            rank_value = blk%rank_value
        class default
            error_msg = 'select rank arm node has unexpected kind'
        end select
    end subroutine select_rank_block_value

    subroutine select_rank_block_body(arena, block_index, body)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: block_index
        integer, allocatable, intent(out) :: body(:)

        if (.not. node_exists(arena, block_index)) return
        select type (blk => arena%entries(block_index)%node)
        type is (rank_block_node)
            if (allocated(blk%body_indices)) body = blk%body_indices
        end select
    end subroutine select_rank_block_body

    logical function allocate_targets_class_star(arena, node, context)
        ! True for `allocate(<type> :: x)` where x is a class(*) allocatable.
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(in) :: context
        character(len=:), allocatable :: name, err
        integer :: idx

        allocate_targets_class_star = .false.
        if (.not. allocated(node%type_spec)) return
        if (len_trim(node%type_spec) == 0) return
        if (.not. allocated(node%var_indices)) return
        if (size(node%var_indices) /= 1) return
        if (.not. is_identifier(arena, node%var_indices(1))) return
        call get_identifier_name(arena, node%var_indices(1), name, err)
        if (len_trim(err) > 0) return
        idx = find_symbol(context, name)
        if (idx <= 0) return
        allocate_targets_class_star = &
            context%symbols(idx)%value_kind == VALUE_CLASS_STAR
    end function allocate_targets_class_star

    subroutine lower_class_star_allocate(arena, node, context, error_msg)
        ! allocate(<type> :: x): malloc storage for the concrete type, point the
        ! descriptor's data slot at it, and record the runtime type id so a later
        ! select type dispatches to the matching arm (#273).
        type(ast_arena_t), intent(in) :: arena
        type(allocate_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: name
        integer :: idx, byte_size, type_id
        type(lr_operand_desc_t) :: data_ptr

        call set_empty(error_msg)
        call get_identifier_name(arena, node%var_indices(1), name, error_msg)
        if (len_trim(error_msg) > 0) return
        idx = find_symbol(context, name)
        if (idx <= 0) then
            error_msg = 'allocate target is not declared: '//trim(name)
            return
        end if
        call class_star_type_spec(node%type_spec, byte_size, type_id, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_malloc(context%session, &
            i64_immediate(context%session, int(byte_size, c_int64_t)), &
            data_ptr, error_msg)) return
        if (.not. emit_ptr_store(context%session, data_ptr, &
            context%symbols(idx)%deferred_data, error_msg)) return
        if (.not. emit_i64_store(context%session, &
            i64_immediate(context%session, int(type_id, c_int64_t)), &
            context%symbols(idx)%deferred_length, error_msg)) return
    end subroutine lower_class_star_allocate

    subroutine class_star_type_spec(type_spec, byte_size, type_id, error_msg)
        ! Map a class(*) allocate type-spec to its storage size and runtime id.
        character(len=*), intent(in) :: type_spec
        integer, intent(out) :: byte_size, type_id
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        select case (trim(lowercase_text(type_spec)))
        case ('integer')
            byte_size = 4
            type_id = TYPE_ID_INTEGER
        case ('real', 'double precision')
            byte_size = 8
            type_id = TYPE_ID_REAL
        case ('logical')
            byte_size = 4
            type_id = TYPE_ID_LOGICAL
        case default
            error_msg = 'allocate(<type> :: x) supports integer, real, '// &
                        'and logical for class(*): '//trim(type_spec)
        end select
    end subroutine class_star_type_spec
