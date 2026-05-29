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

        captured = .false.
        if (.not. node_exists(arena, case_node_index)) then
            error_msg = 'select case arm index does not reference an AST node'
            return
        end if
        select type (arm => arena%entries(case_node_index)%node)
        type is (case_block_node)
            if (.not. allocated(arm%value_indices)) then
                call unsupported_feature_error('select case statement', &
                    arm%line, arm%column, 'case arm requires a label', error_msg)
                return
            end if
            arm_block = create_liric_block(context%session)
            if (is_last_arm) then
                next_check_block = default_block
            else
                next_check_block = create_liric_block(context%session)
            end if
            ! Match the selector against each character label in turn.
            do i = 1, size(arm%value_indices)
                call char_expr_operands(arena, arm%value_indices(i), context, &
                                        lbl_data, lbl_len, error_msg)
                if (len_trim(error_msg) > 0) return
                call char_equals(context, sel_data, sel_len, lbl_data, lbl_len, &
                                 eq, error_msg)
                if (len_trim(error_msg) > 0) return
                if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, eq, &
                        i32_immediate(context%session, 0_c_int64_t), cond, &
                        error_msg)) return
                if (i == size(arm%value_indices)) then
                    false_target = next_check_block
                else
                    false_target = create_liric_block(context%session)
                end if
                if (.not. emit_liric_condbr(context%session, cond, arm_block, &
                        false_target, error_msg)) return
                if (i < size(arm%value_indices)) then
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
            if (allocated(arm%body_indices)) then
                call lower_statement_list(arena, arm%body_indices, context, &
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
        class default
            error_msg = 'select case arm node has unexpected kind'
            return
        end select
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

        captured = .false.
        if (.not. node_exists(arena, case_node_index)) then
            error_msg = 'select case arm index does not reference an AST node'
            return
        end if

        select type (arm => arena%entries(case_node_index)%node)
        type is (case_block_node)
            if (.not. allocated(arm%value_indices)) then
                call unsupported_feature_error('select case statement', &
                                               arm%line, arm%column, &
                                               'case arm requires a label', &
                                               error_msg)
                return
            end if
            if (size(arm%value_indices) < 1) then
                call unsupported_feature_error('select case statement', &
                                               arm%line, arm%column, &
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

            call emit_case_label_dispatch(arena, arm%value_indices, selector, &
                                          arm_block, next_check_block, &
                                          context, error_msg)
            if (len_trim(error_msg) > 0) return

            if (.not. set_liric_block(context%session, arm_block, &
                                      error_msg)) return
            context%current_block_id = arm_block
            context%current_block_terminated = .false.
            call restore_symbols(context, baseline, baseline_count)
            terminated = .false.
            if (allocated(arm%body_indices)) then
                call lower_statement_list(arena, arm%body_indices, context, &
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
        class default
            error_msg = 'select case arm node has unexpected kind'
            return
        end select
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

        captured = .false.
        if (.not. node_exists(arena, default_idx)) then
            error_msg = 'select case default index does not reference an AST node'
            return
        end if

        if (.not. set_liric_block(context%session, default_block, &
                                  error_msg)) return
        context%current_block_id = default_block
        context%current_block_terminated = .false.
        call restore_symbols(context, baseline, baseline_count)

        select type (def => arena%entries(default_idx)%node)
        type is (case_default_node)
            terminated = .false.
            if (allocated(def%body_indices)) then
                call lower_statement_list(arena, def%body_indices, context, &
                                          dummy_value, terminated, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            call capture_select_result(context, terminated, result)
            if (.not. terminated) then
                if (.not. emit_liric_br(context%session, merge_block, error_msg)) &
                    return
                captured = .true.
            end if
        class default
            error_msg = 'select case default node has unexpected kind'
            return
        end select
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

        if (node_exists(arena, value_index)) then
            select type (label => arena%entries(value_index)%node)
            type is (case_range_node)
                ! lo: -> selector >= lo ; :hi -> selector <= hi ;
                ! lo:hi -> selector >= lo AND selector <= hi (chained checks)
                if (label%start_value > 0 .and. label%end_value > 0) then
                    hi_check_block = create_liric_block(context%session)
                    call emit_case_bound(arena, label%start_value, selector, &
                                         LR_CMP_SGE, hi_check_block, &
                                         no_match_block, context, error_msg)
                    if (len_trim(error_msg) > 0) return
                    if (.not. set_liric_block(context%session, hi_check_block, &
                                              error_msg)) return
                    context%current_block_id = hi_check_block
                    context%current_block_terminated = .false.
                    call emit_case_bound(arena, label%end_value, selector, &
                                         LR_CMP_SLE, arm_block, no_match_block, &
                                         context, error_msg)
                else if (label%start_value > 0) then
                    call emit_case_bound(arena, label%start_value, selector, &
                                         LR_CMP_SGE, arm_block, no_match_block, &
                                         context, error_msg)
                else if (label%end_value > 0) then
                    call emit_case_bound(arena, label%end_value, selector, &
                                         LR_CMP_SLE, arm_block, no_match_block, &
                                         context, error_msg)
                else
                    error_msg = 'select case range label has no bounds'
                end if
                return
            end select
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
