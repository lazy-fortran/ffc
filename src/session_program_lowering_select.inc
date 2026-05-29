    subroutine lower_select_case(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: selector
        integer(c_int32_t) :: default_block
        integer :: case_idx

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

        do case_idx = 1, size(node%case_indices)
            call lower_select_case_arm(arena, node%case_indices(case_idx), &
                                       case_idx == size(node%case_indices), &
                                       selector, default_block, context, &
                                       error_msg)
            if (len_trim(error_msg) > 0) return
        end do

        call lower_select_case_default(arena, node%default_index, default_block, &
                                       context, error_msg)
    end subroutine lower_select_case

    logical function is_character_selector(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(in) :: context
        integer :: symbol_index

        is_character_selector = .false.
        if (.not. node_exists(arena, node_index)) return
        select type (n => arena%entries(node_index)%node)
        type is (identifier_node)
            symbol_index = find_symbol(context, n%name)
            if (symbol_index > 0) is_character_selector = &
                context%symbols(symbol_index)%value_kind == VALUE_CHARACTER
        type is (literal_node)
            is_character_selector = is_character_literal(n)
        end select
    end function is_character_selector

    subroutine lower_select_case_character(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: sel_data, sel_len
        integer(c_int32_t) :: default_block
        integer :: case_idx

        call char_expr_operands(arena, node%selector_index, context, sel_data, &
                                sel_len, error_msg)
        if (len_trim(error_msg) > 0) return

        default_block = create_liric_block(context%session)
        do case_idx = 1, size(node%case_indices)
            call lower_select_case_character_arm(arena, &
                node%case_indices(case_idx), &
                case_idx == size(node%case_indices), sel_data, sel_len, &
                default_block, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call lower_select_case_default(arena, node%default_index, default_block, &
                                       context, error_msg)
    end subroutine lower_select_case_character

    subroutine lower_select_case_character_arm(arena, case_node_index, &
            is_last_arm, sel_data, sel_len, default_block, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: case_node_index
        logical, intent(in) :: is_last_arm
        type(lr_operand_desc_t), intent(in) :: sel_data, sel_len
        integer(c_int32_t), intent(in) :: default_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: arm_block, next_check_block, false_target
        type(lr_operand_desc_t) :: dummy_value
        type(lr_operand_desc_t) :: lbl_data, lbl_len, eq, cond
        logical :: terminated
        integer :: i

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
            terminated = .false.
            if (allocated(arm%body_indices)) then
                call lower_statement_list(arena, arm%body_indices, context, &
                                          dummy_value, terminated, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (.not. terminated) then
                call unsupported_feature_error('select case statement', &
                    arm%line, arm%column, 'case arm body must terminate '// &
                    '(stop/return)', error_msg)
                return
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
                                     selector, default_block, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: case_node_index
        logical, intent(in) :: is_last_arm
        type(lr_operand_desc_t), intent(in) :: selector
        integer(c_int32_t), intent(in) :: default_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dummy_value
        integer(c_int32_t) :: arm_block
        integer(c_int32_t) :: next_check_block
        logical :: terminated

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
            if (allocated(arm%body_indices)) then
                call lower_statement_list(arena, arm%body_indices, context, &
                                          dummy_value, terminated, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                terminated = .false.
            end if
            if (.not. terminated) then
                call unsupported_feature_error('select case statement', &
                                               arm%line, arm%column, &
                                               'case arm body must terminate '// &
                                               '(stop/return)', error_msg)
                return
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
                                         context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: default_idx
        integer(c_int32_t), intent(in) :: default_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dummy_value
        logical :: terminated

        if (.not. node_exists(arena, default_idx)) then
            error_msg = 'select case default index does not reference an AST node'
            return
        end if

        if (.not. set_liric_block(context%session, default_block, &
                                  error_msg)) return
        context%current_block_id = default_block
        context%current_block_terminated = .false.

        select type (def => arena%entries(default_idx)%node)
        type is (case_default_node)
            if (allocated(def%body_indices)) then
                call lower_statement_list(arena, def%body_indices, context, &
                                          dummy_value, terminated, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                terminated = .false.
            end if
            if (.not. terminated) then
                call unsupported_feature_error('select case statement', &
                                               def%line, def%column, &
                                               'case default body must terminate '// &
                                               '(stop/return)', error_msg)
                return
            end if
        class default
            error_msg = 'select case default node has unexpected kind'
            return
        end select
        context%current_block_terminated = .true.
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
