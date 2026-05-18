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
        type(lr_operand_desc_t) :: literal_value
        type(lr_operand_desc_t) :: cond
        integer(c_int32_t) :: false_target
        integer :: i

        do i = 1, size(value_indices)
            call lower_i32_expression(arena, value_indices(i), context, &
                                      literal_value, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, selector, &
                                          literal_value, cond, error_msg)) return
            if (i == size(value_indices)) then
                false_target = no_match_block
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
        call set_empty(error_msg)
    end subroutine emit_case_label_dispatch
