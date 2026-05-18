    subroutine lower_select_case(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(select_case_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: selector
        type(lr_operand_desc_t) :: literal_value
        type(lr_operand_desc_t) :: cond
        integer(c_int32_t) :: arm_block
        integer(c_int32_t) :: default_block
        integer :: case_idx
        type(lr_operand_desc_t) :: dummy_value
        logical :: terminated

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
        if (size(node%case_indices) /= 1) then
            call unsupported_feature_error('select case statement', node%line, &
                                           node%column, &
                                           'multi-arm select case not yet supported', &
                                           error_msg)
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

        case_idx = node%case_indices(1)
        if (.not. arena%has_node_at(case_idx)) then
            error_msg = 'select case arm index does not reference an AST node'
            return
        end if

        select type (arm => arena%entries(case_idx)%node)
        type is (case_block_node)
            if (.not. allocated(arm%value_indices)) then
                call unsupported_feature_error('select case statement', &
                                               arm%line, arm%column, &
                                               'case arm requires a label', &
                                               error_msg)
                return
            end if
            if (size(arm%value_indices) /= 1) then
                call unsupported_feature_error('select case statement', &
                                               arm%line, arm%column, &
                                               'multi-label case arms not yet '// &
                                               'supported', error_msg)
                return
            end if
            call lower_i32_expression(arena, arm%value_indices(1), context, &
                                      literal_value, error_msg)
            if (len_trim(error_msg) > 0) return

            arm_block = create_liric_block(context%session)
            default_block = create_liric_block(context%session)

            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_EQ, selector, &
                                          literal_value, cond, error_msg)) return
            if (.not. emit_liric_condbr(context%session, cond, arm_block, &
                                        default_block, error_msg)) return

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
        class default
            error_msg = 'select case arm node has unexpected kind'
            return
        end select

        call lower_select_case_default(arena, node%default_index, default_block, &
                                       context, error_msg)
    end subroutine lower_select_case

    subroutine lower_select_case_default(arena, default_idx, default_block, &
                                         context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: default_idx
        integer(c_int32_t), intent(in) :: default_block
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: dummy_value
        logical :: terminated

        if (.not. arena%has_node_at(default_idx)) then
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
