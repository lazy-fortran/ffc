submodule (session_program_lowering_impl) session_program_lowering_loop_targets
    implicit none
contains
    module subroutine begin_loop_branch_target(context, target, exit_block, &
            latch_block, construct_name)
        type(lowering_context_t), intent(inout) :: context
        type(loop_branch_target_t), target, intent(out) :: target
        integer(c_int32_t), intent(in) :: exit_block, latch_block
        character(len=*), intent(in), optional :: construct_name

        target%parent => context%loop_target
        target%exit_block = exit_block
        target%latch_block = latch_block
        if (present(construct_name)) target%name = lowercase_text(construct_name)
        allocate (target%exits%blocks(0), target%cycles%blocks(0))
        allocate (target%exits%values(context%symbol_count, 0))
        allocate (target%cycles%values(context%symbol_count, 0))
        context%loop_target => target
    end subroutine begin_loop_branch_target

    module subroutine end_loop_branch_target(context, target, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(loop_branch_target_t), intent(in) :: target
        character(len=:), allocatable, intent(inout) :: error_msg

        context%loop_target => target%parent
        if (len_trim(error_msg) > 0) return
        call append_target_edges(context%loop_cycles%blocks, &
            context%loop_cycles%values, target%cycles, error_msg)
        if (len_trim(error_msg) > 0) return
        call append_target_edges(context%loop_exit_blocks, &
            context%loop_exit_values, target%exits, error_msg)
        context%loop_exit_count = size(context%loop_exit_blocks)
    end subroutine end_loop_branch_target

    module subroutine lower_loop_branch(context, is_cycle, label, error_msg)
        type(lowering_context_t), intent(inout) :: context
        logical, intent(in) :: is_cycle
        character(len=:), allocatable, intent(in) :: label
        character(len=:), allocatable, intent(out) :: error_msg
        type(loop_branch_target_t), pointer :: target
        integer(c_int32_t) :: destination
        character(len=:), allocatable :: name

        call set_empty(error_msg)
        target => context%loop_target
        if (allocated(label)) then
            name = lowercase_text(trim(label))
            if (len(name) > 0) then
                do while (associated(target))
                    if (allocated(target%name)) then
                        if (target%name == name) exit
                    end if
                    target => target%parent
                end do
            end if
        end if
        if (.not. associated(target)) then
            error_msg = 'EXIT/CYCLE construct name does not identify an active DO'
            return
        end if
        if (is_cycle) then
            destination = target%latch_block
            if (associated(target, context%loop_target)) then
                call record_loop_cycle(context, error_msg)
            else
                call capture_target_edge(context, target%cycles, error_msg)
            end if
        else
            destination = target%exit_block
            if (associated(target, context%loop_target)) then
                call record_loop_exit(context, error_msg)
            else
                call capture_target_edge(context, target%exits, error_msg)
            end if
        end if
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, destination, error_msg)) return
    end subroutine lower_loop_branch

    subroutine capture_target_edge(context, edges, error_msg)
        type(lowering_context_t), intent(in) :: context
        type(loop_cycle_state_t), intent(inout) :: edges
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t), allocatable :: blocks(:)
        type(lr_operand_desc_t), allocatable :: values(:,:)
        integer :: count, captured_count, i

        call set_empty(error_msg)
        count = size(edges%blocks)
        captured_count = size(edges%values, 1)
        if (context%symbol_count < captured_count) then
            error_msg = 'direct LIRIC session named branch lost loop symbols'
            return
        end if
        allocate (blocks(count + 1), values(captured_count, count + 1))
        blocks(:count) = edges%blocks
        values(:, :count) = edges%values
        blocks(count + 1) = context%current_block_id
        do i = 1, captured_count
            values(i, count + 1) = context%symbols(i)%value
        end do
        call move_alloc(blocks, edges%blocks)
        call move_alloc(values, edges%values)
    end subroutine capture_target_edge

    subroutine append_target_edges(blocks, values, edges, error_msg)
        integer(c_int32_t), allocatable, intent(inout) :: blocks(:)
        type(lr_operand_desc_t), allocatable, intent(inout) :: values(:,:)
        type(loop_cycle_state_t), intent(in) :: edges
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t), allocatable :: merged_blocks(:)
        type(lr_operand_desc_t), allocatable :: merged_values(:,:)
        integer :: count, added

        call set_empty(error_msg)
        added = size(edges%blocks)
        if (added == 0) return
        if (size(values, 1) /= size(edges%values, 1)) then
            error_msg = 'direct LIRIC session named branch cannot merge loop symbols'
            return
        end if
        count = size(blocks)
        allocate (merged_blocks(count + added))
        allocate (merged_values(size(values, 1), count + added))
        merged_blocks(:count) = blocks
        merged_blocks(count + 1:) = edges%blocks
        merged_values(:, :count) = values
        merged_values(:, count + 1:) = edges%values
        call move_alloc(merged_blocks, blocks)
        call move_alloc(merged_values, values)
    end subroutine append_target_edges
end submodule session_program_lowering_loop_targets
