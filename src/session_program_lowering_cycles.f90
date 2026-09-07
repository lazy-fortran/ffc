submodule (session_program_lowering_impl) session_program_lowering_cycles
    implicit none
contains
    module subroutine begin_loop_cycle_tracking(context, saved)
        type(lowering_context_t), intent(inout) :: context
        type(loop_cycle_state_t), intent(out) :: saved

        call move_alloc(context%loop_cycles%blocks, saved%blocks)
        call move_alloc(context%loop_cycles%values, saved%values)
        allocate (context%loop_cycles%blocks(0))
        allocate (context%loop_cycles%values(context%symbol_count, 0))
    end subroutine begin_loop_cycle_tracking

    module subroutine end_loop_cycle_tracking(context, saved)
        type(lowering_context_t), intent(inout) :: context
        type(loop_cycle_state_t), intent(inout) :: saved

        call move_alloc(saved%blocks, context%loop_cycles%blocks)
        call move_alloc(saved%values, context%loop_cycles%values)
    end subroutine end_loop_cycle_tracking

    module subroutine record_loop_cycle(context, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        integer(c_int32_t), allocatable :: blocks(:)
        type(lr_operand_desc_t), allocatable :: values(:,:)
        integer :: count, captured_count, i

        call set_empty(error_msg)
        if (.not. allocated(context%loop_cycles%blocks)) then
            error_msg = 'direct LIRIC session CYCLE has no active loop edge table'
            return
        end if
        count = size(context%loop_cycles%blocks)
        captured_count = size(context%loop_cycles%values, 1)
        if (context%symbol_count < captured_count) then
            error_msg = 'direct LIRIC session CYCLE lost loop symbols'
            return
        end if
        allocate (blocks(count + 1), values(captured_count, count + 1))
        blocks(:count) = context%loop_cycles%blocks
        values(:, :count) = context%loop_cycles%values
        blocks(count + 1) = context%current_block_id
        do i = 1, captured_count
            values(i, count + 1) = context%symbols(i)%value
        end do
        call move_alloc(blocks, context%loop_cycles%blocks)
        call move_alloc(values, context%loop_cycles%values)
    end subroutine record_loop_cycle

    module subroutine merge_loop_cycle_values(context, error_msg)
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        integer :: count, i
        type(lr_operand_desc_t), allocatable :: values(:)

        call set_empty(error_msg)
        count = size(context%loop_cycles%blocks)
        if (count == 0) return
        allocate (values(count))
        ! Every CYCLE and the ordinary fallthrough enter this latch. Values
        ! defined only on the fallthrough cannot be read on a CYCLE edge.
        do i = 1, size(context%loop_cycles%values, 1)
            if (.not. is_carried_kind(context%symbols(i)%value_kind)) cycle
            if (context%symbols(i)%is_array) cycle
            if (context%symbols(i)%is_allocatable) cycle
            if (count == 1) then
                context%symbols(i)%value = context%loop_cycles%values(i, 1)
            else
                values = context%loop_cycles%values(i, :)
                if (.not. emit_liric_phi_n(context%session, &
                    values, context%loop_cycles%blocks, &
                    context%symbols(i)%value, error_msg)) return
            end if
        end do
    end subroutine merge_loop_cycle_values
end submodule session_program_lowering_cycles
