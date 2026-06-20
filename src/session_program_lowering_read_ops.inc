    ! File-unit READ and REWIND lowering (#247 file I/O).
    !
    ! A READ from an opened unit loads the unit's FILE* and emits one fscanf per
    ! list-directed scalar target, mirroring the stdin scanf path in write_ops.
    ! REWIND maps to the C runtime rewind(FILE*).

    logical function is_file_unit_read(node, context)
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer :: sym, ios, unit_number

        is_file_unit_read = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*' .or. &
            trim(node%unit_spec) == '5') return
        read (node%unit_spec, *, iostat=ios) unit_number
        if (ios == 0) then
            ! A numeric unit can be implicitly preconnected on first use.
            is_file_unit_read = .true.
            return
        end if
        sym = find_symbol(context, trim(node%unit_spec))
        if (sym > 0) is_file_unit_read = context%symbols(sym)%is_file_unit .or. &
                                         context%symbols(sym)%has_unit_const
    end function is_file_unit_read

    function file_unit_pseudo_name(unit_number) result(name)
        integer, intent(in) :: unit_number
        character(len=24) :: name
        write (name, '(A,I0)') '.unit.', unit_number
    end function file_unit_pseudo_name

    subroutine lower_read_file(arena, node, context, error_msg)
        ! List-directed READ from an opened file unit into scalar targets.
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp
        integer :: i

        call set_empty(error_msg)
        if (allocated(node%format_spec)) then
            if (trim(node%format_spec) /= '*') then
                call unsupported_feature_error('read from file unit', node%line, &
                    node%column, 'only list-directed (*) file-unit read is '// &
                    'supported (#247)', error_msg)
                return
            end if
        end if
        call load_unit_file_ptr(node%unit_spec, context, fp, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = ''
            call unsupported_feature_error('read from file unit', node%line, &
                node%column, 'unit not opened in same scope (#247)', error_msg)
            return
        end if
        if (.not. allocated(node%var_indices)) return
        do i = 1, size(node%var_indices)
            call lower_read_file_item(arena, node%var_indices(i), fp, node%line, &
                                      node%column, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_read_file

    subroutine lower_read_file_item(arena, node_index, fp, stmt_line, stmt_col, &
                                    context, error_msg)
        ! Emit fscanf(fp, fmt, &slot) for one scalar target, then reload it.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i64_alloca, &
                                                  emit_i32_load, emit_i64_load
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lr_operand_desc_t), intent(in) :: fp
        integer, intent(in) :: stmt_line, stmt_col
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name, scanf_fmt
        integer :: sym
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, slot, args(3)
        character(len=64) :: fmt_name

        call set_empty(error_msg)
        call identifier_name(arena, node_index, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, var_name)
        if (sym <= 0) then
            error_msg = 'read target was not declared: '//trim(var_name)
            return
        end if

        call file_read_slot_for_kind(context, sym, scanf_fmt, slot, error_msg)
        if (len_trim(error_msg) > 0) return
        if (len_trim(scanf_fmt) == 0) then
            call unsupported_feature_error('read from file unit', stmt_line, &
                stmt_col, 'list-directed file-unit read supports integer and '// &
                'real scalars only (#247)', error_msg)
            return
        end if

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.fscanf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = fp
        args(2) = fmt_ptr
        args(3) = slot
        if (.not. emit_fscanf(context%session, args, error_msg)) return

        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32)
            if (.not. emit_i32_load(context%session, slot, &
                                    context%symbols(sym)%value, error_msg)) return
        case (VALUE_I64)
            if (.not. emit_i64_load(context%session, slot, &
                                    context%symbols(sym)%value, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, slot, &
                                          context%symbols(sym)%value, &
                                          error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, slot, &
                                          context%symbols(sym)%value, &
                                          error_msg)) return
        end select
    end subroutine lower_read_file_item

    subroutine file_read_slot_for_kind(context, sym, scanf_fmt, slot, error_msg)
        ! Resolve the scanf conversion and ensure the target has an alloca slot.
        ! scanf_fmt is empty for unsupported kinds.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i64_alloca
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        character(len=:), allocatable, intent(out) :: scanf_fmt
        type(lr_operand_desc_t), intent(out) :: slot
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        scanf_fmt = ''
        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32)
            scanf_fmt = ' %d'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i32_alloca(context%session, slot, error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_I64)
            scanf_fmt = ' %ld'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i64_alloca(context%session, slot, error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_F32)
            scanf_fmt = ' %f'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_liric_f32_alloca(context%session, slot, &
                                                error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_F64)
            scanf_fmt = ' %lf'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_liric_f64_alloca(context%session, slot, &
                                                error_msg)) return
                call set_read_slot(context, sym, slot)
            else
                slot = context%symbols(sym)%address
            end if
        end select
    end subroutine file_read_slot_for_kind

    subroutine set_read_slot(context, sym, slot)
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        type(lr_operand_desc_t), intent(in) :: slot
        context%symbols(sym)%address = slot
        context%symbols(sym)%has_address = .true.
        context%symbols(sym)%is_reference = .true.
    end subroutine set_read_slot

    subroutine lower_rewind(node, context, error_msg)
        type(rewind_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp, fa(1)

        call set_empty(error_msg)
        if (.not. allocated(node%unit_spec)) then
            call unsupported_feature_error('rewind statement', node%line, &
                node%column, 'missing unit specifier (#247)', error_msg)
            return
        end if
        call load_unit_file_ptr(node%unit_spec, context, fp, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = ''
            call unsupported_feature_error('rewind statement', node%line, &
                node%column, 'unit not opened in same scope (#247)', error_msg)
            return
        end if
        fa(1) = fp
        if (.not. emit_void_call(context%session, 'rewind', fa, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_rewind
