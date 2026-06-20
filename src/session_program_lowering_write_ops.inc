    logical function is_stdout_write(node)
        ! True when write's unit is stdout: '*', '6', or the iso_fortran_env
        ! named constant output_unit (which is the standard-output unit).
        type(write_statement_node), intent(in) :: node

        is_stdout_write = .false.
        if (.not. allocated(node%unit_spec)) return
        is_stdout_write = trim(node%unit_spec) == '*' .or. &
                          trim(node%unit_spec) == '6' .or. &
                          unit_spec_is_output_unit(node%unit_spec)
    end function is_stdout_write

    logical function unit_spec_is_output_unit(unit_spec)
        ! True when the unit spec names iso_fortran_env's output_unit constant.
        ! Fortran identifiers are case-insensitive, so fold to lowercase first.
        character(len=*), intent(in) :: unit_spec
        character(len=:), allocatable :: folded
        integer :: i, c

        folded = trim(unit_spec)
        do i = 1, len(folded)
            c = iachar(folded(i:i))
            if (c >= iachar('A') .and. c <= iachar('Z')) &
                folded(i:i) = achar(c + (iachar('a') - iachar('A')))
        end do
        unit_spec_is_output_unit = folded == 'output_unit'
    end function unit_spec_is_output_unit

    logical function is_stdin_read(node)
        ! True when read's unit is stdin: '*' or '5'.
        type(read_statement_node), intent(in) :: node

        is_stdin_read = .false.
        if (.not. allocated(node%unit_spec)) return
        is_stdin_read = trim(node%unit_spec) == '*' .or. &
                        trim(node%unit_spec) == '5'
    end function is_stdin_read

    subroutine lower_write_stdout(arena, node, context, error_msg)
        ! Lower write(*, fmt) list to stdout by building a synthetic
        ! print_statement_node and calling lower_print.
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(print_statement_node) :: pnode

        pnode%line = node%line
        pnode%column = node%column
        if (allocated(node%format_spec)) pnode%format_spec = node%format_spec
        if (allocated(node%arg_indices)) then
            allocate (pnode%expression_indices(size(node%arg_indices)))
            pnode%expression_indices = node%arg_indices
        end if
        call lower_print(arena, pnode, context, error_msg)
    end subroutine lower_write_stdout

    subroutine lower_read_stdin(arena, node, context, error_msg)
        ! Lower read(*, *) var_list from stdin using scanf. Only list-directed
        ! input (* format) to scalar integer or real variables is supported.
        type(ast_arena_t), intent(in) :: arena
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%var_indices)) return
        if (allocated(node%format_spec)) then
            if (trim(node%format_spec) /= '*') then
                call unsupported_feature_error('read statement', node%line, &
                    node%column, 'only list-directed (*) stdin read is '// &
                    'supported (#247 slice B5b)', error_msg)
                return
            end if
        end if
        do i = 1, size(node%var_indices)
            call lower_read_stdin_item(arena, node%var_indices(i), node%line, &
                                       node%column, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end subroutine lower_read_stdin

    subroutine lower_read_stdin_item(arena, node_index, stmt_line, stmt_col, &
                                     context, error_msg)
        ! Emit a scanf call for a single list-directed stdin read target.
        ! Supports: integer(4), integer(8), real(4), real(8).
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer, intent(in) :: stmt_line, stmt_col
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name
        integer :: sym
        integer(c_int32_t) :: fmt_id
        type(lr_operand_desc_t) :: fmt_ptr, slot, args(2)
        character(len=64) :: fmt_name
        character(len=:), allocatable :: scanf_fmt

        call set_empty(error_msg)
        call identifier_name(arena, node_index, var_name, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, var_name)
        if (sym <= 0) then
            error_msg = 'read target was not declared: '//trim(var_name)
            return
        end if

        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32)
            scanf_fmt = ' %d'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i32_alloca(context%session, slot, error_msg)) return
                context%symbols(sym)%address = slot
                context%symbols(sym)%has_address = .true.
                context%symbols(sym)%is_reference = .true.
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_I64)
            scanf_fmt = ' %ld'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i64_alloca(context%session, slot, error_msg)) return
                context%symbols(sym)%address = slot
                context%symbols(sym)%has_address = .true.
                context%symbols(sym)%is_reference = .true.
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_F32)
            scanf_fmt = ' %f'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_liric_f32_alloca(context%session, slot, &
                                                error_msg)) return
                context%symbols(sym)%address = slot
                context%symbols(sym)%has_address = .true.
                context%symbols(sym)%is_reference = .true.
            else
                slot = context%symbols(sym)%address
            end if
        case (VALUE_F64)
            scanf_fmt = ' %lf'
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_liric_f64_alloca(context%session, slot, &
                                                error_msg)) return
                context%symbols(sym)%address = slot
                context%symbols(sym)%has_address = .true.
                context%symbols(sym)%is_reference = .true.
            else
                slot = context%symbols(sym)%address
            end if
        case default
            call unsupported_feature_error('read statement', stmt_line, stmt_col, &
                'list-directed stdin read supports integer and real scalars only '// &
                '(#247 slice B5b)', error_msg)
            return
        end select

        context%string_literal_count = context%string_literal_count + 1
        write (fmt_name, '(A,I0)') '.ffc.scanf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fmt_name), &
                                         scanf_fmt, fmt_id, error_msg)
        if (len_trim(error_msg) > 0) return
        fmt_ptr = printf_format_ptr(context%session, fmt_id)

        args(1) = fmt_ptr
        args(2) = slot
        if (.not. emit_scanf(context%session, args, error_msg)) return

        ! Reload the value from its alloca slot so the symbol's SSA value is
        ! current for subsequent uses.
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
    end subroutine lower_read_stdin_item
