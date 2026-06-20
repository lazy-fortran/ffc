    ! NAMELIST I/O lowering (#247 namelist).
    !
    ! A `namelist /group/ a, b, c` declaration records the group's ordered
    ! members. A `write(unit, nml=group)` then emits the gfortran-style group
    ! banner, one ` NAME= value,` line per member, and a closing ` /` line,
    ! reusing the scalar value formatting of list-directed and file output.
    !
    ! Scope: scalar integer, real, and logical members. Array, character, and
    ! derived members raise a clean diagnostic rather than mis-emitting.

    subroutine record_namelist_group(node, context, error_msg)
        type(namelist_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(namelist_group_t), allocatable :: grown(:)
        integer :: idx, i, n

        call set_empty(error_msg)
        if (.not. allocated(node%group_name)) return

        if (.not. allocated(context%namelist_groups)) then
            allocate (context%namelist_groups(4))
        else if (context%namelist_group_count >= size(context%namelist_groups)) then
            allocate (grown(size(context%namelist_groups) * 2))
            grown(1:context%namelist_group_count) = &
                context%namelist_groups(1:context%namelist_group_count)
            call move_alloc(grown, context%namelist_groups)
        end if

        context%namelist_group_count = context%namelist_group_count + 1
        idx = context%namelist_group_count
        context%namelist_groups(idx)%group_name = trim(node%group_name)
        context%namelist_groups(idx)%member_count = 0
        if (.not. allocated(node%variable_names)) return
        n = min(size(node%variable_names), MAX_NAMELIST_MEMBERS)
        do i = 1, n
            if (.not. allocated(node%variable_names(i)%s)) cycle
            context%namelist_groups(idx)%member_count = &
                context%namelist_groups(idx)%member_count + 1
            context%namelist_groups(idx)%member_names( &
                context%namelist_groups(idx)%member_count) = &
                trim(node%variable_names(i)%s)
        end do
    end subroutine record_namelist_group

    integer function find_namelist_group(context, name) result(idx)
        type(lowering_context_t), intent(in) :: context
        character(len=*), intent(in) :: name
        integer :: i

        idx = 0
        do i = 1, context%namelist_group_count
            if (trim(context%namelist_groups(i)%group_name) == trim(name)) then
                idx = i
                return
            end if
        end do
    end function find_namelist_group

    subroutine lower_namelist_write(node, context, error_msg)
        ! Emit the group banner, each member line, and the closing slash.
        ! Routes to stdout (printf) for unit '*'/'6', else fprintf to the file
        ! unit's FILE* so a file write does not leak onto compared stdout.
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp
        logical :: to_stdout, have_fp
        integer :: gidx, i

        call set_empty(error_msg)
        gidx = find_namelist_group(context, trim(node%namelist_group))
        if (gidx <= 0) then
            call unsupported_feature_error('namelist write', node%line, &
                node%column, 'namelist group not declared in this scope', &
                error_msg)
            return
        end if

        to_stdout = .true.
        have_fp = .false.
        if (allocated(node%unit_spec)) then
            to_stdout = trim(node%unit_spec) == '*' .or. &
                        trim(node%unit_spec) == '6'
            if (.not. to_stdout) then
                call load_unit_file_ptr(node%unit_spec, context, fp, error_msg)
                if (len_trim(error_msg) > 0) return
                have_fp = .true.
            end if
        end if

        call namelist_emit_text(context, fp, have_fp, &
            '&'//upcase(trim(context%namelist_groups(gidx)%group_name)), .true., &
            error_msg)
        if (len_trim(error_msg) > 0) return

        do i = 1, context%namelist_groups(gidx)%member_count
            call lower_namelist_member(context, fp, have_fp, &
                trim(context%namelist_groups(gidx)%member_names(i)), &
                node%line, node%column, error_msg)
            if (len_trim(error_msg) > 0) return
        end do

        call namelist_emit_text(context, fp, have_fp, ' /', .true., error_msg)
    end subroutine lower_namelist_write

    subroutine lower_namelist_member(context, fp, have_fp, name, line, col, &
                                     error_msg)
        ! Emit ` NAME= value,` for one scalar member, loading its value from
        ! the symbol table by kind.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        logical, intent(in) :: have_fp
        character(len=*), intent(in) :: name
        integer, intent(in) :: line, col
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: val, val_f64
        integer :: sym, vk

        call set_empty(error_msg)
        sym = find_symbol(context, name)
        if (sym <= 0) then
            call unsupported_feature_error('namelist member', line, col, &
                'namelist member not declared: '//name, error_msg)
            return
        end if
        if (context%symbols(sym)%is_array) then
            call unsupported_feature_error('namelist member', line, col, &
                'array namelist members are not supported yet', error_msg)
            return
        end if
        vk = context%symbols(sym)%value_kind

        call namelist_emit_text(context, fp, have_fp, &
            ' '//upcase(name)//'=', .false., error_msg)
        if (len_trim(error_msg) > 0) return

        call namelist_load_scalar(context, sym, vk, val, error_msg)
        if (len_trim(error_msg) > 0) return

        select case (vk)
        case (VALUE_LOGICAL)
            if (have_fp) then
                call namelist_fprintf_logical(context, fp, val, error_msg)
            else
                call lower_print_logical_value(context, val, error_msg)
            end if
        case (VALUE_I32)
            call namelist_emit_i32(context, fp, have_fp, val, error_msg)
        case (VALUE_I64)
            call namelist_emit_i64(context, fp, have_fp, val, error_msg)
        case (VALUE_F32)
            if (have_fp) then
                if (.not. emit_liric_f32_to_f64(context%session, val, val_f64, &
                                                error_msg)) return
                call namelist_emit_f64(context, fp, have_fp, val_f64, error_msg)
            else
                if (.not. emit_liric_print_f32_value(context%session, val, &
                                                     error_msg)) return
            end if
        case (VALUE_F64)
            if (have_fp) then
                call namelist_emit_f64(context, fp, have_fp, val, error_msg)
            else
                if (.not. emit_liric_print_f64_value(context%session, val, &
                                                     error_msg)) return
            end if
        case default
            call unsupported_feature_error('namelist member', line, col, &
                'only integer, real, and logical namelist members are '// &
                'supported yet', error_msg)
            return
        end select
        if (len_trim(error_msg) > 0) return

        call namelist_emit_text(context, fp, have_fp, ',', .true., error_msg)
    end subroutine lower_namelist_member

    subroutine namelist_load_scalar(context, sym, vk, val, error_msg)
        ! Materialise a scalar symbol's current value. A symbol kept in memory
        ! loads from its address; a value-only symbol uses its stored operand.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: sym
        integer, intent(in) :: vk
        type(lr_operand_desc_t), intent(out) :: val
        character(len=:), allocatable, intent(out) :: error_msg

        call set_empty(error_msg)
        if (.not. context%symbols(sym)%has_address) then
            val = context%symbols(sym)%value
            return
        end if
        select case (vk)
        case (VALUE_I32, VALUE_LOGICAL)
            if (.not. emit_i32_load(context%session, &
                context%symbols(sym)%address, val, error_msg)) return
        case (VALUE_I64)
            if (.not. emit_i64_load(context%session, &
                context%symbols(sym)%address, val, error_msg)) return
        case (VALUE_F32)
            if (.not. emit_liric_f32_load(context%session, &
                context%symbols(sym)%address, val, error_msg)) return
        case (VALUE_F64)
            if (.not. emit_liric_f64_load(context%session, &
                context%symbols(sym)%address, val, error_msg)) return
        case default
            val = context%symbols(sym)%value
        end select
    end subroutine namelist_load_scalar

    subroutine namelist_emit_text(context, fp, have_fp, text, newline, error_msg)
        ! Emit a literal string to stdout (printf) or the file unit (fprintf),
        ! optionally followed by a newline.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp
        logical, intent(in) :: have_fp
        character(len=*), intent(in) :: text
        logical, intent(in) :: newline
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fmtop, fa(2)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn, sname
        character(len=:), allocatable :: payload

        call set_empty(error_msg)
        if (have_fp) then
            payload = text
            if (newline) payload = payload//achar(10)
            context%string_literal_count = context%string_literal_count + 1
            write (fgn, '(A,I0)') '.ffc.nml.txt.', context%string_literal_count
            call create_printf_format_global(context%session, trim(fgn), &
                                              payload, fgid, error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fgid)
            fa(1) = fp
            fa(2) = fmtop
            if (.not. emit_fprintf(context%session, fa, error_msg)) return
        else
            context%string_literal_count = context%string_literal_count + 1
            write (sname, '(A,I0)') '.ffc.nml.s.', context%string_literal_count
            if (.not. emit_liric_print_string_value(context%session, &
                    context%str_print_format_id, trim(sname), text, &
                    error_msg)) return
            if (newline) then
                if (.not. emit_liric_print_newline(context%session, error_msg)) &
                    return
            end if
        end if
    end subroutine namelist_emit_text

    subroutine namelist_emit_i32(context, fp, have_fp, val, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp, val
        logical, intent(in) :: have_fp
        character(len=:), allocatable, intent(out) :: error_msg

        if (have_fp) then
            call namelist_fprintf_value(context, fp, '%d', val, error_msg)
        else
            if (.not. emit_liric_print_i32_value(context%session, &
                    context%i32_print_format_id, val, error_msg)) return
            call set_empty(error_msg)
        end if
    end subroutine namelist_emit_i32

    subroutine namelist_emit_i64(context, fp, have_fp, val, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp, val
        logical, intent(in) :: have_fp
        character(len=:), allocatable, intent(out) :: error_msg

        if (have_fp) then
            call namelist_fprintf_value(context, fp, '%ld', val, error_msg)
        else
            if (.not. emit_liric_print_i64_value(context%session, &
                    context%i64_print_format_id, val, error_msg)) return
            call set_empty(error_msg)
        end if
    end subroutine namelist_emit_i64

    subroutine namelist_emit_f64(context, fp, have_fp, val, error_msg)
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp, val
        logical, intent(in) :: have_fp
        character(len=:), allocatable, intent(out) :: error_msg

        if (have_fp) then
            call namelist_fprintf_value(context, fp, '%.8g', val, error_msg)
        else
            if (.not. emit_liric_print_f64_value(context%session, val, &
                                                 error_msg)) return
            call set_empty(error_msg)
        end if
    end subroutine namelist_emit_f64

    subroutine namelist_fprintf_logical(context, fp, val, error_msg)
        ! Emit "T"/"F" to a file unit by branching on the i32 logical value.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp, val
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: cond, zero
        integer(c_int32_t) :: true_block, false_block, merge_block

        zero = i32_immediate(context%session, 0_c_int64_t)
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, val, zero, &
                                      cond, error_msg)) return
        true_block = create_liric_block(context%session)
        false_block = create_liric_block(context%session)
        merge_block = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, cond, true_block, &
                                    false_block, error_msg)) return

        if (.not. set_liric_block(context%session, true_block, error_msg)) return
        context%current_block_id = true_block
        call namelist_emit_text(context, fp, .true., 'T', .false., error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, false_block, error_msg)) return
        context%current_block_id = false_block
        call namelist_emit_text(context, fp, .true., 'F', .false., error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_liric_br(context%session, merge_block, error_msg)) return

        if (.not. set_liric_block(context%session, merge_block, error_msg)) return
        context%current_block_id = merge_block
        context%current_block_terminated = .false.
        call set_empty(error_msg)
    end subroutine namelist_fprintf_logical

    subroutine namelist_fprintf_value(context, fp, conv, val, error_msg)
        ! fprintf(fp, conv, val) for a single scalar value to a file unit.
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: fp, val
        character(len=*), intent(in) :: conv
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fmtop, fa(3)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn

        call set_empty(error_msg)
        context%string_literal_count = context%string_literal_count + 1
        write (fgn, '(A,I0)') '.ffc.nml.v.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fgn), conv, &
                                          fgid, error_msg)
        if (len_trim(error_msg) > 0) return
        fmtop = printf_format_ptr(context%session, fgid)
        fa(1) = fp
        fa(2) = fmtop
        fa(3) = val
        if (.not. emit_fprintf(context%session, fa, error_msg)) return
    end subroutine namelist_fprintf_value

    subroutine lower_namelist_read(node, context, error_msg)
        ! NAMELIST input parsing is not implemented. Emit a clean diagnostic so
        ! the construct is rejected rather than mis-compiled.
        type(read_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg

        call unsupported_feature_error('namelist read', node%line, node%column, &
            'direct LIRIC session does not support namelist input yet', &
            error_msg)
    end subroutine lower_namelist_read

    pure function upcase(text) result(out)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: out
        integer :: i, c

        do i = 1, len(text)
            c = iachar(text(i:i))
            if (c >= iachar('a') .and. c <= iachar('z')) then
                out(i:i) = achar(c - 32)
            else
                out(i:i) = text(i:i)
            end if
        end do
    end function upcase
