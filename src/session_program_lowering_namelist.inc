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
        sym = find_symbol_compat(context, name)
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
            fgn = ffc_unit_global_name( &
                context, 'nml.txt.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(fgn), &
                                              payload, fgid, error_msg)
            if (len_trim(error_msg) > 0) return
            fmtop = printf_format_ptr(context%session, fgid)
            fa(1) = fp
            fa(2) = fmtop
            if (.not. emit_fprintf(context%session, fa, error_msg)) return
        else
            context%string_literal_count = context%string_literal_count + 1
            sname = ffc_unit_global_name( &
                context, 'nml.s.', context%string_literal_count)
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
        fgn = ffc_unit_global_name( &
            context, 'nml.v.', context%string_literal_count)
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


    ! NAMELIST member constraints (#389).
    !
    ! A namelist group object must be a variable that is declared before the
    ! NAMELIST statement, must not have the PROCEDURE attribute, and must not
    ! be PRIVATE while the group itself is public; a group object of a derived
    ! type with use-associated PRIVATE components may not appear at all. The
    ! group list itself must be syntactically complete. The transfer-statement
    ! constraints (polymorphic or ALLOCATABLE/POINTER-component object without
    ! a defined input/output procedure, INTENT(IN) object read) live in
    ! session_program_lowering_io_typecheck.inc.
    !
    ! The group lists are read from the source text: FortFront drops a NAMELIST
    ! statement that appears inside a procedure, so the typed nodes alone would
    ! leave those scopes unchecked. Every member is then resolved against the
    ! typed declaration, derived type, and procedure nodes of the arena.
    subroutine check_namelist_members(arena, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: source, stmt
        integer :: pos, line_no, stmt_line
        logical :: found

        call set_empty(error_msg)
        call get_source_text(arena, source, found)
        if (.not. found) return
        pos = 1
        line_no = 1
        do
            call namelist_next_statement(source, pos, line_no, stmt, stmt_line, &
                                         found)
            if (.not. found) exit
            call check_namelist_statement(arena, source, stmt, stmt_line, &
                                          error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call check_namelist_transfers(arena, source, error_msg)
    end subroutine check_namelist_members

    ! Next NAMELIST statement in the source, with continuation lines joined.
    ! `pos` and `line_no` carry the scan state; `found` is false at end of
    ! source.
    subroutine namelist_next_statement(source, pos, line_no, stmt, stmt_line, &
                                       found)
        character(len=*), intent(in) :: source
        integer, intent(inout) :: pos
        integer, intent(inout) :: line_no
        character(len=:), allocatable, intent(out) :: stmt
        integer, intent(out) :: stmt_line
        logical, intent(out) :: found
        character(len=:), allocatable :: line, text
        integer :: next_nl, n

        found = .false.
        stmt = ''
        stmt_line = 0
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                line = source(pos:)
                pos = len(source) + 1
            else
                line = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            stmt_line = line_no
            line_no = line_no + 1
            text = trim(adjustl(strip_data_source_comment(line)))
            if (.not. is_namelist_statement_text(text)) cycle
            found = .true.
            stmt = text
            ! Join continuation lines so the whole group list is one string.
            do while (len_trim(stmt) > 0)
                n = len_trim(stmt)
                if (stmt(n:n) /= '&') exit
                stmt = stmt(1:n - 1)
                if (pos > len(source)) exit
                next_nl = index(source(pos:), new_line('a'))
                if (next_nl == 0) then
                    line = source(pos:)
                    pos = len(source) + 1
                else
                    line = source(pos:pos + next_nl - 2)
                    pos = pos + next_nl
                end if
                line_no = line_no + 1
                text = trim(adjustl(strip_data_source_comment(line)))
                if (len(text) > 0) then
                    if (text(1:1) == '&') text = text(2:)
                end if
                stmt = trim(stmt)//' '//trim(text)
            end do
            return
        end do
    end subroutine namelist_next_statement

    logical function is_namelist_statement_text(text) result(is_nml)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: low

        is_nml = .false.
        low = trim(lowercase_text(text))
        if (len(low) < 9) return
        if (low(1:8) /= 'namelist') return
        is_nml = low(9:9) == ' ' .or. low(9:9) == '/'
    end function is_namelist_statement_text

    ! Walk `namelist /group/ a, b [[,] /group2/ c]` and validate every member.
    subroutine check_namelist_statement(arena, source, stmt, stmt_line, error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stmt
        integer, intent(in) :: stmt_line
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: group, member
        integer :: p, n
        logical :: new_group, ok

        call set_empty(error_msg)
        n = len_trim(stmt)
        p = 9
        group = ''
        new_group = .true.
        do
            if (new_group) then
                call namelist_take_group(stmt, n, p, group, ok)
                if (.not. ok) then
                    error_msg = namelist_syntax_error(stmt_line)
                    return
                end if
                new_group = .false.
            end if
            call namelist_take_member(stmt, n, p, member, new_group, ok)
            if (.not. ok) then
                error_msg = namelist_syntax_error(stmt_line)
                return
            end if
            if (len_trim(member) > 0) then
                call check_namelist_member(arena, source, member, group, &
                                           stmt_line, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (p > n) exit
        end do
    end subroutine check_namelist_statement

    function namelist_syntax_error(stmt_line) result(msg)
        integer, intent(in) :: stmt_line
        character(len=:), allocatable :: msg
        character(len=32) :: location

        write (location, '(" at line ",I0)') stmt_line
        msg = 'Syntax error in NAMELIST statement'//trim(location)
    end function namelist_syntax_error

    ! Consume `/group/`, leaving p on the first member character.
    subroutine namelist_take_group(stmt, n, p, group, ok)
        character(len=*), intent(in) :: stmt
        integer, intent(in) :: n
        integer, intent(inout) :: p
        character(len=:), allocatable, intent(out) :: group
        logical, intent(out) :: ok
        integer :: close_pos

        ok = .false.
        group = ''
        do while (p <= n)
            if (stmt(p:p) /= ' ' .and. stmt(p:p) /= ',') exit
            p = p + 1
        end do
        if (p > n) return
        if (stmt(p:p) /= '/') return
        close_pos = index(stmt(p + 1:n), '/')
        if (close_pos <= 1) return
        group = trim(adjustl(stmt(p + 1:p + close_pos - 1)))
        p = p + close_pos + 1
        ok = len_trim(group) > 0
    end subroutine namelist_take_group

    ! Consume one member name. `new_group` reports that a `/group/` follows.
    ! An empty item, or a list that ends on a separator, is a syntax error.
    subroutine namelist_take_member(stmt, n, p, member, new_group, ok)
        character(len=*), intent(in) :: stmt
        integer, intent(in) :: n
        integer, intent(inout) :: p
        character(len=:), allocatable, intent(out) :: member
        logical, intent(out) :: new_group
        logical, intent(out) :: ok
        integer :: start

        ok = .false.
        new_group = .false.
        member = ''
        do while (p <= n)
            if (stmt(p:p) /= ' ') exit
            p = p + 1
        end do
        if (p > n) return
        if (stmt(p:p) == '/') then
            new_group = .true.
            ok = .true.
            return
        end if
        if (stmt(p:p) == ',') return
        start = p
        do while (p <= n)
            if (stmt(p:p) == ',' .or. stmt(p:p) == '/') exit
            p = p + 1
        end do
        member = trim(adjustl(stmt(start:p - 1)))
        if (len_trim(member) == 0) return
        ok = .true.
        if (p > n) return
        if (stmt(p:p) == '/') then
            new_group = .true.
            return
        end if
        ! Separator consumed; a trailing separator with nothing after it, or a
        ! second separator, is invalid.
        p = p + 1
        do while (p <= n)
            if (stmt(p:p) /= ' ') exit
            p = p + 1
        end do
        if (p > n) then
            ok = .false.
            return
        end if
        if (stmt(p:p) == ',') ok = .false.
        if (stmt(p:p) == '/') new_group = .true.
    end subroutine namelist_take_member

    subroutine check_namelist_member(arena, source, name, group, stmt_line, &
                                     error_msg)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: group
        integer, intent(in) :: stmt_line
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=32) :: location
        integer :: decl_index, later_index

        call set_empty(error_msg)
        write (location, '(" at line ",I0)') stmt_line
        decl_index = namelist_find_declaration(arena, name, stmt_line)
        if (decl_index == 0) then
            later_index = namelist_find_declaration(arena, name, huge(1))
            if (later_index /= 0) then
                error_msg = 'NAMELIST object '''//trim(name)//''' must be '// &
                    'declared before the namelist group '''//trim(group)// &
                    ''''//trim(location)
                return
            end if
            if (namelist_name_is_procedure(arena, name)) then
                error_msg = 'PROCEDURE attribute conflicts with NAMELIST '// &
                    'object '''//trim(name)//''' in group '''//trim(group)// &
                    ''''//trim(location)
            end if
            return
        end if
        if (namelist_declaration_is_private(arena, decl_index, name)) then
            error_msg = 'NAMELIST object '''//trim(name)//''' was declared '// &
                'PRIVATE and cannot be member of PUBLIC namelist '''// &
                trim(group)//''''//trim(location)
            return
        end if
        if (namelist_member_has_private_components(arena, source, decl_index, &
                                                   stmt_line)) then
            error_msg = 'NAMELIST object '''//trim(name)//''' has '// &
                'use-associated PRIVATE components and cannot be a member '// &
                'of namelist '''//trim(group)//''''//trim(location)
        end if
    end subroutine check_namelist_member

    ! Arena index of a declaration of `name` that appears at or before
    ! `max_line`, or 0 when the name has no such declaration in this file.
    integer function namelist_find_declaration(arena, name, max_line) &
            result(decl_index)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer, intent(in) :: max_line
        integer :: n

        decl_index = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (decl => arena%entries(n)%node)
            type is (declaration_node)
                if (decl%line > max_line) cycle
                if (namelist_decl_declares(decl, name)) then
                    decl_index = n
                    return
                end if
            end select
        end do
    end function namelist_find_declaration

    logical function namelist_decl_declares(decl, name) result(declares)
        type(declaration_node), intent(in) :: decl
        character(len=*), intent(in) :: name
        integer :: k

        declares = .false.
        if (allocated(decl%var_name)) then
            if (same_name(decl%var_name, name)) declares = .true.
        end if
        if (.not. decl%is_multi_declaration) return
        if (.not. allocated(decl%var_names)) return
        do k = 1, size(decl%var_names)
            if (same_name(decl%var_names(k), name)) declares = .true.
        end do
    end function namelist_decl_declares

    ! A namelist group object that is a function or subroutine name carries the
    ! PROCEDURE attribute, which conflicts with the namelist object attribute.
    logical function namelist_name_is_procedure(arena, name) result(is_proc)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: n

        is_proc = .false.
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (function_def_node)
                if (allocated(nd%name)) then
                    if (same_name(nd%name, name)) is_proc = .true.
                end if
            type is (subroutine_def_node)
                if (allocated(nd%name)) then
                    if (same_name(nd%name, name)) is_proc = .true.
                end if
            end select
            if (is_proc) return
        end do
    end function namelist_name_is_procedure

    ! A module entity is PRIVATE through the attribute form or through a
    ! visibility statement of its module. A NAMELIST group with no
    ! accessibility statement of its own is public, so a private member is
    ! invalid. Outside a module nothing is private.
    logical function namelist_declaration_is_private(arena, decl_index, name) &
            result(is_private)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: decl_index
        character(len=*), intent(in) :: name
        integer :: mod_index

        is_private = .false.
        mod_index = namelist_owner_module(arena, decl_index)
        if (mod_index == 0) return
        select type (decl => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (allocated(decl%accessibility)) then
                if (trim(lowercase_text(decl%accessibility)) == 'private') then
                    is_private = .true.
                    return
                end if
            end if
        end select
        select type (mod_nd => arena%entries(mod_index)%node)
        type is (module_node)
            is_private = module_symbol_is_private(arena, mod_nd, name)
        end select
    end function namelist_declaration_is_private

    ! Arena index of the module whose specification part holds this node, or 0.
    integer function namelist_owner_module(arena, node_index) result(owner)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        integer :: n, i

        owner = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (nd => arena%entries(n)%node)
            type is (module_node)
                if (.not. allocated(nd%declaration_indices)) cycle
                do i = 1, size(nd%declaration_indices)
                    if (nd%declaration_indices(i) == node_index) then
                        owner = n
                        return
                    end if
                end do
            end select
        end do
    end function namelist_owner_module

    ! True when the declared entity is of a derived type that carries, directly
    ! or through a component, a PRIVATE component defined in another module: a
    ! type whose components are inaccessible wherever it is use-associated
    ! cannot be a namelist group object.
    logical function namelist_member_has_private_components(arena, source, &
            decl_index, stmt_line) result(bad)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: source
        integer, intent(in) :: decl_index
        integer, intent(in) :: stmt_line
        character(len=:), allocatable :: base, here
        integer :: mod_index

        bad = .false.
        mod_index = namelist_owner_module(arena, decl_index)
        select type (decl => arena%entries(decl_index)%node)
        type is (declaration_node)
            if (.not. allocated(decl%type_name)) return
            base = namelist_derived_base_name(decl%type_name)
            if (len_trim(base) == 0) return
            bad = namelist_type_private_components(arena, base, mod_index, 0)
            if (bad) return
            ! A type defined in another program unit of the same file can reach
            ! the arena without its component list, so fall back to the source.
            call namelist_source_module_at(source, stmt_line, here)
            bad = namelist_source_type_private(source, base, here, 0)
        end select
    end function namelist_member_has_private_components

    ! Name of the module whose body contains `line`, or '' outside a module.
    subroutine namelist_source_module_at(source, line, mod_name)
        character(len=*), intent(in) :: source
        integer, intent(in) :: line
        character(len=:), allocatable, intent(out) :: mod_name
        character(len=:), allocatable :: low
        integer :: pos, next_nl, current

        mod_name = ''
        pos = 1
        current = 1
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                low = source(pos:)
                pos = len(source) + 1
            else
                low = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            if (current >= line) return
            current = current + 1
            low = trim(adjustl(lowercase_text(strip_data_source_comment(low))))
            if (index(low, 'end module') == 1) then
                mod_name = ''
            else if (index(low, 'module ') == 1) then
                low = trim(adjustl(low(8:)))
                if (index(low, 'procedure ') /= 1) mod_name = low
            end if
        end do
    end subroutine namelist_source_module_at

    ! True when the type `name` is defined in a module other than `here` and
    ! carries, directly or through a component, a PRIVATE component.
    recursive logical function namelist_source_type_private(source, name, here, &
            depth) result(bad)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: here
        integer, intent(in) :: depth
        character(len=:), allocatable :: low, cur_mod, owner, attrs, base
        integer :: pos, next_nl, sep
        logical :: inside, foreign

        bad = .false.
        if (depth > 8) return
        pos = 1
        cur_mod = ''
        owner = ''
        inside = .false.
        foreign = .false.
        do while (pos <= len(source))
            next_nl = index(source(pos:), new_line('a'))
            if (next_nl == 0) then
                low = source(pos:)
                pos = len(source) + 1
            else
                low = source(pos:pos + next_nl - 2)
                pos = pos + next_nl
            end if
            low = trim(adjustl(lowercase_text(strip_data_source_comment(low))))
            if (.not. inside) then
                if (index(low, 'end module') == 1) then
                    cur_mod = ''
                else if (index(low, 'module ') == 1) then
                    owner = trim(adjustl(low(8:)))
                    if (index(owner, 'procedure ') /= 1) cur_mod = owner
                end if
                if (.not. namelist_type_header_names(low, name)) cycle
                inside = .true.
                foreign = len_trim(cur_mod) > 0
                if (foreign) foreign = .not. same_name(cur_mod, here)
                cycle
            end if
            if (index(low, 'end type') == 1) return
            if (index(low, 'contains') == 1) return
            if (foreign .and. trim(low) == 'private') then
                bad = .true.
                return
            end if
            sep = index(low, '::')
            if (sep == 0) cycle
            attrs = low(1:sep - 1)
            if (foreign .and. index(attrs, 'private') > 0) then
                bad = .true.
                return
            end if
            base = namelist_derived_base_name(attrs)
            if (len_trim(base) == 0) cycle
            if (same_name(base, name)) cycle
            bad = namelist_source_type_private(source, base, here, depth + 1)
            if (bad) return
        end do
    end function namelist_source_type_private

    ! 'type(tp1)' or 'class(tp1)' -> 'tp1'; an intrinsic type -> ''.
    function namelist_derived_base_name(type_name) result(base)
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: base
        character(len=:), allocatable :: low
        integer :: lp, rp

        base = ''
        low = trim(lowercase_text(type_name))
        if (index(low, 'type(') /= 1 .and. index(low, 'class(') /= 1) return
        lp = index(low, '(')
        rp = index(low, ')', back=.true.)
        if (rp <= lp + 1) return
        base = trim(adjustl(low(lp + 1:rp - 1)))
    end function namelist_derived_base_name

    recursive logical function namelist_type_private_components(arena, name, &
            mod_index, depth) result(bad)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer, intent(in) :: mod_index
        integer, intent(in) :: depth
        integer :: t, owner

        bad = .false.
        if (depth > 8) return
        t = namelist_find_derived_type(arena, name)
        if (t == 0) return
        owner = namelist_owner_module(arena, t)
        select type (td => arena%entries(t)%node)
        type is (derived_type_node)
            if (.not. allocated(td%component_indices)) return
            if (owner /= 0 .and. owner /= mod_index) then
                if (namelist_components_private(arena, td%component_indices)) then
                    bad = .true.
                    return
                end if
            end if
            bad = namelist_components_private_nested(arena, td%component_indices, &
                                                     mod_index, depth)
        end select
    end function namelist_type_private_components

    logical function namelist_components_private(arena, components) result(bad)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: components(:)
        integer :: i

        bad = .false.
        do i = 1, size(components)
            if (.not. node_exists(arena, components(i))) cycle
            select type (comp => arena%entries(components(i))%node)
            type is (declaration_node)
                if (.not. allocated(comp%accessibility)) cycle
                if (trim(lowercase_text(comp%accessibility)) == 'private') then
                    bad = .true.
                    return
                end if
            type is (visibility_statement_node)
                ! A bare PRIVATE statement in the component part makes every
                ! component of the type private.
                if (comp%has_list) cycle
                if (comp%is_private) then
                    bad = .true.
                    return
                end if
            end select
        end do
    end function namelist_components_private

    recursive logical function namelist_components_private_nested(arena, &
            components, mod_index, depth) result(bad)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: components(:)
        integer, intent(in) :: mod_index
        integer, intent(in) :: depth
        character(len=:), allocatable :: base
        integer :: i

        bad = .false.
        do i = 1, size(components)
            if (.not. node_exists(arena, components(i))) cycle
            select type (comp => arena%entries(components(i))%node)
            type is (declaration_node)
                if (.not. allocated(comp%type_name)) cycle
                base = namelist_derived_base_name(comp%type_name)
                if (len_trim(base) == 0) cycle
                bad = namelist_type_private_components(arena, base, mod_index, &
                                                       depth + 1)
                if (bad) return
            end select
        end do
    end function namelist_components_private_nested

    integer function namelist_find_derived_type(arena, name) result(idx)
        type(ast_arena_t), intent(in) :: arena
        character(len=*), intent(in) :: name
        integer :: n

        idx = 0
        do n = 1, arena%size
            if (.not. node_exists(arena, n)) cycle
            select type (td => arena%entries(n)%node)
            type is (derived_type_node)
                if (.not. allocated(td%name)) cycle
                if (.not. same_name(td%name, name)) cycle
                ! A type can appear both as a forward entry and as the entry
                ! that carries the component list; prefer the latter.
                if (allocated(td%component_indices)) then
                    idx = n
                    return
                end if
                if (idx == 0) idx = n
            end select
        end do
    end function namelist_find_derived_type
