    ! OPEN / CLOSE / file-unit WRITE lowering (#247 slice B5c).
    !
    ! Each variable that receives a unit number via OPEN gets an alloca'd ptr
    ! slot in the lowering context (symbol_t%file_ptr_address). The fopen
    ! result is stored there; WRITE/CLOSE load it back. The Fortran unit number
    ! stored in the integer variable is not used at runtime.

    ! --- spec-text helpers ---------------------------------------------------

    subroutine parse_open_spec(spec, unit_str, newunit_var, file_path, &
                                status_str, error_msg)
        ! Extract keyword values from an OPEN specifier-list text.
        ! spec is the raw text inside the OPEN(...) parentheses.
        character(len=*), intent(in) :: spec
        character(len=:), allocatable, intent(out) :: unit_str
        character(len=:), allocatable, intent(out) :: newunit_var
        character(len=:), allocatable, intent(out) :: file_path
        character(len=:), allocatable, intent(out) :: status_str
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: text, kw, val
        integer :: p, eq_pos, cp, qend, arg_end
        logical :: first_arg
        character :: qc

        call set_empty(error_msg)
        unit_str = ''
        newunit_var = ''
        file_path = ''
        status_str = ''
        text = trim(adjustl(spec))
        p = 1
        first_arg = .true.
        do while (p <= len(text))
            do while (p <= len(text))
                if (text(p:p) == ' ' .or. text(p:p) == ',') then
                    p = p + 1
                else
                    exit
                end if
            end do
            if (p > len(text)) exit
            ! A keyword's '=' must appear before the next comma to count as a
            ! keyword argument; otherwise the argument is positional. The first
            ! positional argument of OPEN is the unit number (no unit= keyword).
            cp = index(text(p:), ',')
            if (cp <= 0) then
                arg_end = len(text)
            else
                arg_end = p + cp - 2
            end if
            eq_pos = index(text(p:arg_end), '=')
            if (eq_pos <= 0) then
                if (first_arg) unit_str = trim(adjustl(text(p:arg_end)))
                p = arg_end + 2
                first_arg = .false.
                cycle
            end if
            first_arg = .false.
            eq_pos = p + eq_pos - 1
            kw = spec_lower(text(p:eq_pos - 1))
            p = eq_pos + 1
            do while (p <= len(text) .and. text(p:p) == ' ')
                p = p + 1
            end do
            if (p > len(text)) exit
            if (text(p:p) == "'" .or. text(p:p) == '"') then
                qc = text(p:p)
                qend = index(text(p + 1:), qc)
                if (qend <= 0) then
                    error_msg = 'unterminated string in OPEN spec'
                    return
                end if
                qend = p + qend
                val = text(p + 1:qend - 1)
                p = qend + 1
            else
                cp = index(text(p:), ',')
                if (cp <= 0) then
                    val = adjustl(text(p:))
                    p = len(text) + 1
                else
                    val = adjustl(text(p:p + cp - 2))
                    p = p + cp
                end if
            end if
            select case (trim(kw))
            case ('unit')
                unit_str = trim(val)
            case ('newunit')
                newunit_var = trim(val)
            case ('file')
                file_path = trim(val)
            case ('status')
                status_str = spec_lower(trim(val))
            end select
        end do
    end subroutine parse_open_spec

    function spec_lower(s) result(t)
        character(len=*), intent(in) :: s
        character(len=len(s)) :: t
        integer :: i, c
        do i = 1, len(s)
            c = ichar(s(i:i))
            if (c >= ichar('A') .and. c <= ichar('Z')) then
                t(i:i) = char(c + 32)
            else
                t(i:i) = s(i:i)
            end if
        end do
    end function spec_lower

    ! --- OPEN ----------------------------------------------------------------

    subroutine lower_open(arena, node, context, error_msg)
        ! Emit fopen(file, mode) and store FILE* in the unit-variable's slot.
        use liric_session_memory_bindings, only: emit_ptr_alloca, emit_ptr_store, &
                                                  emit_i32_alloca, emit_i32_store
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(open_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: ustr, nuvar, fpath, sstr, mode_str
        type(lr_operand_desc_t) :: file_ptr_op, mode_ptr_op, file_handle
        type(lr_operand_desc_t) :: uval_op, ptr_slot
        type(lr_operand_desc_t) :: fa(2)
        type(lr_operand_desc_t), allocatable :: noargs(:)
        integer :: unit_number, sym, ios
        character(len=64) :: fgn, mgn
        integer(c_int32_t) :: fgid, mgid
        call set_empty(error_msg)
        if (.not. allocated(node%unit_spec)) then
            call unsupported_feature_error('open statement', node%line, &
                node%column, 'missing specifier list (#247 B5c)', error_msg)
            return
        end if

        call parse_open_spec(node%unit_spec, ustr, nuvar, fpath, sstr, error_msg)
        if (len_trim(error_msg) > 0) return

        if (len_trim(fpath) == 0) then
            ! A connection without file= is a scratch file. gfortran also
            ! treats status='scratch' and a bare default connection as a
            ! temporary; tmpfile() gives a read/write handle deleted on close.
            allocate (noargs(0))
            if (.not. emit_ptr_call(context%session, 'tmpfile', noargs, &
                                     file_handle, error_msg)) return
        else
            ! Fortran units are read/write; use update modes so a unit can be
            ! written, rewound, and read back within the same scope.
            select case (trim(sstr))
            case ('old')
                mode_str = 'r+'
            case default
                mode_str = 'w+'
            end select

            context%string_literal_count = context%string_literal_count + 1
            write (fgn, '(A,I0)') '.ffc.open.file.', context%string_literal_count
            call create_printf_format_global(context%session, trim(fgn), &
                                             trim(fpath), fgid, error_msg)
            if (len_trim(error_msg) > 0) return
            file_ptr_op = printf_format_ptr(context%session, fgid)

            context%string_literal_count = context%string_literal_count + 1
            write (mgn, '(A,I0)') '.ffc.open.mode.', context%string_literal_count
            call create_printf_format_global(context%session, trim(mgn), &
                                             trim(mode_str), mgid, error_msg)
            if (len_trim(error_msg) > 0) return
            mode_ptr_op = printf_format_ptr(context%session, mgid)

            fa(1) = file_ptr_op
            fa(2) = mode_ptr_op
            if (.not. emit_ptr_call(context%session, 'fopen', fa, &
                                     file_handle, error_msg)) return
        end if

        if (len_trim(nuvar) > 0) then
            unit_number = int(context%next_file_unit)
            context%next_file_unit = context%next_file_unit + 1_c_int32_t
            uval_op = i32_immediate(context%session, int(unit_number, c_int64_t))
            sym = find_symbol(context, trim(nuvar))
            if (sym > 0) then
                if (.not. context%symbols(sym)%has_address) then
                    if (.not. emit_i32_alloca(context%session, &
                                               context%symbols(sym)%address, &
                                               error_msg)) return
                    context%symbols(sym)%has_address = .true.
                    context%symbols(sym)%is_reference = .true.
                end if
                if (.not. emit_i32_store(context%session, uval_op, &
                                          context%symbols(sym)%address, &
                                          error_msg)) return
                context%symbols(sym)%value = uval_op
                if (.not. context%symbols(sym)%is_file_unit) then
                    if (.not. emit_ptr_alloca(context%session, ptr_slot, &
                                               error_msg)) return
                    context%symbols(sym)%file_ptr_address = ptr_slot
                    context%symbols(sym)%is_file_unit = .true.
                end if
                if (.not. emit_ptr_store(context%session, file_handle, &
                                          context%symbols(sym)%file_ptr_address, &
                                          error_msg)) return
            end if
        else if (len_trim(ustr) > 0) then
            read (ustr, *, iostat=ios) unit_number
            if (ios == 0) then
                call track_literal_unit(context, unit_number, file_handle, &
                                         error_msg)
            else
                sym = find_symbol(context, trim(ustr))
                if (sym <= 0) then
                    error_msg = 'open unit= variable not declared: '//trim(ustr)
                    return
                end if
                if (.not. context%symbols(sym)%is_file_unit) then
                    if (.not. emit_ptr_alloca(context%session, ptr_slot, &
                                               error_msg)) return
                    context%symbols(sym)%file_ptr_address = ptr_slot
                    context%symbols(sym)%is_file_unit = .true.
                end if
                if (.not. emit_ptr_store(context%session, file_handle, &
                                          context%symbols(sym)%file_ptr_address, &
                                          error_msg)) return
                ! Alias the numeric unit so references by literal number resolve
                ! to the same FILE* slot as references by variable name.
                if (context%symbols(sym)%has_unit_const) then
                    call alias_unit_to_slot(context, &
                        context%symbols(sym)%unit_const, &
                        context%symbols(sym)%file_ptr_address, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end if
        else
            call unsupported_feature_error('open statement', node%line, &
                node%column, 'unit= or newunit= required (#247 B5c)', error_msg)
            return
        end if
        call set_empty(error_msg)
    end subroutine lower_open

    subroutine track_literal_unit(context, unit_number, file_handle, error_msg)
        use liric_session_memory_bindings, only: emit_ptr_alloca, emit_ptr_store
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: unit_number
        type(lr_operand_desc_t), intent(in) :: file_handle
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=24) :: pseudo_name
        type(lr_operand_desc_t) :: ptr_slot
        integer :: sym

        write (pseudo_name, '(A,I0)') '.unit.', unit_number
        sym = find_symbol(context, trim(pseudo_name))
        if (sym <= 0) then
            call define_symbol(context, trim(pseudo_name), VALUE_C_PTR, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, trim(pseudo_name))
        end if
        if (.not. context%symbols(sym)%is_file_unit) then
            if (.not. emit_ptr_alloca(context%session, ptr_slot, error_msg)) return
            context%symbols(sym)%file_ptr_address = ptr_slot
            context%symbols(sym)%is_file_unit = .true.
        end if
        if (.not. emit_ptr_store(context%session, file_handle, &
                                  context%symbols(sym)%file_ptr_address, &
                                  error_msg)) return
        call set_empty(error_msg)
    end subroutine track_literal_unit

    subroutine alias_unit_to_slot(context, unit_number, slot, error_msg)
        ! Make the numeric pseudo-symbol .unit.<N> share an existing FILE* slot.
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: unit_number
        type(lr_operand_desc_t), intent(in) :: slot
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=24) :: pseudo_name
        integer :: sym

        call set_empty(error_msg)
        write (pseudo_name, '(A,I0)') '.unit.', unit_number
        sym = find_symbol(context, trim(pseudo_name))
        if (sym <= 0) then
            call define_symbol(context, trim(pseudo_name), VALUE_C_PTR, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol(context, trim(pseudo_name))
        end if
        context%symbols(sym)%file_ptr_address = slot
        context%symbols(sym)%is_file_unit = .true.
    end subroutine alias_unit_to_slot

    ! --- CLOSE ---------------------------------------------------------------

    subroutine lower_close(node, context, error_msg)
        use liric_session_memory_bindings, only: emit_ptr_load, emit_ptr_store
        type(close_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp, fa(1), is_open, slot
        integer(c_int32_t) :: do_close, after_close

        call set_empty(error_msg)
        if (.not. allocated(node%unit_spec)) then
            call unsupported_feature_error('close statement', node%line, &
                node%column, 'missing unit specifier (#247 B5c)', error_msg)
            return
        end if
        call resolve_unit_slot(node%unit_spec, context, slot, error_msg)
        if (len_trim(error_msg) > 0) return

        ! Guard fclose on a non-null handle and null the slot afterwards.
        ! Closing an already-closed unit is a no-op in Fortran; without the
        ! guard a second CLOSE would fclose a freed FILE* and abort.
        if (.not. emit_ptr_load(context%session, slot, fp, error_msg)) return
        if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, fp, &
                null_ptr_operand(context), is_open, error_msg)) return
        do_close = create_liric_block(context%session)
        after_close = create_liric_block(context%session)
        if (.not. emit_liric_condbr(context%session, is_open, do_close, &
                                     after_close, error_msg)) return

        if (.not. set_liric_block(context%session, do_close, error_msg)) return
        fa(1) = fp
        if (.not. emit_void_call(context%session, 'fclose', fa, error_msg)) return
        if (.not. emit_ptr_store(context%session, null_ptr_operand(context), &
                                  slot, error_msg)) return
        if (.not. emit_liric_br(context%session, after_close, error_msg)) return

        if (.not. set_liric_block(context%session, after_close, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_close

    subroutine resolve_unit_slot(unit_spec, context, slot, error_msg)
        ! Resolve unit_spec (raw OPEN/CLOSE specifier text, e.g. "u", "10",
        ! "unit=u") to the alloca'd ptr slot holding the unit's FILE*.
        character(len=*), intent(in) :: unit_spec
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: slot
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: plain, head
        integer :: unit_number, sym, ios, eq_pos, cpos
        character(len=24) :: pseudo_name

        call set_empty(error_msg)
        plain = trim(adjustl(unit_spec))
        ! REWIND/BACKSPACE keep the surrounding parentheses, e.g. "(u)".
        if (len(plain) >= 2) then
            if (plain(1:1) == '(' .and. plain(len(plain):len(plain)) == ')') then
                plain = trim(adjustl(plain(2:len(plain) - 1)))
            end if
        end if
        ! CLOSE/REWIND carry trailing keyword arguments (e.g. status="delete");
        ! the unit is the first argument. Take only that argument, then drop a
        ! leading unit= keyword if present.
        cpos = index(plain, ',')
        if (cpos > 0) then
            head = trim(adjustl(plain(1:cpos - 1)))
        else
            head = trim(plain)
        end if
        eq_pos = index(head, '=')
        if (eq_pos > 0) head = adjustl(head(eq_pos + 1:))
        plain = trim(head)

        unit_number = -1
        read (plain, *, iostat=ios) unit_number
        if (ios == 0) then
            write (pseudo_name, '(A,I0)') '.unit.', unit_number
            sym = find_symbol(context, trim(pseudo_name))
        else
            unit_number = -1
            sym = find_symbol(context, trim(plain))
            if (sym > 0) then
                if (context%symbols(sym)%has_unit_const) &
                    unit_number = context%symbols(sym)%unit_const
            end if
        end if
        if (sym <= 0 .or. .not. context%symbols(sym)%is_file_unit) then
            ! Implicit preconnection: a numeric unit used without OPEN connects
            ! to fort.<N>, matching gfortran's default unit handling.
            if (unit_number >= 0) then
                call connect_implicit_unit(context, unit_number, sym, error_msg)
                if (len_trim(error_msg) > 0) return
            else
                error_msg = 'no open unit for: '//trim(unit_spec)//' (#247 B5c)'
                return
            end if
        end if
        slot = context%symbols(sym)%file_ptr_address
    end subroutine resolve_unit_slot

    subroutine load_unit_file_ptr(unit_spec, context, fp, error_msg)
        ! Load the current FILE* for a unit from its alloca'd ptr slot.
        use liric_session_memory_bindings, only: emit_ptr_load
        character(len=*), intent(in) :: unit_spec
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: fp
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: slot

        call resolve_unit_slot(unit_spec, context, slot, error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_load(context%session, slot, fp, error_msg)) return
        call set_empty(error_msg)
    end subroutine load_unit_file_ptr

    subroutine connect_implicit_unit(context, unit_number, sym, error_msg)
        ! fopen("fort.<N>", "w+") for a numeric unit used without OPEN.
        use liric_session_memory_bindings, only: emit_ptr_alloca, emit_ptr_store
        type(lowering_context_t), intent(inout) :: context
        integer, intent(in) :: unit_number
        integer, intent(out) :: sym
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: file_ptr_op, mode_ptr_op, file_handle
        type(lr_operand_desc_t) :: ptr_slot, fa(2)
        character(len=64) :: fgn, mgn, fname
        integer(c_int32_t) :: fgid, mgid

        call set_empty(error_msg)
        write (fname, '(A,I0)') 'fort.', unit_number
        context%string_literal_count = context%string_literal_count + 1
        write (fgn, '(A,I0)') '.ffc.open.file.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fgn), &
                                         trim(fname), fgid, error_msg)
        if (len_trim(error_msg) > 0) return
        file_ptr_op = printf_format_ptr(context%session, fgid)

        context%string_literal_count = context%string_literal_count + 1
        write (mgn, '(A,I0)') '.ffc.open.mode.', context%string_literal_count
        call create_printf_format_global(context%session, trim(mgn), 'w+', &
                                         mgid, error_msg)
        if (len_trim(error_msg) > 0) return
        mode_ptr_op = printf_format_ptr(context%session, mgid)

        fa(1) = file_ptr_op
        fa(2) = mode_ptr_op
        if (.not. emit_ptr_call(context%session, 'fopen', fa, file_handle, &
                                error_msg)) return
        if (.not. emit_ptr_alloca(context%session, ptr_slot, error_msg)) return
        if (.not. emit_ptr_store(context%session, file_handle, ptr_slot, &
                                 error_msg)) return
        call alias_unit_to_slot(context, unit_number, ptr_slot, error_msg)
        if (len_trim(error_msg) > 0) return
        sym = find_symbol(context, file_unit_pseudo_name(unit_number))
    end subroutine connect_implicit_unit

    ! --- file-unit WRITE -----------------------------------------------------

    logical function is_file_unit_write(node, context)
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        integer :: sym, ios, unit_number

        is_file_unit_write = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*' .or. &
            trim(node%unit_spec) == '6') return
        read (node%unit_spec, *, iostat=ios) unit_number
        if (ios == 0) then
            is_file_unit_write = .true.
            return
        end if
        sym = find_symbol(context, trim(node%unit_spec))
        if (sym <= 0) return
        is_file_unit_write = context%symbols(sym)%is_file_unit .or. &
                             context%symbols(sym)%has_unit_const
    end function is_file_unit_write

    subroutine lower_write_file(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fp
        logical :: list_dir
        integer :: i

        call set_empty(error_msg)
        call load_unit_file_ptr(node%unit_spec, context, fp, error_msg)
        if (len_trim(error_msg) > 0) then
            error_msg = ''
            call unsupported_feature_error('write to file unit', &
                node%line, node%column, &
                'unit not opened in same scope (#247 B5c)', error_msg)
            return
        end if

        ! A write to a connected unit always succeeds here; iostat = 0.
        call store_write_iostat_success(node, context, error_msg)
        if (len_trim(error_msg) > 0) return

        list_dir = .not. allocated(node%format_spec)
        if (allocated(node%format_spec)) then
            if (trim(node%format_spec) == '*') list_dir = .true.
        end if

        if (list_dir) then
            if (.not. allocated(node%arg_indices)) return
            do i = 1, size(node%arg_indices)
                call lower_file_write_item(arena, node%arg_indices(i), fp, &
                                            context, error_msg)
                if (len_trim(error_msg) > 0) return
            end do
            call lower_file_write_newline(fp, context, error_msg)
        else
            call lower_file_write_formatted(arena, node, fp, context, error_msg)
        end if
    end subroutine lower_write_file

    subroutine store_write_iostat_success(node, context, error_msg)
        ! Populate a WRITE's iostat= target with 0 (success). The variable name
        ! lives in the io_control_list text (iostat=<name>).
        type(write_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: var_name
        integer :: sym

        call set_empty(error_msg)
        if (.not. allocated(node%io_control_list)) return
        call io_control_value(node%io_control_list, 'iostat', var_name)
        if (len_trim(var_name) == 0) return
        sym = find_symbol(context, var_name)
        if (sym <= 0) then
            error_msg = 'iostat target was not declared: '//trim(var_name)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_I32) then
            error_msg = 'iostat target must be a default integer: '//trim(var_name)
            return
        end if
        call assign_i32_to_symbol(context, sym, &
                                  i32_immediate(context%session, 0_c_int64_t), &
                                  error_msg)
    end subroutine store_write_iostat_success

    integer function file_write_value_kind(arena, node_index, context)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lowering_context_t), intent(inout) :: context

        file_write_value_kind = expression_value_kind(arena, node_index, context, &
                                                       VALUE_I32)
    end function file_write_value_kind

    subroutine lower_file_write_item(arena, node_index, fp, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        type(lr_operand_desc_t), intent(in) :: fp
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: val, fmtop
        type(lr_operand_desc_t), allocatable :: fa(:)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn
        character(len=:), allocatable :: fmts
        integer :: vk

        call set_empty(error_msg)
        vk = file_write_value_kind(arena, node_index, context)

        select case (vk)
        case (VALUE_I32, VALUE_LOGICAL)
            fmts = ' %d'
        case (VALUE_I64)
            fmts = ' %ld'
        case (VALUE_F32)
            fmts = ' %g'
        case (VALUE_F64)
            fmts = ' %.15g'
        case default
            fmts = ' %d'
        end select

        select case (vk)
        case (VALUE_I32, VALUE_LOGICAL)
            call lower_i32_expression(arena, node_index, context, val, error_msg)
        case (VALUE_I64)
            call lower_i64_expression(arena, node_index, context, val, error_msg)
        case (VALUE_F32)
            call lower_f32_expression(arena, node_index, context, val, error_msg)
        case (VALUE_F64)
            call lower_f64_expression(arena, node_index, context, val, error_msg)
        case default
            call lower_i32_expression(arena, node_index, context, val, error_msg)
        end select
        if (len_trim(error_msg) > 0) return

        context%string_literal_count = context%string_literal_count + 1
        write (fgn, '(A,I0)') '.ffc.fprintf.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fgn), &
                                          trim(fmts), fgid, error_msg)
        if (len_trim(error_msg) > 0) return
        fmtop = printf_format_ptr(context%session, fgid)

        allocate (fa(3))
        fa(1) = fp
        fa(2) = fmtop
        fa(3) = val
        if (.not. emit_fprintf(context%session, fa, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_file_write_item

    subroutine lower_file_write_newline(fp, context, error_msg)
        type(lr_operand_desc_t), intent(in) :: fp
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: fmtop, fa(2)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn

        context%string_literal_count = context%string_literal_count + 1
        write (fgn, '(A,I0)') '.ffc.fprintf.nl.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fgn), &
                                          achar(10), fgid, error_msg)
        if (len_trim(error_msg) > 0) return
        fmtop = printf_format_ptr(context%session, fgid)
        fa(1) = fp
        fa(2) = fmtop
        if (.not. emit_fprintf(context%session, fa, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_file_write_newline

    subroutine lower_file_write_formatted(arena, node, fp, context, error_msg)
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(ast_arena_t), intent(in) :: arena
        type(write_statement_node), intent(in) :: node
        type(lr_operand_desc_t), intent(in) :: fp
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: fmt_body, c_fmt
        type(lr_operand_desc_t) :: fmtop
        type(lr_operand_desc_t), allocatable :: fa(:)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn
        integer :: i

        call set_empty(error_msg)
        call normalize_format_body(node%format_spec, fmt_body)
        if (.not. fortran_fmt_to_c(trim(fmt_body), c_fmt, error_msg)) return

        context%string_literal_count = context%string_literal_count + 1
        write (fgn, '(A,I0)') '.ffc.fprintf.fmt.', context%string_literal_count
        call create_printf_format_global(context%session, trim(fgn), &
                                          trim(c_fmt)//achar(10), fgid, error_msg)
        if (len_trim(error_msg) > 0) return
        fmtop = printf_format_ptr(context%session, fgid)

        if (.not. allocated(node%arg_indices)) then
            allocate (fa(2))
            fa(1) = fp
            fa(2) = fmtop
            if (.not. emit_fprintf(context%session, fa, error_msg)) return
            return
        end if

        allocate (fa(size(node%arg_indices) + 2))
        fa(1) = fp
        fa(2) = fmtop
        do i = 1, size(node%arg_indices)
            select case (file_write_value_kind(arena, node%arg_indices(i), context))
            case (VALUE_I32, VALUE_LOGICAL)
                call lower_i32_expression(arena, node%arg_indices(i), context, &
                                           fa(i + 2), error_msg)
            case (VALUE_I64)
                call lower_i64_expression(arena, node%arg_indices(i), context, &
                                           fa(i + 2), error_msg)
            case (VALUE_F32)
                call lower_f32_expression(arena, node%arg_indices(i), context, &
                                           fa(i + 2), error_msg)
            case (VALUE_F64)
                call lower_f64_expression(arena, node%arg_indices(i), context, &
                                           fa(i + 2), error_msg)
            case default
                call lower_i32_expression(arena, node%arg_indices(i), context, &
                                           fa(i + 2), error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
        if (.not. emit_fprintf(context%session, fa, error_msg)) return
        call set_empty(error_msg)
    end subroutine lower_file_write_formatted

    logical function fortran_fmt_to_c(fort_fmt, c_fmt, error_msg)
        character(len=*), intent(in) :: fort_fmt
        character(len=:), allocatable, intent(out) :: c_fmt
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: i, w, d
        character :: ch
        character(len=32) :: buf

        fortran_fmt_to_c = .false.
        c_fmt = ''
        call set_empty(error_msg)
        i = 1
        do while (i <= len(fort_fmt))
            ch = fort_fmt(i:i)
            select case (ch)
            case ('I', 'i')
                i = i + 1
                call read_fmt_int(fort_fmt, i, w)
                if (w > 0) then
                    write (buf, '(I0)') w
                    c_fmt = c_fmt//'%'//trim(buf)//'d'
                else
                    c_fmt = c_fmt//'%d'
                end if
            case ('F', 'f')
                i = i + 1
                call read_fmt_int(fort_fmt, i, w)
                d = 6
                if (i <= len(fort_fmt) .and. fort_fmt(i:i) == '.') then
                    i = i + 1
                    call read_fmt_int(fort_fmt, i, d)
                end if
                write (buf, '(I0,A,I0)') w, '.', d
                c_fmt = c_fmt//'%'//trim(buf)//'f'
            case ('E', 'e', 'G', 'g')
                i = i + 1
                call read_fmt_int(fort_fmt, i, w)
                d = 6
                if (i <= len(fort_fmt) .and. fort_fmt(i:i) == '.') then
                    i = i + 1
                    call read_fmt_int(fort_fmt, i, d)
                end if
                write (buf, '(I0,A,I0)') w, '.', d
                c_fmt = c_fmt//'%'//trim(buf)//'g'
            case ('A', 'a')
                i = i + 1
                call read_fmt_int(fort_fmt, i, w)
                if (w > 0) then
                    write (buf, '(I0)') w
                    c_fmt = c_fmt//'%'//trim(buf)//'s'
                else
                    c_fmt = c_fmt//'%s'
                end if
            case ('X', 'x')
                c_fmt = c_fmt//' '
                i = i + 1
            case ('/', ',', ' ')
                i = i + 1
            case ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
                call read_fmt_int(fort_fmt, i, w)
            case default
                i = i + 1
            end select
        end do
        fortran_fmt_to_c = .true.
    end function fortran_fmt_to_c

    subroutine read_fmt_int(s, pos, val)
        character(len=*), intent(in) :: s
        integer, intent(inout) :: pos
        integer, intent(out) :: val
        val = 0
        do while (pos <= len(s))
            if (s(pos:pos) >= '0' .and. s(pos:pos) <= '9') then
                val = val * 10 + ichar(s(pos:pos)) - ichar('0')
                pos = pos + 1
            else
                exit
            end if
        end do
    end subroutine read_fmt_int
