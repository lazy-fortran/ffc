submodule (session_program_lowering_impl) session_program_lowering_open_close
    implicit none
contains
    module procedure parse_open_spec
        ! Extract keyword values from an OPEN specifier-list text.
        ! spec is the raw text inside the OPEN(...) parentheses.
        character(len=:), allocatable :: text, kw, val
        integer :: p, eq_pos, cp, qend, arg_end
        logical :: first_arg, val_quoted
        character :: qc

        call set_empty(error_msg)
        unit_str = ''
        newunit_var = ''
        file_path = ''
        file_quoted = .false.
        status_str = ''
        status_quoted = .false.
        form_str = ''
        access_str = ''
        sign_str = ''
        iostat_var = ''
        iomsg_var = ''
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
            val_quoted = .false.
            if (text(p:p) == "'" .or. text(p:p) == '"') then
                val_quoted = .true.
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
                file_quoted = val_quoted
            case ('status')
                status_str = spec_lower(trim(val))
                status_quoted = val_quoted
            case ('form')
                form_str = spec_lower(trim(val))
            case ('access')
                access_str = spec_lower(trim(val))
            case ('sign')
                sign_str = spec_lower(trim(val))
            case ('iostat')
                iostat_var = trim(val)
            case ('iomsg')
                iomsg_var = trim(val)
            end select
        end do
    end procedure parse_open_spec
    module procedure spec_lower
        integer :: i, c
        do i = 1, len(s)
            c = ichar(s(i:i))
            if (c >= ichar('A') .and. c <= ichar('Z')) then
                t(i:i) = char(c + 32)
            else
                t(i:i) = s(i:i)
            end if
        end do
    end procedure spec_lower
    module procedure lower_open
        ! Connect a unit through the runtime (#396). The compiler decides
        ! which unit number and which STATUS= apply; the runtime owns the
        ! connection itself, so a unit opened here stays connected for the
        ! life of the process rather than for the life of a lowered function.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i32_store
        use, intrinsic :: iso_c_binding, only: c_int64_t
        character(len=:), allocatable :: ustr, nuvar, fpath, sstr, formstr, accessstr
        character(len=:), allocatable :: signstr
        character(len=:), allocatable :: iostat_var, iomsg_var
        type(lr_operand_desc_t) :: unit_op, args(4), status_op
        integer :: sym, ios, unit_number
        logical :: file_quoted, status_quoted, unformatted

        call set_empty(error_msg)
        if (.not. allocated(node%unit_spec)) then
            call unsupported_feature_error('open statement', node%line, &
                node%column, 'missing specifier list (#247 B5c)', error_msg)
            return
        end if

        call parse_open_spec(node%unit_spec, ustr, nuvar, fpath, file_quoted, &
                              sstr, status_quoted, formstr, accessstr, &
                              signstr, iostat_var, iomsg_var, error_msg)
        if (len_trim(error_msg) > 0) return
        unformatted = trim(formstr) == 'unformatted' .or. &
                      (len_trim(formstr) == 0 .and. &
                       (trim(accessstr) == 'stream' .or. &
                        trim(accessstr) == 'direct'))

        ! F2018 12.5.6.12: OPEN with NEWUNIT= must also give FILE= or
        ! STATUS='SCRATCH'; without either there is nothing to connect to.
        if (len_trim(nuvar) > 0 .and. len_trim(fpath) == 0) then
            if (status_quoted .and. trim(sstr) /= 'scratch') then
                error_msg = 'NEWUNIT specifier must have FILE= or '// &
                    'STATUS=''SCRATCH'' in OPEN statement'
                return
            end if
        end if

        ! Unit 6 is preconnected to stdout: OPEN reconfigures that connection
        ! rather than opening a distinct file, so its sign mode also governs
        ! PRINT and WRITE(*,...), which share the same connection (#280).
        if (trim(ustr) == '6') then
            context%stdout_force_plus_sign = (trim(signstr) == 'plus')
        end if

        if (len_trim(nuvar) > 0) then
            call allocate_newunit(context, nuvar, unit_op, error_msg)
            if (len_trim(error_msg) > 0) return
            call set_unit_form(context, nuvar, unformatted, error_msg)
            if (len_trim(error_msg) > 0) return
        else if (len_trim(ustr) > 0) then
            call unit_number_operand(ustr, context, unit_op, error_msg)
            if (len_trim(error_msg) > 0) return
            read (ustr, *, iostat=ios) unit_number
            if (ios == 0) then
                call mark_unit_symbol(context, file_unit_pseudo_name(unit_number), &
                                      error_msg)
                if (len_trim(error_msg) == 0) then
                    call set_unit_form(context, file_unit_pseudo_name(unit_number), &
                                       unformatted, error_msg)
                end if
            else
                sym = find_symbol_compat(context, trim(ustr))
                if (sym <= 0) then
                    error_msg = 'open unit= variable not declared: '//trim(ustr)
                    return
                end if
                context%symbols(sym)%is_file_unit = .true.
                context%symbols(sym)%is_unformatted = unformatted
                ! References by literal number reach the same connection,
                ! because both resolve to the same runtime unit number.
                if (context%symbols(sym)%has_unit_const) then
                    call mark_unit_symbol(context, &
                        file_unit_pseudo_name(context%symbols(sym)%unit_const), &
                        error_msg)
                    if (len_trim(error_msg) > 0) return
                    call set_unit_form(context, &
                                       file_unit_pseudo_name(context%symbols(sym)%unit_const), &
                                       unformatted, error_msg)
                    if (len_trim(error_msg) > 0) return
                end if
            end if
            if (len_trim(error_msg) > 0) return
        else
            call unsupported_feature_error('open statement', node%line, &
                node%column, 'unit= or newunit= required (#247 B5c)', error_msg)
            return
        end if

        args(1) = unit_op
        call open_file_operands(context, fpath, file_quoted, args(2), args(3), &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call open_status_operand(context, trim(sstr), status_quoted, args(4), &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_call(context%session, '_ffc_unit_open', args, &
                                status_op, error_msg)) return
        ! The OPEN's own return value is its IOSTAT=, so a failed connection
        ! reports the runtime's code instead of being lost (#427).
        call store_io_status(context, iostat_var, iomsg_var, status_op, &
                             error_msg)
        if (len_trim(error_msg) > 0) return
        call set_empty(error_msg)
    end procedure lower_open
    module procedure open_status_operand
        ! STATUS= may be a character expression. Literal values are emitted as
        ! globals; a variable must pass its current character buffer so the
        ! runtime sees its value rather than the variable's spelling (#628).
        integer :: sym
        type(lr_operand_desc_t) :: unused_length

        call set_empty(error_msg)
        if (status_quoted .or. len_trim(status_text) == 0) then
            call unit_string_operand(context, 'open.status.', status_text, &
                                     status_op, error_msg)
            return
        end if
        sym = find_symbol_compat(context, trim(status_text))
        if (sym <= 0) then
            error_msg = 'open status= variable not declared: '//trim(status_text)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_CHARACTER) then
            error_msg = 'open status= requires a character value: '// &
                trim(status_text)
            return
        end if
        call char_length_operands(context, sym, status_op, unused_length, &
                                   error_msg)
    end procedure open_status_operand
    module procedure open_file_operands
        ! FILE= yields a data pointer and a byte count. A quoted literal
        ! contributes its own text; anything else names a character variable,
        ! whose declared width is passed so the runtime can trim the padding.
        use, intrinsic :: iso_c_binding, only: c_int64_t
        integer :: sym

        call set_empty(error_msg)
        if (file_quoted .or. len_trim(fpath) == 0) then
            call unit_string_operand(context, 'open.file.', fpath, data_op, &
                                     error_msg)
            if (len_trim(error_msg) > 0) return
            len_op = i32_immediate(context%session, int(len(fpath), c_int64_t))
            return
        end if

        sym = find_symbol_compat(context, trim(fpath))
        if (sym <= 0) then
            error_msg = 'open file= variable not declared: '//trim(fpath)
            return
        end if
        if (context%symbols(sym)%character_length <= 0 .and. &
            .not. context%symbols(sym)%is_deferred_character) then
            error_msg = 'open file= requires a character value: '//trim(fpath)
            return
        end if
        call char_length_operands(context, sym, data_op, len_op, error_msg)
    end procedure open_file_operands
    module procedure unit_string_operand
        ! Materialise a NUL-terminated literal and yield a pointer to it.
        character(len=64) :: gname
        integer(c_int32_t) :: gid

        call set_empty(error_msg)
        context%string_literal_count = context%string_literal_count + 1
        gname = ffc_unit_global_name(context, tag, context%string_literal_count)
        call create_printf_format_global(context%session, trim(gname), &
                                         trim(text), gid, error_msg)
        if (len_trim(error_msg) > 0) return
        ptr_op = printf_format_ptr(context%session, gid)
    end procedure unit_string_operand
    module procedure allocate_newunit
        ! OPEN(newunit=u): the runtime picks the unit number, so two scopes
        ! that both use NEWUNIT= can never be handed the same unit.
        use liric_session_memory_bindings, only: emit_i32_alloca, emit_i32_store
        type(lr_operand_desc_t), allocatable :: noargs(:)
        integer :: sym

        call set_empty(error_msg)
        allocate (noargs(0))
        if (.not. emit_i32_call(context%session, '_ffc_unit_newunit', noargs, &
                                unit_op, error_msg)) return
        sym = find_symbol_compat(context, trim(nuvar))
        if (sym <= 0) then
            error_msg = 'open newunit= variable not declared: '//trim(nuvar)
            return
        end if
        if (.not. context%symbols(sym)%has_address) then
            if (.not. emit_i32_alloca(context%session, &
                                       context%symbols(sym)%address, &
                                       error_msg)) return
            context%symbols(sym)%has_address = .true.
            context%symbols(sym)%is_reference = .true.
        end if
        if (.not. emit_i32_store(context%session, unit_op, &
                                  context%symbols(sym)%address, error_msg)) return
        context%symbols(sym)%value = unit_op
        context%symbols(sym)%is_file_unit = .true.
    end procedure allocate_newunit
    module procedure mark_unit_symbol
        ! Record that a numeric unit is in use, so statement classification
        ! and INQUIRE(opened=) can tell a file unit from a plain integer.
        integer :: sym

        call set_empty(error_msg)
        sym = find_symbol_compat(context, pseudo_name)
        if (sym <= 0) then
            call define_symbol(context, pseudo_name, VALUE_I32, error_msg)
            if (len_trim(error_msg) > 0) return
            sym = find_symbol_compat(context, pseudo_name)
        end if
        if (sym > 0) context%symbols(sym)%is_file_unit = .true.
    end procedure mark_unit_symbol
    module procedure set_unit_form
        integer :: sym

        call set_empty(error_msg)
        sym = find_symbol_compat(context, trim(name))
        if (sym <= 0) then
            error_msg = 'open unit was not declared: '//trim(name)
            return
        end if
        context%symbols(sym)%is_unformatted = unformatted
    end procedure set_unit_form
    module procedure store_io_status
        ! Report one I/O statement's outcome. status_op is the value the
        ! runtime returned for the operation; when it is absent the caller
        ! wants the runtime's record of the last operation instead, which is
        ! the same number reached by a different route.
        !
        ! Both specifiers are optional and independent, and each is left
        ! alone when its specifier is absent.
        use, intrinsic :: iso_c_binding, only: c_int64_t

        call set_empty(error_msg)
        if (len_trim(iostat_name) > 0) then
            call store_iostat_value(context, iostat_name, status_op, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        if (len_trim(iomsg_name) > 0) then
            call store_iomsg_text(context, iomsg_name, error_msg)
            if (len_trim(error_msg) > 0) return
        end if
    end procedure store_io_status
    module procedure store_iostat_value
        integer :: sym

        call set_empty(error_msg)
        sym = find_symbol_compat(context, trim(name))
        if (sym <= 0) then
            error_msg = 'iostat target was not declared: '//trim(name)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_I32) then
            error_msg = 'iostat target must be a default integer: '//trim(name)
            return
        end if
        call assign_i32_to_symbol(context, sym, status_op, error_msg)
    end procedure store_iostat_value
    module procedure store_iomsg_text
        ! _ffc_iomsg(buffer, len) fills a fresh buffer of the variable's
        ! declared length with the message for the recorded status, truncated
        ! or blank padded to that length by Fortran character assignment
        ! rules. The symbol is rebound to the buffer, the same way a runtime
        ! character assignment rebinds it.
        use liric_session_memory_bindings, only: emit_malloc
        use, intrinsic :: iso_c_binding, only: c_int64_t
        type(lr_operand_desc_t) :: buffer, args(2), unused
        integer :: sym, length

        call set_empty(error_msg)
        sym = find_symbol_compat(context, trim(name))
        if (sym <= 0) then
            error_msg = 'iomsg target was not declared: '//trim(name)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_CHARACTER) then
            error_msg = 'iomsg target must be a character variable: '// &
                trim(name)
            return
        end if
        length = context%symbols(sym)%character_length
        if (length <= 0) then
            error_msg = 'iomsg target must have a declared length: '//trim(name)
            return
        end if
        ! One byte more than the declared length: _ffc_iomsg writes len
        ! characters and a terminating NUL, matching the NUL-terminated
        ! buffers the compiler's character values point at.
        if (.not. emit_malloc(context%session, &
                              i64_immediate(context%session, &
                                            int(length + 1, c_int64_t)), &
                              buffer, error_msg)) return
        args(1) = buffer
        args(2) = i32_immediate(context%session, int(length, c_int64_t))
        if (.not. emit_void_call(context%session, '_ffc_iomsg', args, &
                                 error_msg)) return
        context%symbols(sym)%value = buffer
        context%symbols(sym)%has_character_value = .true.
    end procedure store_iomsg_text
    module procedure runtime_iostat_operand
        ! The runtime's record of the last I/O operation, for statements that
        ! do not have a single call whose return value is the status.
        type(lr_operand_desc_t), allocatable :: noargs(:)

        allocate (noargs(0))
        if (.not. emit_i32_call(context%session, '_ffc_iostat', noargs, &
                                status_op, error_msg)) return
        call set_empty(error_msg)
    end procedure runtime_iostat_operand
    module procedure lower_close
        ! CLOSE hands the unit number to the runtime, which owns the
        ! connection. Closing a unit that is not connected is not an error in
        ! Fortran, and the runtime reports success for it, so the guard the
        ! compiler used to emit around fclose is gone (#396).
        type(lr_operand_desc_t) :: args(2), status_op
        character(len=:), allocatable :: status_text, iostat_name, iomsg_name
        logical :: status_quoted
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%unit_spec)) then
            call unsupported_feature_error('close statement', node%line, &
                node%column, 'missing unit specifier (#247 B5c)', error_msg)
            return
        end if
        call unit_number_operand(node%unit_spec, context, args(1), error_msg)
        if (len_trim(error_msg) > 0) return
        status_text = ''
        status_quoted = .false.
        iostat_name = ''
        iomsg_name = ''
        if (allocated(node%specifiers)) then
            do i = 1, size(node%specifiers)
                if (.not. allocated(node%specifiers(i)%name)) cycle
                select case (trim(node%specifiers(i)%name))
                case ('status')
                    if (allocated(node%specifiers(i)%value)) then
                        status_text = trim(node%specifiers(i)%value)
                        if (len(status_text) >= 2) then
                            if (status_text(1:1) == "'") then
                                status_quoted = &
                                    status_text(len(status_text):len(status_text)) == "'"
                            else if (status_text(1:1) == '"') then
                                status_quoted = &
                                    status_text(len(status_text):len(status_text)) == '"'
                            end if
                            if (status_quoted) then
                                status_quoted = .true.
                                status_text = status_text(2:len(status_text) - 1)
                            end if
                        end if
                    end if
                case ('iostat')
                    if (allocated(node%specifiers(i)%value)) &
                        iostat_name = trim(node%specifiers(i)%value)
                case ('iomsg')
                    if (allocated(node%specifiers(i)%value)) &
                        iomsg_name = trim(node%specifiers(i)%value)
                end select
            end do
        end if
        call open_status_operand(context, status_text, status_quoted, args(2), &
                                 error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_i32_call(context%session, '_ffc_unit_close_status', args, &
                                status_op, error_msg)) return
        call store_io_status(context, iostat_name, iomsg_name, status_op, &
                             error_msg)
    end procedure lower_close
    module procedure unit_number_operand
        ! Resolve unit_spec (raw OPEN/CLOSE/REWIND specifier text, e.g. "u",
        ! "10", "unit=u", "(u)") to an i32 operand holding the unit number.
        !
        ! A variable unit is read at run time rather than folded to the
        ! constant it happened to hold at the point of OPEN, so a unit number
        ! computed at run time reaches the right connection (#396).
        use liric_session_memory_bindings, only: emit_i32_load
        use, intrinsic :: iso_c_binding, only: c_int64_t
        character(len=:), allocatable :: plain
        integer :: unit_number, sym, ios

        call set_empty(error_msg)
        plain = unit_spec_text(unit_spec)

        unit_number = -1
        read (plain, *, iostat=ios) unit_number
        if (ios == 0) then
            unit_op = i32_immediate(context%session, &
                                    int(unit_number, c_int64_t))
            return
        end if

        sym = find_symbol_compat(context, trim(plain))
        if (sym <= 0) then
            error_msg = 'no open unit for: '//trim(unit_spec)//' (#247 B5c)'
            return
        end if
        if (context%symbols(sym)%has_address .and. &
            context%symbols(sym)%is_reference) then
            if (.not. emit_i32_load(context%session, &
                                     context%symbols(sym)%address, unit_op, &
                                     error_msg)) return
        else
            unit_op = context%symbols(sym)%value
        end if
        call set_empty(error_msg)
    end procedure unit_number_operand
    module procedure unit_spec_text
        ! Strip the surrounding parentheses REWIND/BACKSPACE keep, take the
        ! first argument (CLOSE/REWIND carry trailing keywords such as
        ! status="delete"), and drop a leading unit= keyword.
        character(len=:), allocatable :: head
        integer :: cpos, eq_pos

        plain = trim(adjustl(unit_spec))
        if (len(plain) >= 2) then
            if (plain(1:1) == '(' .and. plain(len(plain):len(plain)) == ')') then
                plain = trim(adjustl(plain(2:len(plain) - 1)))
            end if
        end if
        cpos = index(plain, ',')
        if (cpos > 0) then
            head = trim(adjustl(plain(1:cpos - 1)))
        else
            head = trim(plain)
        end if
        eq_pos = index(head, '=')
        if (eq_pos > 0) head = adjustl(head(eq_pos + 1:))
        plain = trim(head)
    end procedure unit_spec_text
    module procedure load_unit_file_ptr
        ! The stream behind a unit, from the runtime. An unopened numeric unit
        ! is connected to fort.<N> there, on first use, so implicit
        ! preconnection no longer needs its own emitted fopen (#396).
        type(lr_operand_desc_t) :: args(1)

        call unit_number_operand(unit_spec, context, args(1), error_msg)
        if (len_trim(error_msg) > 0) return
        if (.not. emit_ptr_call(context%session, '_ffc_unit_file', args, fp, &
                                error_msg)) return
        call set_empty(error_msg)
    end procedure load_unit_file_ptr
    module procedure is_file_unit_write
        integer :: sym, ios, unit_number

        result_value = .false.
        if (.not. allocated(node%unit_spec)) return
        if (trim(node%unit_spec) == '*' .or. &
            trim(node%unit_spec) == '6') return
        read (node%unit_spec, *, iostat=ios) unit_number
        if (ios == 0) then
            result_value = .true.
            return
        end if
        sym = find_symbol_compat(context, trim(node%unit_spec))
        if (sym <= 0) return
        result_value = context%symbols(sym)%is_file_unit .or. &
                             context%symbols(sym)%has_unit_const
    end procedure is_file_unit_write
    module procedure unit_is_unformatted
        integer :: sym, unit_number, ios

        result_value = .false.
        if (.not. allocated(node%unit_spec)) return
        read (node%unit_spec, *, iostat=ios) unit_number
        if (ios == 0) then
            sym = find_symbol_compat(context, file_unit_pseudo_name(unit_number))
        else
            sym = find_symbol_compat(context, trim(node%unit_spec))
        end if
        if (sym > 0) result_value = context%symbols(sym)%is_unformatted
    end procedure unit_is_unformatted
    module procedure lower_write_file
        type(lr_operand_desc_t) :: fp
        logical :: list_dir, unformatted
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

        list_dir = .not. allocated(node%format_spec)
        if (allocated(node%format_spec)) then
            if (trim(node%format_spec) == '*') list_dir = .true.
        end if

        unformatted = list_dir .and. unit_is_unformatted(node, context)
        if (unformatted) then
            call lower_file_write_unformatted(arena, node, fp, context, &
                                              error_msg)
        else if (list_dir) then
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
        if (len_trim(error_msg) > 0) return
        call store_write_iostat_success(node, context, error_msg)
    end procedure lower_write_file
    module procedure lower_file_write_unformatted
        integer :: i

        call set_empty(error_msg)
        if (.not. allocated(node%arg_indices)) return
        do i = 1, size(node%arg_indices)
            call lower_file_write_unformatted_item(arena, node%arg_indices(i), &
                                                    fp, context, error_msg)
            if (len_trim(error_msg) > 0) return
        end do
    end procedure lower_file_write_unformatted
    module procedure lower_file_write_unformatted_item
        type(lr_operand_desc_t) :: value, status, args(2)
        type(lr_operand_desc_t) :: narrow_value
        integer :: vk
        integer :: logical_bytes

        call set_empty(error_msg)
        vk = file_write_value_kind(arena, node_index, context)
        select case (vk)
        case (VALUE_I8)
            call lower_i8_expression(arena, node_index, context, value, error_msg)
        case (VALUE_I16)
            call lower_i16_expression(arena, node_index, context, value, error_msg)
        case (VALUE_I32, VALUE_LOGICAL)
            call lower_i32_expression(arena, node_index, context, value, error_msg)
        case (VALUE_I64)
            call lower_i64_expression(arena, node_index, context, value, error_msg)
        case default
            call unsupported_feature_error('unformatted write', &
                get_node_line(arena, node_index), get_node_column(arena, node_index), &
                'only integer and logical scalar values are supported', &
                error_msg)
            return
        end select
        if (len_trim(error_msg) > 0) return

        args(1) = fp
        args(2) = value
        select case (vk)
        case (VALUE_I8)
            if (.not. emit_i32_call(context%session, &
                    '_ffc_write_unformatted_i8', args, status, error_msg)) return
        case (VALUE_I16)
            if (.not. emit_i32_call(context%session, &
                    '_ffc_write_unformatted_i16', args, status, error_msg)) return
        case (VALUE_I32, VALUE_LOGICAL)
            if (vk == VALUE_LOGICAL) then
                logical_bytes = logical_write_kind_bytes(arena, node_index, context)
                select case (logical_bytes)
                case (1)
                    if (.not. emit_liric_i32_to_i8(context%session, value, &
                                                   narrow_value, error_msg)) return
                    args(2) = narrow_value
                    if (.not. emit_i32_call(context%session, &
                            '_ffc_write_unformatted_i8', args, status, error_msg)) return
                case (2)
                    if (.not. emit_liric_i32_to_i16(context%session, value, &
                                                    narrow_value, error_msg)) return
                    args(2) = narrow_value
                    if (.not. emit_i32_call(context%session, &
                            '_ffc_write_unformatted_i16', args, status, error_msg)) return
                case (8)
                    if (.not. emit_liric_i32_to_i64(context%session, value, &
                                                   narrow_value, error_msg)) return
                    args(2) = narrow_value
                    if (.not. emit_i32_call(context%session, &
                            '_ffc_write_unformatted_i64', args, status, error_msg)) return
                case default
                    args(2) = value
                    if (.not. emit_i32_call(context%session, &
                            '_ffc_write_unformatted_i32', args, status, error_msg)) return
                end select
            else
                args(2) = value
                if (.not. emit_i32_call(context%session, &
                        '_ffc_write_unformatted_i32', args, status, error_msg)) return
            end if
        case (VALUE_I64)
            args(2) = value
            if (.not. emit_i32_call(context%session, &
                    '_ffc_write_unformatted_i64', args, status, error_msg)) return
        end select
    end procedure lower_file_write_unformatted_item
    module procedure logical_write_kind_bytes
        character(len=:), allocatable :: name, name_error
        integer :: symbol_index

        bytes = 4
        if (.not. is_identifier(arena, node_index)) return
        call get_identifier_name(arena, node_index, name, name_error)
        if (len_trim(name_error) > 0) return
        symbol_index = resolve_symbol_at_node(context, node_index, name)
        if (symbol_index <= 0) return
        if (context%symbols(symbol_index)%logical_kind_bytes > 0) then
            bytes = context%symbols(symbol_index)%logical_kind_bytes
        end if
    end procedure logical_write_kind_bytes
    module procedure store_write_iostat_success
        ! Report a file WRITE's outcome. The status comes from the runtime's
        ! record of the operation rather than an assumed 0, so a write to an
        ! unusable unit is visible to the program (#427). The specifier names
        ! live in the io_control_list text (iostat=<name>, iomsg=<name>).
        character(len=:), allocatable :: iostat_name, iomsg_name
        type(lr_operand_desc_t) :: status_op

        call set_empty(error_msg)
        if (.not. allocated(node%io_control_list)) return
        call io_control_value(node%io_control_list, 'iostat', iostat_name)
        call io_control_value(node%io_control_list, 'iomsg', iomsg_name)
        if (len_trim(iostat_name) == 0 .and. len_trim(iomsg_name) == 0) return
        status_op = runtime_iostat_operand(context, error_msg)
        if (len_trim(error_msg) > 0) return
        call store_io_status(context, iostat_name, iomsg_name, status_op, &
                             error_msg)
    end procedure store_write_iostat_success
    module procedure file_write_value_kind

        if (is_character_operand(arena, node_index, context)) then
        result_value = VALUE_CHARACTER
            return
        end if
        result_value = expression_value_kind(arena, node_index, context, &
                                                       VALUE_I32)
    end procedure file_write_value_kind
    module procedure lower_file_write_item
        use, intrinsic :: iso_c_binding, only: c_int64_t
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
        fgn = ffc_unit_global_name( &
            context, 'fprintf.', context%string_literal_count)
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
    end procedure lower_file_write_item
    module procedure lower_file_write_newline
        type(lr_operand_desc_t) :: fmtop, fa(2)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn

        context%string_literal_count = context%string_literal_count + 1
        fgn = ffc_unit_global_name( &
            context, 'fprintf.nl.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(fgn), &
                                          achar(10), fgid, error_msg)
        if (len_trim(error_msg) > 0) return
        fmtop = printf_format_ptr(context%session, fgid)
        fa(1) = fp
        fa(2) = fmtop
        if (.not. emit_fprintf(context%session, fa, error_msg)) return
        call set_empty(error_msg)
    end procedure lower_file_write_newline
    module procedure lower_file_write_formatted
        use, intrinsic :: iso_c_binding, only: c_int64_t
        character(len=:), allocatable :: fmt_body, c_fmt
        type(lr_operand_desc_t) :: fmtop, char_len
        type(lr_operand_desc_t), allocatable :: fa(:)
        integer(c_int32_t) :: fgid
        character(len=64) :: fgn
        integer :: i

        call set_empty(error_msg)
        call normalize_format_body(node%format_spec, fmt_body)
        if (.not. fortran_fmt_to_c(trim(fmt_body), c_fmt, error_msg)) return

        context%string_literal_count = context%string_literal_count + 1
        fgn = ffc_unit_global_name( &
            context, 'fprintf.fmt.', context%string_literal_count)
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
            case (VALUE_CHARACTER)
                call char_expr_operands(arena, node%arg_indices(i), context, &
                                        fa(i + 2), char_len, error_msg)
            case default
                call lower_i32_expression(arena, node%arg_indices(i), context, &
                                           fa(i + 2), error_msg)
            end select
            if (len_trim(error_msg) > 0) return
        end do
        if (.not. emit_fprintf(context%session, fa, error_msg)) return
        call set_empty(error_msg)
    end procedure lower_file_write_formatted
    module procedure fortran_fmt_to_c
        integer :: i, w, d
        character :: ch
        character(len=32) :: buf

        result_value = .false.
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
        result_value = .true.
    end procedure fortran_fmt_to_c
    module procedure read_fmt_int
        val = 0
        do while (pos <= len(s))
            if (s(pos:pos) >= '0' .and. s(pos:pos) <= '9') then
                val = val * 10 + ichar(s(pos:pos)) - ichar('0')
                pos = pos + 1
            else
                exit
            end if
        end do
    end procedure read_fmt_int
end submodule session_program_lowering_open_close
