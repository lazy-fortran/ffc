    ! Minimal INQUIRE lowering (I/O cluster W10): exist=, opened=, and
    ! iostat= by file= or unit=. opened= reads the compile-time is_file_unit
    ! flag set by a preceding OPEN in the same control-flow path; exist= on a
    ! file= form asks the C library with access(2).

    subroutine parse_inquire_spec(spec, unit_str, file_path, exist_var, &
                                  opened_var, size_var, iostat_var, error_msg)
        ! Extract keyword values from an INQUIRE specifier-list text, mirroring
        ! parse_open_spec's tokenizer (keywords may carry stray spaces around
        ! '=' from the parser's token-joining rules).
        character(len=*), intent(in) :: spec
        character(len=:), allocatable, intent(out) :: unit_str, file_path
        character(len=:), allocatable, intent(out) :: exist_var, opened_var
        character(len=:), allocatable, intent(out) :: size_var, iostat_var
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: text, kw, val
        integer :: p, eq_pos, cp, qend, arg_end
        logical :: first_arg
        character :: qc

        call set_empty(error_msg)
        unit_str = ''
        file_path = ''
        exist_var = ''
        opened_var = ''
        size_var = ''
        iostat_var = ''
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
            kw = spec_lower(trim(text(p:eq_pos - 1)))
            p = eq_pos + 1
            do while (p <= len(text) .and. text(p:p) == ' ')
                p = p + 1
            end do
            if (p > len(text)) exit
            if (text(p:p) == "'" .or. text(p:p) == '"') then
                qc = text(p:p)
                qend = index(text(p + 1:), qc)
                if (qend <= 0) then
                    error_msg = 'unterminated string in INQUIRE spec'
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
            case ('file')
                file_path = trim(val)
            case ('exist')
                exist_var = trim(val)
            case ('opened')
                opened_var = trim(val)
            case ('size')
                size_var = trim(val)
            case ('iostat')
                iostat_var = trim(val)
            end select
        end do
    end subroutine parse_inquire_spec

    subroutine lower_inquire(node, context, error_msg)
        type(inquire_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: unit_str, file_path, exist_var
        character(len=:), allocatable :: opened_var, size_var, iostat_var

        call set_empty(error_msg)
        if (.not. allocated(node%spec_list)) then
            call unsupported_feature_error('inquire statement', node%line, &
                node%column, 'missing specifier list', error_msg)
            return
        end if
        call parse_inquire_spec(node%spec_list, unit_str, file_path, &
                                exist_var, opened_var, size_var, iostat_var, &
                                error_msg)
        if (len_trim(error_msg) > 0) return

        if (len_trim(size_var) > 0) then
            call unsupported_feature_error('inquire statement', node%line, &
                node%column, 'size= is not supported (#247)', error_msg)
            return
        end if

        if (len_trim(file_path) > 0) then
            call lower_inquire_by_file(node, context, file_path, exist_var, &
                                       iostat_var, error_msg)
            return
        end if

        if (len_trim(unit_str) > 0) then
            call lower_inquire_by_unit(node, context, unit_str, opened_var, &
                                       exist_var, iostat_var, error_msg)
            return
        end if

        call unsupported_feature_error('inquire statement', node%line, &
            node%column, 'inquire requires file= or unit= (#247)', error_msg)
    end subroutine lower_inquire

    subroutine store_inquire_logical(context, node, var_name, kind_label, &
                                     value, error_msg)
        ! Resolve var_name to a declared logical (or plain integer) scalar and
        ! store value into it, matching the storage convention iostat= uses.
        type(lowering_context_t), intent(inout) :: context
        type(inquire_statement_node), intent(in) :: node
        character(len=*), intent(in) :: var_name, kind_label
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym

        call set_empty(error_msg)
        sym = find_symbol(context, var_name)
        if (sym <= 0) then
            error_msg = 'inquire '//trim(kind_label)//'= target was not '// &
                        'declared: '//trim(var_name)
            return
        end if
        if (context%symbols(sym)%value_kind /= VALUE_LOGICAL .and. &
            context%symbols(sym)%value_kind /= VALUE_I32) then
            call unsupported_feature_error('inquire statement', node%line, &
                node%column, trim(kind_label)//'= target must be a '// &
                'logical or default integer scalar', error_msg)
            return
        end if
        call assign_i32_to_symbol(context, sym, value, error_msg)
    end subroutine store_inquire_logical

    subroutine lower_inquire_by_file(node, context, file_path, exist_var, &
                                     iostat_var, error_msg)
        ! exist= probes the filesystem with a guarded fopen(path, "r")/fclose
        ! rather than access(2): repeated OPEN statements already prove fopen
        ! is safe to call more than once per program. The exists/not-exists
        ! outcome is decided by branching on the fopen result and storing a
        ! literal 1/0 constant on each side (store_inquire_bool_via_branch's
        ! pattern): storing the icmp result to a symbol directly, repeated
        ! across two inquire statements, collapses onto one shared comparison
        ! result in the LIRIC backend.
        type(inquire_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: file_path, exist_var, iostat_var
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: path_id, mode_id
        character(len=64) :: path_name, mode_name
        type(lr_operand_desc_t) :: path_ptr, mode_ptr, fp, is_open
        type(lr_operand_desc_t) :: args(2)
        integer(c_int32_t) :: close_block, false_block, done_block
        integer :: sym

        call set_empty(error_msg)
        context%string_literal_count = context%string_literal_count + 1
        path_name = ffc_unit_global_name( &
            context, 'inq.file.', context%string_literal_count)
        call create_printf_format_global(context%session, trim(path_name), &
                                         trim(file_path), path_id, error_msg)
        if (len_trim(error_msg) > 0) return
        path_ptr = printf_format_ptr(context%session, path_id)

        if (len_trim(exist_var) > 0) then
            sym = find_symbol(context, exist_var)
            if (sym <= 0) then
                error_msg = 'inquire exist= target was not declared: '// &
                            trim(exist_var)
                return
            end if
            if (context%symbols(sym)%value_kind /= VALUE_LOGICAL .and. &
                context%symbols(sym)%value_kind /= VALUE_I32) then
                call unsupported_feature_error('inquire statement', node%line, &
                    node%column, 'exist= target must be a logical or '// &
                    'default integer scalar', error_msg)
                return
            end if

            context%string_literal_count = context%string_literal_count + 1
            mode_name = ffc_unit_global_name( &
                context, 'inq.mode.', context%string_literal_count)
            call create_printf_format_global(context%session, trim(mode_name), &
                                             'r', mode_id, error_msg)
            if (len_trim(error_msg) > 0) return
            mode_ptr = printf_format_ptr(context%session, mode_id)

            args(1) = path_ptr
            args(2) = mode_ptr
            if (.not. emit_ptr_call(context%session, 'fopen', args, fp, &
                                    error_msg)) return
            if (.not. emit_liric_i32_icmp(context%session, LR_CMP_NE, fp, &
                    null_ptr_operand(context), is_open, error_msg)) return

            close_block = create_liric_block(context%session)
            false_block = create_liric_block(context%session)
            done_block = create_liric_block(context%session)
            if (.not. emit_liric_condbr(context%session, is_open, close_block, &
                                        false_block, error_msg)) return

            if (.not. set_liric_block(context%session, close_block, &
                                      error_msg)) return
            if (.not. emit_void_call(context%session, 'fclose', [fp], &
                                     error_msg)) return
            call assign_i32_to_symbol(context, sym, &
                i32_immediate(context%session, 1_c_int64_t), error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_br(context%session, done_block, error_msg)) &
                return

            if (.not. set_liric_block(context%session, false_block, &
                                      error_msg)) return
            call assign_i32_to_symbol(context, sym, &
                i32_immediate(context%session, 0_c_int64_t), error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_liric_br(context%session, done_block, error_msg)) &
                return

            if (.not. set_liric_block(context%session, done_block, &
                                      error_msg)) return
            context%current_block_id = done_block
            context%current_block_terminated = .false.
            call reload_i32_symbol(context, sym, error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (len_trim(iostat_var) > 0) then
            call store_inquire_logical(context, node, iostat_var, 'iostat', &
                i32_immediate(context%session, 0_c_int64_t), error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call set_empty(error_msg)
    end subroutine lower_inquire_by_file

    subroutine lower_inquire_by_unit(node, context, unit_str, opened_var, &
                                     exist_var, iostat_var, error_msg)
        type(inquire_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: unit_str, opened_var, exist_var
        character(len=*), intent(in) :: iostat_var
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: sym, ios, unit_number
        logical :: is_opened
        character(len=24) :: pseudo_name

        call set_empty(error_msg)
        is_opened = .false.
        read (unit_str, *, iostat=ios) unit_number
        if (ios == 0) then
            write (pseudo_name, '(A,I0)') '.unit.', unit_number
            sym = find_symbol(context, trim(pseudo_name))
        else
            sym = find_symbol(context, trim(unit_str))
        end if
        if (sym > 0) is_opened = context%symbols(sym)%is_file_unit

        if (len_trim(opened_var) > 0) then
            call store_inquire_logical(context, node, opened_var, 'opened', &
                i32_immediate(context%session, &
                              merge(1_c_int64_t, 0_c_int64_t, is_opened)), &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (len_trim(exist_var) > 0) then
            call store_inquire_logical(context, node, exist_var, 'exist', &
                i32_immediate(context%session, 1_c_int64_t), error_msg)
            if (len_trim(error_msg) > 0) return
        end if

        if (len_trim(iostat_var) > 0) then
            call store_inquire_logical(context, node, iostat_var, 'iostat', &
                i32_immediate(context%session, 0_c_int64_t), error_msg)
            if (len_trim(error_msg) > 0) return
        end if
        call set_empty(error_msg)
    end subroutine lower_inquire_by_unit
