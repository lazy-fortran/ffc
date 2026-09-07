    ! INQUIRE lowering (I/O cluster W10): exist=, opened=, size=, and
    ! iostat= by file= or unit=. opened= reads the compile-time is_file_unit
    ! flag set by a preceding OPEN in the same control-flow path; size= is
    ! delegated to the runtime so closed files and connected units share one
    ! byte-size contract.

    subroutine parse_inquire_spec(specifiers, unit_str, file_node, exist_var, &
                                  opened_var, size_var, iostat_var, error_msg)
        type(io_specifier_t), intent(in) :: specifiers(:)
        character(len=:), allocatable, intent(out) :: unit_str, exist_var
        character(len=:), allocatable, intent(out) :: opened_var, size_var
        character(len=:), allocatable, intent(out) :: iostat_var, error_msg
        integer, intent(out) :: file_node
        integer :: i

        call set_empty(error_msg)
        unit_str = ''
        file_node = 0
        exist_var = ''
        opened_var = ''
        size_var = ''
        iostat_var = ''
        do i = 1, size(specifiers)
            if (.not. allocated(specifiers(i)%name)) cycle
            if (.not. allocated(specifiers(i)%value)) cycle
            select case (trim(spec_lower(specifiers(i)%name)))
            case ('unit')
                unit_str = trim(specifiers(i)%value)
            case ('file')
                file_node = specifiers(i)%value_node_index
                if (file_node <= 0) then
                    error_msg = 'inquire file= requires character expression metadata'
                    return
                end if
            case ('exist')
                exist_var = trim(specifiers(i)%value)
            case ('opened')
                opened_var = trim(specifiers(i)%value)
            case ('size')
                size_var = trim(specifiers(i)%value)
            case ('iostat')
                iostat_var = trim(specifiers(i)%value)
            end select
        end do
    end subroutine parse_inquire_spec

    subroutine lower_inquire(arena, node, context, error_msg)
        type(ast_arena_t), intent(in) :: arena
        type(inquire_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: unit_str, exist_var
        character(len=:), allocatable :: opened_var, size_var, iostat_var
        integer :: file_node
        type(lr_operand_desc_t) :: path_ptr

        call set_empty(error_msg)
        if (.not. allocated(node%specifiers)) then
            call unsupported_feature_error('inquire statement', node%line, &
                node%column, 'missing specifier list', error_msg)
            return
        end if
        call parse_inquire_spec(node%specifiers, unit_str, file_node, &
                                exist_var, opened_var, size_var, iostat_var, &
                                error_msg)
        if (len_trim(error_msg) > 0) return

        if (file_node > 0) then
            call lower_inquire_file_path(arena, file_node, context, path_ptr, error_msg)
            if (len_trim(error_msg) > 0) return
            call lower_inquire_by_file(node, context, path_ptr, exist_var, &
                                       size_var, iostat_var, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. emit_free(context%session, path_ptr, error_msg)) return
            return
        end if

        if (len_trim(unit_str) > 0) then
            call lower_inquire_by_unit(node, context, unit_str, opened_var, &
                                       exist_var, size_var, iostat_var, error_msg)
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
        sym = find_symbol_compat(context, var_name)
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

    subroutine store_inquire_size(context, node, var_name, value, error_msg)
        ! SIZE= is a default integer in ordinary code, but accepting integer(8)
        ! here also covers the common large-file spelling without losing the
        ! runtime's 64-bit result before storage.
        type(lowering_context_t), intent(inout) :: context
        type(inquire_statement_node), intent(in) :: node
        character(len=*), intent(in) :: var_name
        type(lr_operand_desc_t), intent(in) :: value
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: converted_value
        integer :: sym

        call set_empty(error_msg)
        sym = find_symbol_compat(context, var_name)
        if (sym <= 0) then
            error_msg = 'inquire size= target was not declared: '//trim(var_name)
            return
        end if
        select case (context%symbols(sym)%value_kind)
        case (VALUE_I32)
            if (.not. emit_liric_i64_to_i32(context%session, value, converted_value, &
                                            error_msg)) return
            call assign_i32_to_symbol(context, sym, converted_value, error_msg)
        case (VALUE_I64)
            context%symbols(sym)%value = value
            if (.not. context%symbols(sym)%has_address) then
                if (.not. emit_i64_alloca(context%session, &
                                          context%symbols(sym)%address, &
                                          error_msg)) return
                context%symbols(sym)%has_address = .true.
                context%symbols(sym)%is_reference = .true.
            end if
            if (.not. emit_i64_store(context%session, value, &
                                     context%symbols(sym)%address, error_msg)) return
        case default
            call unsupported_feature_error('inquire statement', node%line, &
                node%column, 'size= target must be an integer scalar', error_msg)
        end select
    end subroutine store_inquire_size

    subroutine lower_inquire_file_path(arena, file_node, context, path_ptr, error_msg)
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: file_node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(out) :: path_ptr
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: path_data, path_len, trimmed_len

        ! FILE is evaluated once. The C APIs need an owned, terminated copy;
        ! trimming that copy must leave the Fortran expression's bytes intact.
        call char_expr_operands(arena, file_node, context, path_data, path_len, &
                                error_msg)
        if (len_trim(error_msg) > 0) return
        call compute_len_trim(context, path_data, path_len, trimmed_len, error_msg)
        if (len_trim(error_msg) > 0) return
        call materialize_character_view(context, path_data, trimmed_len, &
                                        path_ptr, error_msg)
    end subroutine lower_inquire_file_path

    subroutine lower_inquire_by_file(node, context, path_ptr, exist_var, &
                                     size_var, iostat_var, error_msg)
        ! exist= probes the filesystem with a guarded fopen(path, "r")/fclose
        ! rather than access(2). size= uses the runtime stat query so it also
        ! reports the standard -1 sentinel for a missing file.
        type(inquire_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        type(lr_operand_desc_t), intent(in) :: path_ptr
        character(len=*), intent(in) :: exist_var, size_var, iostat_var
        character(len=:), allocatable, intent(out) :: error_msg
        integer(c_int32_t) :: mode_id
        character(len=64) :: mode_name
        type(lr_operand_desc_t) :: mode_ptr, fp, is_open
        type(lr_operand_desc_t) :: args(2)
        type(lr_operand_desc_t) :: single_arg(1)
        type(lr_operand_desc_t) :: size_value
        integer(c_int32_t) :: close_block, false_block, done_block
        integer :: sym

        call set_empty(error_msg)

        if (len_trim(exist_var) > 0) then
            sym = find_symbol_compat(context, exist_var)
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
            single_arg(1) = fp
            if (.not. emit_void_call(context%session, 'fclose', single_arg, &
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

        if (len_trim(size_var) > 0) then
            single_arg(1) = path_ptr
            if (.not. emit_i64_call(context%session, '_ffc_inquire_file_size', &
                                    single_arg, size_value, error_msg)) return
            call store_inquire_size(context, node, size_var, size_value, &
                                    error_msg)
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
                                     exist_var, size_var, iostat_var, error_msg)
        type(inquire_statement_node), intent(in) :: node
        type(lowering_context_t), intent(inout) :: context
        character(len=*), intent(in) :: unit_str, opened_var, exist_var, size_var
        character(len=*), intent(in) :: iostat_var
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_operand_desc_t) :: unit_args(1), opened_op, size_value

        call set_empty(error_msg)

        if (len_trim(opened_var) > 0 .or. len_trim(size_var) > 0) then
            ! Ask the runtime, which owns the connection, instead of reading a
            ! compile-time flag that could not see a unit opened elsewhere in
            ! the program (#396).
            call unit_number_operand(unit_str, context, unit_args(1), error_msg)
            if (len_trim(error_msg) > 0) return
            if (len_trim(opened_var) > 0) then
                if (.not. emit_i32_call(context%session, '_ffc_unit_is_open', &
                                        unit_args, opened_op, error_msg)) return
                call store_inquire_logical(context, node, opened_var, 'opened', &
                                           opened_op, error_msg)
                if (len_trim(error_msg) > 0) return
            end if
            if (len_trim(size_var) > 0) then
                if (.not. emit_i64_call(context%session, '_ffc_inquire_unit_size', &
                                        unit_args, size_value, error_msg)) return
                call store_inquire_size(context, node, size_var, size_value, &
                                        error_msg)
                if (len_trim(error_msg) > 0) return
            end if
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
