submodule (session_program_lowering_impl) session_program_lowering_transfer
    implicit none
contains
    ! TRANSFER(source, mold) for scalar intrinsic types of matching byte size:
    ! integer(4)<->real(4), integer(8)<->real(8), and canonical logical
    ! values. The result reinterprets
    ! source's bit pattern as mold's type, done here by storing through a
    ! typed stack slot and loading the same address back with the other
    ! type - the same memory-punning trick EQUIVALENCE already relies on.
    !
    ! Scope: two-argument scalar TRANSFER (no SIZE argument, no array mold),
    ! source drawn from an identifier, a declared/allocatable array element,
    ! or a literal. Any other form declines (handled=.false., error_msg
    ! empty) so the caller falls back to its normal unsupported-feature
    ! diagnostic.

    module procedure lower_transfer_intrinsic
    integer :: source_kind
    type(lr_operand_desc_t) :: source_value, address

    handled = .false.
    call set_empty(error_msg)
    if (.not. allocated(node%arg_indices)) return
    if (size(node%arg_indices) /= 2) return

    source_kind = transfer_operand_kind(arena, node%arg_indices(1), context)
    if (source_kind < 0) return
    if (.not. transfer_pair_supported(source_kind, target_kind)) return

    call lower_transfer_source_element(arena, node%arg_indices(1), context, &
        source_kind, 0_c_int64_t, &
        source_value, error_msg)
    if (len_trim(error_msg) > 0) return

    if (source_kind == target_kind) then
        value = source_value
        handled = .true.
        return
    end if

    if (.not. transfer_alloca(context, source_kind, address, error_msg)) &
        return
    if (.not. transfer_store(context, source_kind, source_value, address, &
        error_msg)) return
    if (.not. transfer_load(context, target_kind, address, value, &
        error_msg)) return
    handled = .true.
    end procedure lower_transfer_intrinsic

    ! Produce one source element for TRANSFER: element linear_index of a
    ! whole-array source identifier, or the scalar expression itself.
    module procedure lower_transfer_source_element
    integer :: source_symbol

    call set_empty(error_msg)
    source_symbol = transfer_array_source_symbol(arena, node_index, context)
    if (source_symbol > 0) then
        call load_array_linear_element(context, source_symbol, linear_index, &
            value, error_msg)
        return
    end if

    select case (source_kind)
    case (VALUE_I32)
        call lower_i32_expression(arena, node_index, context, value, error_msg)
    case (VALUE_LOGICAL)
        call lower_i32_expression(arena, node_index, context, value, error_msg)
    case (VALUE_I64)
        call lower_i64_expression(arena, node_index, context, value, error_msg)
    case (VALUE_F32)
        call lower_f32_expression(arena, node_index, context, value, error_msg)
    case default
        call lower_f64_expression(arena, node_index, context, value, error_msg)
    end select
    end procedure lower_transfer_source_element

    ! Symbol index of a whole-array identifier usable as a TRANSFER source,
    ! or 0 when the argument is not such an array.
    module procedure transfer_array_source_symbol
    character(len=:), allocatable :: id_name, id_err
    integer :: candidate

    symbol_index = 0
    if (.not. node_exists(arena, node_index)) return
    if (.not. is_identifier(arena, node_index)) return
    call get_identifier_name(arena, node_index, id_name, id_err)
    if (len_trim(id_err) > 0) return
    candidate = find_symbol_compat(context, id_name)
    if (candidate <= 0) return
    if (.not. context%symbols(candidate)%is_array) return
    if (context%symbols(candidate)%is_runtime_array) return
    if (context%symbols(candidate)%array_size <= 0) return
    symbol_index = candidate
    end procedure transfer_array_source_symbol

    module procedure transfer_operand_kind
    integer :: symbol_index
    character(len=:), allocatable :: id_name, id_err
    character(len=:), allocatable :: lv, lt, le

    kind = -1
    if (.not. node_exists(arena, node_index)) return

    if (is_identifier(arena, node_index)) then
        call get_identifier_name(arena, node_index, id_name, id_err)
        if (len_trim(id_err) > 0) return
        symbol_index = find_symbol_compat(context, id_name)
        if (symbol_index > 0) then
            if (.not. context%symbols(symbol_index)%is_array) then
                kind = context%symbols(symbol_index)%value_kind
            else if (transfer_array_source_symbol(arena, node_index, &
                    context) > 0) then
                ! A whole-array source contributes its element bits in
                ! array-element order, starting with the first element.
                kind = context%symbols(symbol_index)%value_kind
            end if
        end if
        return
    end if

    if (is_real_literal(arena, node_index)) then
        call get_literal_info(arena, node_index, lv, lt, le)
        if (len_trim(le) > 0) return
        if (literal_is_f64(lv, context, node_index)) then
            kind = VALUE_F64
        else
            kind = VALUE_F32
        end if
        return
    end if

    if (is_logical_literal(arena, node_index)) then
        kind = VALUE_LOGICAL
        return
    end if

    if (is_literal(arena, node_index)) then
        kind = VALUE_I32
        return
    end if

    select type (node => arena%entries(node_index)%node)
        type is (call_or_subscript_node)
        if (node%base_expr_index > 0) return
        if (.not. allocated(node%name)) return
        if (node%is_array_access .or. &
            is_declared_array_element_ref(node, context) .or. &
            is_allocatable_element_ref(node, context)) then
            symbol_index = find_symbol_compat(context, node%name)
            if (symbol_index > 0) &
                kind = context%symbols(symbol_index)%value_kind
        end if
    end select
    end procedure transfer_operand_kind

    module procedure transfer_operand_bytes
    integer :: symbol_index
    character(len=:), allocatable :: name, name_error

    bytes = transfer_kind_bytes(value_kind)
    if (value_kind /= VALUE_LOGICAL) return
    if (.not. is_identifier(arena, node_index)) return
    call get_identifier_name(arena, node_index, name, name_error)
    if (len_trim(name_error) > 0) return
    symbol_index = find_symbol_compat(context, name)
    if (symbol_index > 0) then
        if (context%symbols(symbol_index)%logical_kind_bytes > 0) then
            bytes = context%symbols(symbol_index)%logical_kind_bytes
        end if
    end if
    end procedure transfer_operand_bytes

    module procedure transfer_kind_bytes
    select case (value_kind)
    case (VALUE_I8)
        bytes = 1
    case (VALUE_I16)
        bytes = 2
    case (VALUE_I32, VALUE_LOGICAL, VALUE_F32)
        bytes = 4
    case (VALUE_I64, VALUE_F64)
        bytes = 8
    case default
        bytes = 0
    end select
    end procedure transfer_kind_bytes

    module procedure transfer_pair_supported
    ok = .false.
    if (source_kind == target_kind) then
        select case (source_kind)
        case (VALUE_I32, VALUE_I64, VALUE_F32, VALUE_F64)
            ok = .true.
        end select
        return
    end if
    ok = (source_kind == VALUE_I32 .and. target_kind == VALUE_F32) .or. &
        (source_kind == VALUE_F32 .and. target_kind == VALUE_I32) .or. &
        (source_kind == VALUE_I64 .and. target_kind == VALUE_F64) .or. &
        (source_kind == VALUE_F64 .and. target_kind == VALUE_I64)
    end procedure transfer_pair_supported

    module procedure transfer_alloca
    select case (kind)
    case (VALUE_I32)
        ok = emit_i32_alloca(context%session, address, error_msg)
    case (VALUE_I64)
        ok = emit_i64_alloca(context%session, address, error_msg)
    case (VALUE_F32)
        ok = emit_liric_f32_alloca(context%session, address, error_msg)
    case default
        ok = emit_liric_f64_alloca(context%session, address, error_msg)
    end select
    end procedure transfer_alloca

    module procedure transfer_store
    select case (kind)
    case (VALUE_I32)
        ok = emit_i32_store(context%session, value, address, error_msg)
    case (VALUE_I64)
        ok = emit_i64_store(context%session, value, address, error_msg)
    case (VALUE_F32)
        ok = emit_liric_f32_store(context%session, value, address, &
            error_msg)
    case default
        ok = emit_liric_f64_store(context%session, value, address, &
            error_msg)
    end select
    end procedure transfer_store

    module procedure transfer_load
    select case (kind)
    case (VALUE_I32)
        ok = emit_i32_load(context%session, address, value, error_msg)
    case (VALUE_I64)
        ok = emit_i64_load(context%session, address, value, error_msg)
    case (VALUE_F32)
        ok = emit_liric_f32_load(context%session, address, value, error_msg)
    case default
        ok = emit_liric_f64_load(context%session, address, value, error_msg)
    end select
    end procedure transfer_load

    ! Whole-array TRANSFER(source, mold [, size]) assigned to an array target.
    ! Scope: source and result element kinds share a byte size (integer(4)
    ! <-> real(4), integer(8) <-> real(8)); the source is a scalar expression
    ! or a declared whole array; SIZE, when present, is a compile-time
    ! non-negative constant. Every other form reports a diagnostic.
    module procedure lower_transfer_array_assignment
    integer :: source_kind, target_kind, source_symbol
    integer :: source_bytes, target_bytes
    integer :: arg_count, source_count, result_count, i
    integer(c_int64_t) :: size_value
    type(lr_operand_desc_t) :: element, converted, address
    logical :: needs_pun

    call set_empty(error_msg)
    target_kind = context%symbols(symbol_index)%value_kind

    select type (rhs => arena%entries(node%value_index)%node)
        type is (call_or_subscript_node)
        if (.not. allocated(rhs%arg_indices)) then
            error_msg = 'transfer requires a source and a mold argument'
            return
        end if
        arg_count = size(rhs%arg_indices)
        if (arg_count < 2 .or. arg_count > 3) then
            error_msg = 'transfer requires two or three arguments'
            return
        end if

        source_kind = transfer_operand_kind(arena, rhs%arg_indices(1), context)
        if (source_kind < 0) then
            error_msg = 'transfer source must be a scalar or whole-array '// &
                'entity of a supported intrinsic type'
            return
        end if
        source_symbol = transfer_array_source_symbol(arena, rhs%arg_indices(1), &
            context)
        source_bytes = transfer_operand_bytes(arena, rhs%arg_indices(1), &
            source_kind, context)
        target_bytes = transfer_kind_bytes(target_kind)
        if (source_kind == VALUE_LOGICAL) then
            if (target_kind /= VALUE_I8 .and. target_kind /= VALUE_I16 .and. &
                target_kind /= VALUE_I32 .and. target_kind /= VALUE_I64) then
                error_msg = 'logical transfer target must be an integer'
                return
            end if
            if (target_bytes <= 0 .or. source_bytes < target_bytes .or. &
                mod(source_bytes, target_bytes) /= 0) then
                error_msg = 'transfer requires source and result element kinds '// &
                    'of the same byte size'
                return
            end if
            if (source_symbol > 0 .and. target_bytes /= source_bytes) then
                error_msg = 'array logical transfer requires matching element size'
                return
            end if
        else if (.not. transfer_pair_supported(source_kind, target_kind)) then
            error_msg = 'transfer requires source and result element kinds '// &
                'of the same byte size'
            return
        end if
        source_count = 1
        if (source_symbol > 0) then
            source_count = context%symbols(source_symbol)%array_size
        else if (source_kind == VALUE_LOGICAL) then
            source_count = source_bytes / target_bytes
        end if

        result_count = context%symbols(symbol_index)%array_size
        if (arg_count == 3) then
            call eval_i32_constant(arena, rhs%arg_indices(3), context, &
                size_value, error_msg)
            if (len_trim(error_msg) > 0) then
                error_msg = 'transfer size must be a compile-time constant'
                return
            end if
            if (size_value < 0_c_int64_t) then
                error_msg = 'transfer size must not be negative'
                return
            end if
            if (size_value /= int(result_count, c_int64_t)) then
                error_msg = 'transfer size does not match the result array size'
                return
            end if
        end if
        if (result_count > source_count) then
            error_msg = 'transfer source supplies fewer elements than the '// &
                'result requires'
            return
        end if

        needs_pun = source_kind /= target_kind .and. source_kind /= VALUE_LOGICAL
        if (needs_pun) then
            if (.not. transfer_alloca(context, source_kind, address, &
                error_msg)) return
        end if

        do i = 0, result_count - 1
            call lower_transfer_source_element(arena, rhs%arg_indices(1), &
                context, source_kind, &
                int(i, c_int64_t), element, &
                error_msg)
            if (len_trim(error_msg) > 0) return
            converted = element
            if (source_kind == VALUE_LOGICAL) then
                ! FFC normalizes logical truth to one in its i32 semantic
                ! slot. For a scalar logical transferred into narrower
                ! integer elements, only the low byte is nonzero.
                if (source_symbol == 0 .and. i > 0) then
                    element = i32_immediate(context%session, 0_c_int64_t)
                end if
                select case (target_kind)
                case (VALUE_I8)
                    if (.not. emit_liric_i32_to_i8(context%session, element, &
                        converted, error_msg)) return
                case (VALUE_I16)
                    if (.not. emit_liric_i32_to_i16(context%session, element, &
                        converted, error_msg)) return
                case (VALUE_I64)
                    if (.not. emit_liric_i32_to_i64(context%session, element, &
                        converted, error_msg)) return
                case default
                    converted = element
                end select
            else if (needs_pun) then
                if (.not. transfer_store(context, source_kind, element, &
                    address, error_msg)) return
                if (.not. transfer_load(context, target_kind, address, &
                    converted, error_msg)) return
            end if
            call store_array_linear_element(context, symbol_index, &
                int(i, c_int64_t), converted, &
                error_msg)
            if (len_trim(error_msg) > 0) return
        end do
        call set_empty(error_msg)
    class default
        error_msg = 'transfer requires an intrinsic call'
    end select
    end procedure lower_transfer_array_assignment
end submodule session_program_lowering_transfer
